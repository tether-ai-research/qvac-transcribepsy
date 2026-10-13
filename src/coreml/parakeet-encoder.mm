// Core ML FastConformer encoder sidecar for parakeet-cpp.
//
// Derived from the whisper.cpp Core ML wrapper pattern (src/coreml/whisper-encoder.mm),
// generalised to load an arbitrary compiled encoder via the generic MLModel API so it
// works with whatever `.mlmodelc` the mobius export produces (no code-generated model
// interface required). Tensor orientation is discovered from the model description, so
// the export only has to agree on dimension sizes, not their order.

#if !__has_feature(objc_arc)
#error "parakeet-encoder.mm must be compiled with -fobjc-arc"
#endif

#import "coreml/parakeet-encoder.h"
#include "coreml/parakeet_coreml_shape.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Input names the ANE exporter (scripts/export-encoder-coreml-ane.py) declares.
// Their presence marks a fixed-window ANE sidecar; any other model is bound as a
// single-input export.
NSString * const kAneMelInput      = @"mel";
NSString * const kAnePadMaskInput  = @"pad_mask";
NSString * const kAneKeyBiasInput  = @"key_bias";

// Additive attention bias for pad keys. Matches KEY_BIAS_PAD in the exporter:
// fp16 -inf would make an all-pad softmax row NaN, -1e4 saturates just as hard.
constexpr float kKeyBiasPad = -1.0e4f;

// Mel frames per encoder frame: the subsampling stack is three stride-2 convs.
constexpr int64_t kAneSubsample = 8;

float decode_float16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int      shift = 0;
            uint32_t m     = mant;
            do { m <<= 1; ++shift; } while ((m & 0x400u) == 0);
            bits = sign | ((uint32_t) (127 - 15 - shift + 1) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t encode_float16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t        exp  = (int32_t) ((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t       mant = bits & 0x7fffffu;

    if (exp >= 0x1f) return (uint16_t) (sign | 0x7c00u);            // overflow -> inf
    if (exp <= 0) {                                                  // subnormal / underflow
        if (exp < -10) return (uint16_t) sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        return (uint16_t) (sign | (mant >> shift));
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}

NSString * sole_multiarray_feature(NSDictionary<NSString *, MLFeatureDescription *> * descs) {
    NSString * found = nil;
    for (NSString * key in descs) {
        if (descs[key].type == MLFeatureTypeMultiArray) {
            if (found != nil) return nil;
            found = key;
        }
    }
    return found;
}

std::vector<int64_t> dims_of(NSArray<NSNumber *> * shape) {
    std::vector<int64_t> dims;
    if (shape != nil) {
        dims.reserve(shape.count);
        for (NSNumber * dim in shape) {
            dims.push_back(dim.longLongValue);
        }
    }
    return dims;
}

// Resolves the concrete input shape to allocate and whether it is feature-major.
// Thin Objective-C adapter over parakeet::coreml_resolve_input_dims (unit-tested in
// test-coreml-shapes): a fixed-shape model already sized to this input is honoured
// verbatim, a flexible export is rebuilt at the requested mel length in the model's
// declared orientation, and an unknown/non-concrete shape falls back to
// features-major [1, n_mels, n_mel_frames].
NSArray<NSNumber *> * resolve_input_shape(NSArray<NSNumber *> * declared,
                                          int64_t n_mel_frames, int64_t n_mels,
                                          bool * transpose) {
    const parakeet::CoremlInputDims resolved =
        parakeet::coreml_resolve_input_dims(dims_of(declared), n_mel_frames, n_mels);
    *transpose = resolved.transpose;
    NSMutableArray<NSNumber *> * shape = [NSMutableArray arrayWithCapacity:resolved.dims.size()];
    for (int64_t dim : resolved.dims) {
        [shape addObject:@(dim)];
    }
    return shape;
}

MLMultiArray * build_input_array(NSArray<NSNumber *> * shape, bool transpose,
                                 const float * mel, int64_t n_mel_frames, int64_t n_mels,
                                 NSError ** err) {
    MLMultiArray * arr = [[MLMultiArray alloc] initWithShape:shape
                                                    dataType:MLMultiArrayDataTypeFloat32
                                                       error:err];
    if (arr == nil) return nil;

    float * dst = (float *) arr.dataPointer;
    if (!transpose) {
        std::memcpy(dst, mel, (size_t) n_mel_frames * n_mels * sizeof(float));
        return arr;
    }
    for (int64_t m = 0; m < n_mels; ++m) {
        for (int64_t t = 0; t < n_mel_frames; ++t) {
            dst[m * n_mel_frames + t] = mel[t * n_mels + m];
        }
    }
    return arr;
}

bool read_scalar(const void * base, MLMultiArrayDataType dt, int64_t off, float * out) {
    switch (dt) {
        case MLMultiArrayDataTypeFloat32: *out = ((const float *)    base)[off];                 return true;
        case MLMultiArrayDataTypeDouble:  *out = (float) ((const double *) base)[off];            return true;
        case MLMultiArrayDataTypeFloat16: *out = decode_float16(((const uint16_t *) base)[off]);  return true;
        default:                          return false;
    }
}

// Copies the model output into `dst` as row-major (n_enc_frames, d_model), adapting to the
// output tensor orientation and dtype and honouring its element strides.
bool copy_output_array(MLMultiArray * arr, float * dst, int64_t n_enc_frames, int64_t d_model) {
    if (arr == nil) return false;

    const void               * base  = arr.dataPointer;
    const MLMultiArrayDataType dt    = arr.dataType;
    const int64_t              total = n_enc_frames * d_model;

    const parakeet::CoremlTrailingMatch match =
        parakeet::coreml_match_trailing_dims(dims_of(arr.shape), n_enc_frames, d_model);
    if (match.matched) {
        const bool            transpose = match.transpose;
        const NSUInteger      n       = arr.shape.count;
        NSArray<NSNumber *> * strides = arr.strides;
        const int64_t         s_outer = strides[n - 2].longLongValue;
        const int64_t         s_inner = strides[n - 1].longLongValue;
        for (int64_t t = 0; t < n_enc_frames; ++t) {
            for (int64_t f = 0; f < d_model; ++f) {
                const int64_t outer = transpose ? f : t;
                const int64_t inner = transpose ? t : f;
                float value;
                if (!read_scalar(base, dt, outer * s_outer + inner * s_inner, &value)) return false;
                dst[t * d_model + f] = value;
            }
        }
        return true;
    }

    // Unmatched shape but matching element count: assume a contiguous row-major
    // (n_enc_frames, d_model) buffer -- the natural layout for a flattened export.
    if ((int64_t) arr.count != total) return false;
    for (int64_t i = 0; i < total; ++i) {
        float value;
        if (!read_scalar(base, dt, i, &value)) return false;
        dst[i] = value;
    }
    return true;
}

// Copies `n_rows` frames starting at `row_begin` of one ANE window's
// [1, d_model, 1, T] output into row-major (frame, d_model) rows of `dst`,
// honouring the declared element strides. `row_begin` skips the window's left
// context, which encode_ane has already committed from the previous window.
bool copy_ane_window(MLMultiArray * arr, float * dst,
                     int64_t row_begin, int64_t n_rows, int64_t d_model) {
    if (arr == nil || arr.shape.count != 4) return false;

    const void               * base    = arr.dataPointer;
    const MLMultiArrayDataType dt      = arr.dataType;
    NSArray<NSNumber *>      * strides = arr.strides;
    const int64_t              s_chan  = strides[1].longLongValue;
    const int64_t              s_time  = strides[3].longLongValue;

    if (arr.shape[1].longLongValue != d_model)             return false;
    if (arr.shape[3].longLongValue < row_begin + n_rows)   return false;

    for (int64_t t = 0; t < n_rows; ++t) {
        const int64_t src_t = row_begin + t;
        for (int64_t f = 0; f < d_model; ++f) {
            float value;
            if (!read_scalar(base, dt, f * s_chan + src_t * s_time, &value)) return false;
            dst[t * d_model + f] = value;
        }
    }
    return true;
}

}  // namespace

struct parakeet_coreml_context {
    const void * model = nullptr;  // CFBridgingRetain'd MLModel *
    std::string  input_name;
    std::string  output_name;
    std::string  label;

    // Fixed-window ANE sidecar (scripts/export-encoder-coreml-ane.py). When set,
    // the encoder graph accepts exactly `window_mel` mel frames plus the two
    // masks, and parakeet_coreml_encode tiles an utterance of any length over
    // it. Zero means this is a single-input export.
    int64_t window_mel  = 0;
    int64_t enc_frames  = 0;
    int64_t n_mels      = 0;
    int64_t d_model     = 0;

    // Encoder frames of context each window carries on both sides of the region
    // it commits, trimmed off before the output is copied out. See encode_ane.
    int64_t ane_context = 0;
};

struct parakeet_coreml_context * parakeet_coreml_init(const char * path_mlmodelc) {
    if (path_mlmodelc == nullptr) return nullptr;

    @autoreleasepool {
        NSString * path = [[NSString alloc] initWithUTF8String:path_mlmodelc];
        if (path == nil) return nullptr;

        // PARAKEET_COREML_COMPUTE_UNITS=all loads with Core ML's default
        // compute-unit preference instead of CPU and Neural Engine.
        const char * units_env = std::getenv("PARAKEET_COREML_COMPUTE_UNITS");
        const bool   want_all  = units_env != nullptr && std::strcmp(units_env, "all") == 0;

        // Load with the placement the fixed-window ANE export needs, so it is
        // initialised once. A single-input export reloads further down with the
        // default preference.
        MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
        config.computeUnits = want_all ? MLComputeUnitsAll
                                       : MLComputeUnitsCPUAndNeuralEngine;

        NSError * err   = nil;
        MLModel * model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                            configuration:config
                                                    error:&err];
        if (model == nil || err != nil) return nullptr;

        NSDictionary<NSString *, MLFeatureDescription *> * ins =
            model.modelDescription.inputDescriptionsByName;
        NSString * out_name = sole_multiarray_feature(model.modelDescription.outputDescriptionsByName);
        if (out_name == nil) return nullptr;

        // Fixed-window ANE export: three named inputs, all shapes concrete.
        if (ins[kAneMelInput] != nil && ins[kAnePadMaskInput] != nil && ins[kAneKeyBiasInput] != nil) {
            const std::vector<int64_t> mel_dims =
                dims_of(ins[kAneMelInput].multiArrayConstraint.shape);
            const std::vector<int64_t> mask_dims =
                dims_of(ins[kAnePadMaskInput].multiArrayConstraint.shape);
            const std::vector<int64_t> out_dims =
                dims_of(model.modelDescription.outputDescriptionsByName[out_name].multiArrayConstraint.shape);
            if (mel_dims.size() != 4 || mask_dims.size() != 4 || out_dims.size() != 4) return nullptr;

            // `model` is already pinned to CPU and Neural Engine unless want_all.
            // MLComputeUnitsAll is only a preference, and Core ML may resolve it
            // to the GPU even for a graph that is fully ANE-capable.
            auto * ctx = new parakeet_coreml_context();
            ctx->model       = CFBridgingRetain(model);
            ctx->input_name  = kAneMelInput.UTF8String;
            ctx->output_name = out_name.UTF8String;
            ctx->label       = want_all ? "coreml-ane-all" : "coreml-ane";
            ctx->n_mels      = mel_dims[1];
            ctx->window_mel  = mel_dims[3];
            ctx->enc_frames  = mask_dims[3];
            ctx->d_model     = out_dims[1];

            // Seam context in encoder frames: 16 by default (1.28 s of audio),
            // capped at a quarter of the window. Context shrinks the stride
            // between windows. PARAKEET_COREML_ANE_CONTEXT_FRAMES overrides it;
            // 0 tiles windows edge to edge.
            int64_t ane_ctx = std::min<int64_t>(16, ctx->enc_frames / 4);
            if (const char * ctx_env = std::getenv("PARAKEET_COREML_ANE_CONTEXT_FRAMES")) {
                ane_ctx = std::max<int64_t>(0, atoll(ctx_env));
            }
            ctx->ane_context = std::min<int64_t>(ane_ctx, ctx->enc_frames / 4);

            if (ctx->n_mels <= 0 || ctx->window_mel <= 0 || ctx->enc_frames <= 0 || ctx->d_model <= 0) {
                CFRelease(ctx->model);
                delete ctx;
                return nullptr;
            }
            return ctx;
        }

        NSString * in_name = sole_multiarray_feature(ins);
        if (in_name == nil) return nullptr;

        // Single-input export: not pinned to the Neural Engine, so reload it
        // with the default compute-unit preference.
        if (!want_all) {
            config.computeUnits = MLComputeUnitsAll;
            MLModel * relaxed = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path]
                                                  configuration:config
                                                          error:&err];
            if (relaxed != nil && err == nil) model = relaxed;
        }

        auto * ctx = new parakeet_coreml_context();
        ctx->model       = CFBridgingRetain(model);
        ctx->input_name  = in_name.UTF8String;
        ctx->output_name = out_name.UTF8String;
        ctx->label       = "coreml";
        return ctx;
    }
}

void parakeet_coreml_free(struct parakeet_coreml_context * ctx) {
    if (ctx == nullptr) return;
    if (ctx->model != nullptr) CFRelease(ctx->model);
    delete ctx;
}

// Runs a fixed-window ANE sidecar over an utterance of arbitrary length by
// tiling it into overlapping `ctx->window_mel`-frame windows and keeping only
// each window's committed centre -- the same scheme run_encoder_windowed() uses
// over the ggml encoder (see long_form.h), specialised for a fed length that is
// fixed by the graph.
//
// Windows overlap because a frame at a hard cut cannot attend past it, which
// degrades the encoder output near the cut. Each window is positioned
// `ane_context` encoder frames before the region it commits, and those leading
// frames are dropped on the way out. The mel offset is pinned to a multiple of
// the subsampling factor so that a window starting at mel frame 8*a produces
// global encoder frame a+j at its local frame j, which makes the trim exact.
//
// The committed range is destination-driven: summing the frames each window's
// own mel length produces would not add up, because the subsampling stack's
// three stride-2 convs round up once per window rather than once per utterance.
static int encode_ane(struct parakeet_coreml_context * ctx,
                      int64_t       n_mel_frames,
                      int64_t       n_mels,
                      const float * mel,
                      int64_t       n_enc_frames,
                      int64_t       d_model,
                      float       * encoder_out) {
    if (n_mels != ctx->n_mels || d_model != ctx->d_model) return 8;

    const int64_t window_mel = ctx->window_mel;
    const int64_t enc_frames = ctx->enc_frames;

    // Offline (non-causal) subsampling: three stride-2, kernel-3, pad-1 convs.
    // should_use_coreml_encoder only routes offline encoders here, so the
    // causal variant of this recurrence cannot apply.
    auto frames_for = [](int64_t len) {
        for (int i = 0; i < 3; ++i) len = (len + 2 - 3) / 2 + 1;
        return len;
    };

    @autoreleasepool {
        MLModel  * model    = (__bridge MLModel *) ctx->model;
        NSString * out_name = [NSString stringWithUTF8String:ctx->output_name.c_str()];

        NSError      * err      = nil;
        MLMultiArray * mel_arr  = [[MLMultiArray alloc] initWithShape:@[@1, @(n_mels), @1, @(window_mel)]
                                                             dataType:MLMultiArrayDataTypeFloat16
                                                                error:&err];
        MLMultiArray * mask_arr = [[MLMultiArray alloc] initWithShape:@[@1, @1, @1, @(enc_frames)]
                                                             dataType:MLMultiArrayDataTypeFloat16
                                                                error:&err];
        MLMultiArray * bias_arr = [[MLMultiArray alloc] initWithShape:@[@1, @(enc_frames), @1, @1]
                                                             dataType:MLMultiArrayDataTypeFloat16
                                                                error:&err];
        if (mel_arr == nil || mask_arr == nil || bias_arr == nil || err != nil) return 3;

        uint16_t * mel_dst  = (uint16_t *) mel_arr.dataPointer;
        uint16_t * mask_dst = (uint16_t *) mask_arr.dataPointer;
        uint16_t * bias_dst = (uint16_t *) bias_arr.dataPointer;

        const uint16_t one_h  = encode_float16(1.0f);
        const uint16_t zero_h = encode_float16(0.0f);
        const uint16_t pad_h  = encode_float16(kKeyBiasPad);

        // An utterance that fits in one window has no seams, so it commits the
        // whole window with no context.
        const int64_t ctx_enc    = n_enc_frames <= enc_frames ? 0 : ctx->ane_context;
        const int64_t center_enc = enc_frames - 2 * ctx_enc;
        if (center_enc <= 0) return 8;

        int64_t committed = 0;

        for (int64_t center_begin = 0; center_begin < n_enc_frames; center_begin += center_enc) {
            const int64_t win_begin = std::max<int64_t>(0, center_begin - ctx_enc);
            const int64_t mel_begin = win_begin * kAneSubsample;
            const int64_t mel_avail =
                std::min(window_mel, std::max<int64_t>(0, n_mel_frames - mel_begin));
            if (mel_avail <= 0) break;
            const int64_t valid_enc = std::min(enc_frames, frames_for(mel_avail));

            // Frames this window carries before its committed centre, and the
            // slice of the caller's slab it is responsible for.
            const int64_t left_drop = center_begin - win_begin;
            const int64_t rows      = std::min(std::min(center_enc, n_enc_frames - center_begin),
                                               valid_enc - left_drop);
            if (rows <= 0) break;

            for (int64_t m = 0; m < n_mels; ++m) {
                uint16_t * row = mel_dst + m * window_mel;
                for (int64_t t = 0; t < mel_avail; ++t) {
                    row[t] = encode_float16(mel[(mel_begin + t) * n_mels + m]);
                }
                for (int64_t t = mel_avail; t < window_mel; ++t) row[t] = zero_h;
            }
            for (int64_t t = 0; t < enc_frames; ++t) {
                const bool valid = t < valid_enc;
                mask_dst[t] = valid ? one_h  : zero_h;
                bias_dst[t] = valid ? zero_h : pad_h;
            }

            MLDictionaryFeatureProvider * provider =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                    kAneMelInput     : mel_arr,
                    kAnePadMaskInput : mask_arr,
                    kAneKeyBiasInput : bias_arr,
                } error:&err];
            if (provider == nil || err != nil) return 4;

            id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&err];
            if (result == nil || err != nil) return 5;

            MLMultiArray * out_arr = [result featureValueForName:out_name].multiArrayValue;
            if (!copy_ane_window(out_arr, encoder_out + center_begin * d_model,
                                 left_drop, rows, d_model)) return 7;

            committed = center_begin + rows;
        }

        // Every frame the caller sized its slab for must have been written by
        // some window; leaving a tail uninitialised would feed the decoder
        // garbage rather than fail.
        if (committed != n_enc_frames) return 9;
        return 0;
    }
}

int parakeet_coreml_encode(struct parakeet_coreml_context * ctx,
                           int64_t       n_mel_frames,
                           int64_t       n_mels,
                           const float * mel,
                           int64_t       n_enc_frames,
                           int64_t       d_model,
                           float       * encoder_out) {
    if (ctx == nullptr || ctx->model == nullptr || mel == nullptr || encoder_out == nullptr) return 1;
    if (n_mel_frames <= 0 || n_mels <= 0 || n_enc_frames <= 0 || d_model <= 0) return 1;

    if (ctx->window_mel > 0) {
        return encode_ane(ctx, n_mel_frames, n_mels, mel, n_enc_frames, d_model, encoder_out);
    }

    @autoreleasepool {
        MLModel  * model    = (__bridge MLModel *) ctx->model;
        NSString * in_name  = [NSString stringWithUTF8String:ctx->input_name.c_str()];
        NSString * out_name = [NSString stringWithUTF8String:ctx->output_name.c_str()];

        MLMultiArrayConstraint * in_constraint =
            model.modelDescription.inputDescriptionsByName[in_name].multiArrayConstraint;

        bool                  in_transpose = true;
        NSArray<NSNumber *> * in_shape     =
            resolve_input_shape(in_constraint.shape, n_mel_frames, n_mels, &in_transpose);
        if (in_shape == nil) return 2;

        NSError      * err    = nil;
        MLMultiArray * in_arr = build_input_array(in_shape, in_transpose, mel, n_mel_frames, n_mels, &err);
        if (in_arr == nil || err != nil) return 3;

        MLDictionaryFeatureProvider * provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ in_name : in_arr } error:&err];
        if (provider == nil || err != nil) return 4;

        id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&err];
        if (result == nil || err != nil) return 5;

        MLMultiArray * out_arr = [result featureValueForName:out_name].multiArrayValue;
        if (out_arr == nil) return 6;
        if (!copy_output_array(out_arr, encoder_out, n_enc_frames, d_model)) return 7;
        return 0;
    }
}

const char * parakeet_coreml_backend_label(const struct parakeet_coreml_context * ctx) {
    return ctx != nullptr ? ctx->label.c_str() : "coreml";
}

int64_t parakeet_coreml_window_mel_frames(const struct parakeet_coreml_context * ctx) {
    return ctx != nullptr ? ctx->window_mel : 0;
}

// Core ML FastConformer encoder sidecar for parakeet-cpp.
//
// Mirrors the whisper.cpp `whisper_coreml_*` C shim: an opaque context wraps a
// compiled `.mlmodelc` encoder that runs on the Apple Neural Engine, and the
// caller hands log-mel features in / reads encoder hidden states out. The rest
// of the pipeline (mel preprocessing, TDT/CTC decode, tokenizer) stays on ggml.
//
// This header is Apple-only; it is compiled and referenced solely when the
// PARAKEET_USE_COREML build definition is set. All entry points return failure
// (nullptr / non-zero) rather than aborting so the caller can fall back to the
// ggml encoder on any error.
//
// Two export shapes bind here, told apart by their input names; the C entry
// points below are the same for both.
//
// Fixed-window ANE export (scripts/export-encoder-coreml-ane.py):
//   - Three MLMultiArray inputs, all Float16 and all concrete:
//       `mel`       {1, n_mels, 1, window_mel_frames}
//       `pad_mask`  {1, 1, 1, enc_frames}          1.0 valid, 0.0 pad
//       `key_bias`  {1, enc_frames, 1, 1}          0.0 valid, -1e4 pad
//   - One MLMultiArray output {1, d_model, 1, enc_frames}.
//   The graph only ever sees one window. parakeet_coreml_encode tiles an
//   utterance of any length across it and builds the two masks per window, so
//   callers see the same contract as the single-input export below.
//
// Single-input export (e.g. the mobius `.mlmodelc`):
//   - Exactly one MLMultiArray input  = log-mel features, dims {n_mels, n_mel_frames}
//     (either order; the wrapper adapts). This is the offline FastConformer encoder
//     input, i.e. everything from the subsampling stack through the last Conformer
//     block; it must NOT include the TDT joint projection. The mel time axis may be
//     fixed (accelerates only that exact length; other lengths fall back to ggml) or
//     flexible (RangeDim / enumerated shapes): the wrapper allocates the input at the
//     caller's length in the model's declared orientation and lets Core ML reject an
//     unsupported length.
//   - Exactly one MLMultiArray output = encoder hidden states, dims {n_enc_frames,
//     d_model} (either order; Float32 or Float16). n_enc_frames is n_mel_frames put
//     through the three stride-2 subsampling convs (matching run_encoder's sizing).
//   - The model is the full-context (non-causal, non-chunked) encoder; streaming /
//     cache-aware configurations are served by ggml, not this sidecar.

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct parakeet_coreml_context;

// Load a compiled Core ML encoder from a `.mlmodelc` directory.
// Returns nullptr on any failure (missing directory, load error, unsupported
// OS, unexpected model interface) so the caller falls back to ggml.
struct parakeet_coreml_context * parakeet_coreml_init(const char * path_mlmodelc);

// Release a context created by parakeet_coreml_init. Safe to call with nullptr.
void parakeet_coreml_free(struct parakeet_coreml_context * ctx);

// Run the encoder on the Apple Neural Engine.
//
//   mel           log-mel features, row-major (n_mel_frames, n_mels),
//                 n_mels contiguous (the layout `compute_log_mel` emits).
//   encoder_out   destination for the encoder hidden states, row-major
//                 (n_enc_frames, d_model), d_model contiguous. The caller
//                 sizes it to n_enc_frames * d_model floats.
//
// The wrapper adapts the mel/output tensor orientation to whatever the exported
// model declares (features-major or time-major), so the mobius export only has
// to agree on the dimension sizes, not their order.
//
// Returns 0 on success; non-zero on a shape mismatch or prediction failure, in
// which case the caller falls back to the ggml encoder.
int parakeet_coreml_encode(struct parakeet_coreml_context * ctx,
                           int64_t       n_mel_frames,
                           int64_t       n_mels,
                           const float * mel,
                           int64_t       n_enc_frames,
                           int64_t       d_model,
                           float       * encoder_out);

// Human-readable compute label for stats/logging, e.g. "coreml-ane". The concrete
// unit (ANE / GPU / CPU) is chosen by Core ML at runtime and is not knowable
// up front, so this reports the backend family rather than the physical unit.
const char * parakeet_coreml_backend_label(const struct parakeet_coreml_context * ctx);

// Mel frames the compiled encoder graph accepts per dispatch (the export's
// --window-mel-frames), or 0 for a single-input export. Callers that cut audio
// into independently encoded segments can use this to keep a segment inside one
// dispatch.
int64_t parakeet_coreml_window_mel_frames(const struct parakeet_coreml_context * ctx);

#if defined(__cplusplus)
}
#endif

#pragma once

// TDT (token-and-duration transducer) decoder on FastConformer encoder output (run_encoder).
//
// Prediction LSTM, joint MLP, duration head, and greedy decode over encoder frames.
// GPU paths run per-step ops as ggml graphs on the loaded backend; CPU decode uses
// host GEMV/LSTM with weights prepared at load time.
//
// Typical layout (e.g. parakeet-tdt-0.6b-v3): 2-layer LSTM (hidden 640), joint with
// enc/pred projections and duration logits, greedy loop advancing the encoder index
// by predicted duration.

#include "parakeet_ctc.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_gallocr * ggml_gallocr_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace parakeet {

// Per-layer host-dequantised LSTM weights, used by the CPU fallback path
// (per-step ggml-graph dispatch on the CPU backend has too much overhead
// for the 200-300 emission steps in a typical 20s utterance).
// A host-resident weight matrix for the CPU decoder's gemvs. q8_0 GGUF tensors
// are kept in their on-disk form and dotted against a q8_0-quantised
// activation. Tensors that aren't q8_0 (or whose row length isn't a multiple
// of the block size) are expanded to f32.
struct HostMat {
    int                  out_dim   = 0;
    int                  in_dim    = 0;
    bool                 quant     = false;
    size_t               row_bytes = 0;  // q8_0 bytes per row
    std::vector<uint8_t> q;
    std::vector<float>   f;
};

struct TdtHostLstmLayer {
    HostMat            w_ih;
    HostMat            w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;
};

// Row-parallel worker pool for the CPU decoder's gemvs; defined in the .cpp.
struct GemvPool;

// Per-decoder runtime context. Two implementation paths:
//   - GPU path  (Metal / CUDA / Vulkan): per-step LSTM, joint and full-window
//                enc-projection are ggml graphs running on the active backend
//                with native quantised GGUF weights.
//   - CPU path  (`use_graphs == false`): host-dequantised f32 weights + scalar
//                gemv loops, since per-step graph dispatch on the CPU backend
//                is slower than these loops.
//
// Per-window enc-projection graphs are cached by frame count (up to
// `k_enc_proj_cache_max`) so streaming chunks of common sizes don't pay
// graph-build overhead more than once.
//
// Move-only: graph scaffolding owns backend resources that must outlive the
// engine but cannot be duplicated.
struct TdtRuntimeWeights {
    int H_pred       = 640;
    int H_joint      = 640;
    int D_enc        = 1024;
    int V_plus_1     = 8193;
    int V_out        = 8198;
    int L            = 2;
    int num_durations = 5;

    const TdtWeights * weights = nullptr;
    ggml_backend_t     backend = nullptr;
    int                n_threads = 0;
    bool               use_graphs = false;
    // false on ggml-opencl (no ARGMAX kernel): the joint graph emits raw logits
    // for the host to argmax; true elsewhere keeps the argmax on-device.
    bool               argmax_on_gpu = true;

    // ---- CPU-fallback host weights (populated only when !use_graphs) ----
    HostMat                        embed;
    std::vector<TdtHostLstmLayer>  host_lstm;
    HostMat                        host_joint_enc_w;
    std::vector<float>             host_joint_enc_b;
    HostMat                        host_joint_pred_w;
    std::vector<float>             host_joint_pred_b;
    HostMat                        host_joint_out_w;
    std::vector<float>             host_joint_out_b;

    // Reused q8_0 activation buffer for the gemvs above. Each gemv quantises
    // its input at entry and consumes it before returning, so one buffer is
    // enough.
    mutable std::vector<uint8_t>   qact;

    // Splits the gemvs above across cores. Null when the host path is unused,
    // the weights aren't q8_0, or PARAKEET_TDT_DECODER_THREADS is unset or 1.
    std::unique_ptr<GemvPool>      pool;

    // ---- GPU graph scaffolding (populated only when use_graphs) ----
    ggml_context * gctx = nullptr;

    // Persistent decoder state on the GPU backend (Metal/CUDA/Vulkan).
    //
    // Each emission step pays command-buffer submit/wait overhead; splitting
    // joint and LSTM into separate graphs per step doubled that cost. Here
    // h, c, pred, and full-window enc_proj stay in persist_buffer, wired with
    // ggml_cpy, plus a fused `g_lstm_joint` graph after non-blank emissions
    // so LSTM update and joint run in one graph commit when possible.
    ggml_context *           persist_ctx    = nullptr;
    ggml_backend_buffer_t    persist_buffer = nullptr;
    ggml_tensor *            h_persist        = nullptr;  // [H_pred, L]
    ggml_tensor *            c_persist        = nullptr;  // [H_pred, L]
    ggml_tensor *            pred_persist     = nullptr;  // [H_pred]
    ggml_tensor *            enc_proj_persist = nullptr;  // [H_joint, T_max]
    int                      enc_proj_T_max   = 0;

    // (1) Init-only LSTM graph: zeroes h/c and runs LSTM with the blank
    //     token to seed pred_persist. Used once per call (tdt_init_state).
    ggml_cgraph *  g_lstm     = nullptr;
    ggml_gallocr_t alloc_lstm = nullptr;
    ggml_tensor *  lstm_token_in = nullptr;
    ggml_tensor *  lstm_h_out    = nullptr;  // ggml_cpy result aliasing h_persist
    ggml_tensor *  lstm_c_out    = nullptr;  // ggml_cpy result aliasing c_persist
    ggml_tensor *  lstm_pred_out = nullptr;  // ggml_cpy result aliasing pred_persist

    // (2) Joint-only graph: used after a blank emission (pred unchanged
    //     from previous iteration). Reads pred_persist + enc_proj_persist
    //     [frame_idx]; emits token + dur argmax i32 indices instead of
    //     full logits to keep PCIe-class backends from paying the
    //     V_out * 4 B readback per step (~32 KB at V_out = 8198, fine on
    //     Apple unified memory at ~17 us / call but order-of-magnitude
    //     worse on a discrete GPU bus; one int32 per step is the right
    //     shape for both).  Token argmax is over logits[0:V_plus_1],
    //     duration argmax is over logits[V_plus_1:V_plus_1+num_durations].
    ggml_cgraph *  g_joint     = nullptr;
    ggml_gallocr_t alloc_joint = nullptr;
    ggml_tensor *  joint_frame_idx_in = nullptr;  // i32[1]
    ggml_tensor *  joint_token_out    = nullptr;  // i32 argmax, or f32 token logits when !argmax_on_gpu
    ggml_tensor *  joint_dur_out      = nullptr;  // i32 argmax, or f32 dur logits when !argmax_on_gpu

    // (3) Fused LSTM + joint graph: used after a non-blank emission.
    //     LSTM updates h/c/pred from the last emitted token, then joint
    //     reads the *fresh* pred and enc_proj_persist[frame_idx] in the
    //     same compute_graph (one command-buffer commit instead of two).
    //     Same on-device argmax as g_joint.
    ggml_cgraph *  g_lstm_joint     = nullptr;
    ggml_gallocr_t alloc_lstm_joint = nullptr;
    ggml_tensor *  lj_token_in        = nullptr;  // i32[1]
    ggml_tensor *  lj_frame_idx_in    = nullptr;  // i32[1]
    ggml_tensor *  lj_token_out       = nullptr;  // i32 argmax, or f32 token logits when !argmax_on_gpu
    ggml_tensor *  lj_dur_out         = nullptr;  // i32 argmax, or f32 dur logits when !argmax_on_gpu

    struct EncProjGraph {
        // Each cached graph owns its own ggml_context for the cgraph + tensor
        // metadata.  Previous design parented these on `gctx` (the long-lived
        // runtime context), and the LRU eviction below freed only the gallocr
        // — which leaks ~32 graph slots from gctx per evicted entry.  Mode 1
        // (single T_enc per call) hits this once and is fine; Mode 3 streaming
        // with varying right-lookahead-ms can chew through the gctx slot pool
        // and silently fail-to-allocate after ~30 distinct T_enc values.
        // Owning the metadata locally and freeing it at eviction keeps Mode 3
        // bounded by the LRU cap regardless of how many distinct T_enc values
        // a session sees.
        ggml_context * ctx    = nullptr;
        ggml_cgraph *  cg     = nullptr;
        ggml_gallocr_t alloc  = nullptr;
        ggml_tensor *  enc_in = nullptr;
        ggml_tensor *  out    = nullptr;  // ggml_cpy aliasing enc_proj_persist[:T]
        int            T      = 0;
    };
    std::vector<EncProjGraph> enc_proj_cache;
    static constexpr size_t k_enc_proj_cache_max = 3;
    // ~5 minutes of audio at the encoder's 80 ms-frame rate fits in 4096
    // rows; H_joint=640 * f32 → ~10 MB, fine for any backend we target.
    // Audio that exceeds this falls back to a per-call dynamic-T allocation.
    static constexpr int k_enc_proj_T_max = 4096;

    // Out of line: GemvPool is incomplete here, so the implicit member
    // cleanup these would need cannot be generated in other TUs.
    TdtRuntimeWeights();
    TdtRuntimeWeights(const TdtRuntimeWeights &) = delete;
    TdtRuntimeWeights & operator=(const TdtRuntimeWeights &) = delete;
    TdtRuntimeWeights(TdtRuntimeWeights && other) noexcept;
    TdtRuntimeWeights & operator=(TdtRuntimeWeights && other) noexcept;
    ~TdtRuntimeWeights();

    bool ready() const { return weights != nullptr; }
};

struct TdtDecodeOptions {
    int max_symbols_per_step = 10;
};

struct TdtDecodeResult {
    std::vector<int32_t> token_ids;
    std::string text;
    int steps = 0;
    double decode_ms = 0.0;
};

struct TdtDecodeState {
    std::vector<float> h_state;
    std::vector<float> c_state;
    std::vector<float> pred_out;

    int  symbols_this_step = 0;
    bool initialized       = false;
    int  carry_frames      = 0;
};

int tdt_prepare_runtime(const ParakeetCtcModel & model, TdtRuntimeWeights & out);

void tdt_init_state(TdtRuntimeWeights & W,
                    int blank_id,
                    TdtDecodeState & state);

int tdt_decode_window(const ParakeetCtcModel & model,
                      TdtRuntimeWeights & W,
                      const float * encoder_out_window,
                      int n_frames, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeState & state,
                      std::vector<int32_t> & out_tokens,
                      int & out_steps);

int tdt_greedy_decode(const ParakeetCtcModel & model,
                      TdtRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeResult & result);

#ifdef PARAKEET_USE_COREML
// One independently decoded window: a slice of encoder output that gets its own
// TDT state. Slices point into a buffer the caller owns and must outlive the
// call; nothing here copies encoder output.
struct TdtLaneWindow {
    const float * enc      = nullptr;  // [n_frames][D_enc], row-major
    int           n_frames = 0;
};

// Decode independent windows on the Core ML sidecar, `lanes` of them per
// dispatch. Emits one token vector per input window, in input order, so the
// caller can detokenise and join them however it stitches.
//
// Windows are seated into lanes as lanes free up, so the batch stays full until
// the input runs out; `out_dispatches` (optional) receives the number of
// dispatches.
//
// A window that emits nothing at all is re-decoded in a second batched pass,
// warm-started from the previous window's retired state. `out_rescued`
// (optional) receives the number of windows re-decoded.
//
// Returns 0 on success. Any non-zero return leaves out_tokens unusable and means
// the caller should fall back to tdt_greedy_decode for the whole utterance: a
// half-decoded lane batch cannot be resumed.
int tdt_decode_lanes(const ParakeetCtcModel & model,
                     TdtRuntimeWeights & W,
                     struct parakeet_coreml_decoder * dec,
                     const std::vector<TdtLaneWindow> & windows,
                     const TdtDecodeOptions & opts,
                     std::vector<std::vector<int32_t>> & out_tokens,
                     int * out_dispatches,
                     int * out_rescued);
#endif

}

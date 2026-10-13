#pragma once

// Bounded-memory long-form encoder windowing for the offline transcribe paths.
//
// The conformer encoder's self-attention is O(T_enc^2) in the number of encoder
// frames, so running it over a multi-hour input in a single graph allocates
// tens of GB of attention scores and OOMs the process. The offline paths
// compute the mel-spectrogram once (so per-feature CMVN statistics are global,
// matching the single-pass path) and then slide the encoder over the mel in
// overlapping windows: plan_long_form_windows() tiles the mel into gap-free,
// non-overlapping "committed" centre regions, each padded with left/right
// context; run_encoder_windowed() (see parakeet_engine.cpp) runs the encoder
// per window, trims the context back off at the seams
// (compute_window_trim), and concatenates the committed frames
// (append_committed_frames) into the same [T_enc, d_model] buffer a single
// full-length pass would produce, at bounded peak memory.
//
// The CLI's offline path (src/main.cpp) uses the same scheme with equal-length
// windows (plan_long_form_windows_asymmetric) sized from --window-sec /
// --overlap (long_form_frames_from_seconds, resolve_cli_long_form_plan).
//
// This header holds only pure, dependency-free helpers (the window planners, the
// window-size resolvers, and the seam trim + append) so they can be unit-tested
// without loading a model (see test/test_long_form_windows.cpp).

#include <climits>
#include <cmath>
#include <vector>
#include <cstddef>

namespace parakeet {

struct LongFormWindow {
    int  window_start;  // first unit fed to the encoder (incl. left context)
    int  window_len;    // number of units fed to the encoder for this window
    int  center_start;  // first committed (kept) unit, in [0, n_units)
    int  center_end;    // one-past-last committed unit
    bool is_final;      // last window: keeps every encoder frame from center_start on
};

// Tiles [0, n_units) into gap-free, non-overlapping committed centres of
// `center_units` each (the final centre is the remainder), padding every centre
// with up to `ctx_units` of left/right context clamped to the bounds. Units are
// mel frames in the offline windowed path; the planner itself is unit-agnostic.
// The context is what the encoder needs to produce committed frames that match
// the single-pass encoder; run_encoder_windowed() trims it off after the encoder
// runs, so no committed unit is ever emitted twice.
//
// Guarantees (exercised by test/test_long_form_windows.cpp):
//   - returns >= 1 window for any n_units > 0;
//   - the committed centres are contiguous and cover exactly [0, n_units)
//     (windows[0].center_start == 0; windows[i].center_end ==
//      windows[i+1].center_start; windows.back().center_end == n_units);
//   - each window fully contains its committed centre
//     (window_start <= center_start && window_start + window_len >= center_end);
//   - window_len <= center_units + 2 * ctx_units, so peak encoder memory is
//     bounded regardless of total input length.
inline std::vector<LongFormWindow>
plan_long_form_windows(int n_units, int center_units, int ctx_units) {
    std::vector<LongFormWindow> windows;
    if (n_units <= 0 || center_units <= 0) {
        return windows;
    }
    if (ctx_units < 0) {
        ctx_units = 0;
    }

    for (int center_start = 0; center_start < n_units; center_start += center_units) {
        int center_end = center_start + center_units;
        if (center_end > n_units) {
            center_end = n_units;
        }

        int window_start = center_start - ctx_units;
        if (window_start < 0) {
            window_start = 0;
        }
        int window_end = center_end + ctx_units;
        if (window_end > n_units) {
            window_end = n_units;
        }

        LongFormWindow w;
        w.window_start = window_start;
        w.window_len   = window_end - window_start;
        w.center_start = center_start;
        w.center_end   = center_end;
        w.is_final     = (center_end >= n_units);
        windows.push_back(w);
    }
    return windows;
}

// Same tiling and guarantees as plan_long_form_windows(), except that context a
// boundary window cannot place on one side is moved to the other side while
// input remains. Every window is then min(center + left + right, n_units) units
// long (`exact_window_units` instead, when non-zero), so an input longer than
// one window reaches the encoder in a single shape -- one compute graph -- and
// boundary frames get real audio as context instead of less of it. The window
// bound is center_units + left_ctx_units + right_ctx_units.
inline std::vector<LongFormWindow>
plan_long_form_windows_asymmetric(int n_units, int center_units,
                                  int left_ctx_units, int right_ctx_units,
                                  int exact_window_units = 0) {
    std::vector<LongFormWindow> windows;
    if (n_units <= 0 || center_units <= 0) {
        return windows;
    }
    if (left_ctx_units < 0) left_ctx_units = 0;
    if (right_ctx_units < 0) right_ctx_units = 0;

    for (int center_start = 0; center_start < n_units; center_start += center_units) {
        int center_end = center_start + center_units;
        if (center_end > n_units) {
            center_end = n_units;
        }

        int window_start = center_start - left_ctx_units;
        if (window_start < 0) {
            window_start = 0;
        }
        int window_end = center_end + right_ctx_units;
        if (window_end > n_units) {
            window_end = n_units;
        }

        const long long requested_len_ll = exact_window_units > 0
            ? (long long) exact_window_units
            : (long long) center_units + left_ctx_units + right_ctx_units;
        const int target_len = requested_len_ll < n_units
            ? (int) requested_len_ll : n_units;
        int missing = target_len - (window_end - window_start);
        if (missing > 0) {
            int grow_right = n_units - window_end;
            if (grow_right > missing) grow_right = missing;
            window_end += grow_right;
            missing -= grow_right;
            if (missing > 0) {
                window_start -= missing;
                if (window_start < 0) window_start = 0;
            }
        }

        LongFormWindow w;
        w.window_start = window_start;
        w.window_len   = window_end - window_start;
        w.center_start = center_start;
        w.center_end   = center_end;
        w.is_final     = (center_end >= n_units);
        windows.push_back(w);
    }
    return windows;
}

// Resolve the effective per-window encoder-frame ceiling from the requested
// value (EngineOptions::long_form_window_frames, which must be >= 0 here -- the
// < 0 "disabled" case is handled by the caller) and the model's trained
// positional range:
//   - `requested` == 0 selects `auto_window_frames`; otherwise it is the
//     explicit request;
//   - the result is floored to `min_window_frames` so a tiny request can't
//     collapse the committed centre;
//   - `pos_emb_max_len` (when > 0) is a HARD ceiling that wins over that floor:
//     a window past it would index positional embeddings the model never saw,
//     so a pathologically small range yields a small window (which the caller's
//     center_frames <= 0 guard may then reject in favour of single-pass) rather
//     than one that exceeds the trained range.
//
// Exercised model-free by test/test_long_form_windows.cpp.
inline int resolve_long_form_window_frames(int requested, int pos_emb_max_len,
                                           int auto_window_frames,
                                           int min_window_frames) {
    int window_frames = (requested == 0) ? auto_window_frames : requested;
    if (window_frames < min_window_frames) {
        window_frames = min_window_frames;
    }
    if (pos_emb_max_len > 0 && window_frames > pos_emb_max_len) {
        window_frames = pos_emb_max_len;
    }
    return window_frames;
}

// Encoder-frame window/context pair, with the sign conventions of
// EngineOptions::long_form_*: window_frames < 0 disables windowing, and
// context_frames < 0 means no context. The converter below never emits a
// context of 0, which the engine would read as its auto context (~50 % overlap
// after the window/4 clamp).
struct LongFormFrames {
    int window_frames  = -1;
    int context_frames = -1;
};

// Converts the CLI's --window-sec / --overlap into encoder frames:
//   - window_sec == 0 disables windowing;
//   - otherwise window_frames = round(window_sec / frame_sec), which must lie in
//     [min_window_frames, max_window_frames] (max_window_frames <= 0: no ceiling
//     other than INT_MAX);
//   - overlap is the fraction of each window shared with its neighbours, in
//     [0, 0.5]: context_frames = round(window_frames * overlap / 2), clamped to
//     window_frames / 4 like the engine, and -1 when it rounds to 0.
// Returns false, leaving `out` untouched, for non-finite or out-of-range input.
inline bool long_form_frames_from_seconds(double window_sec, double overlap,
                                          double frame_sec,
                                          int min_window_frames,
                                          int max_window_frames,
                                          LongFormFrames & out) {
    if (!std::isfinite(window_sec) || !std::isfinite(overlap) ||
        window_sec < 0.0 || overlap < 0.0 || overlap > 0.5) {
        return false;
    }
    if (window_sec == 0.0) {
        out.window_frames  = -1;
        out.context_frames = -1;
        return true;
    }
    if (!std::isfinite(frame_sec) || frame_sec <= 0.0) {
        return false;
    }

    const double frames = std::round(window_sec / frame_sec);
    if (frames < (double) min_window_frames || frames > (double) INT_MAX ||
        (max_window_frames > 0 && frames > (double) max_window_frames)) {
        return false;
    }
    const int window_frames = (int) frames;

    int context_frames = (int) std::lround((double) window_frames * overlap / 2.0);
    if (context_frames > window_frames / 4) {
        context_frames = window_frames / 4;
    }
    out.window_frames  = window_frames;
    out.context_frames = context_frames > 0 ? context_frames : -1;
    return true;
}

// The window the CLI uses when --window-sec is not given: default_sec moved
// into [min_window_frames, max_window_frames] frames, or 0 (single pass) when
// the model cannot hold a min_window_frames window, frame_sec is unusable, or
// the window would exceed INT_MAX frames (no int-sized input needs one). The
// result always passes long_form_frames_from_seconds with the same bounds, so a
// run without the flag never fails on it.
inline double fit_default_window_sec(double default_sec, double frame_sec,
                                     int min_window_frames, int max_window_frames) {
    if (!std::isfinite(default_sec) || default_sec <= 0.0 ||
        !std::isfinite(frame_sec) || frame_sec <= 0.0 ||
        (max_window_frames > 0 && max_window_frames < min_window_frames)) {
        return 0.0;
    }
    const double frames = std::round(default_sec / frame_sec);
    if (frames > (double) INT_MAX) {
        return 0.0;
    }
    if (frames < (double) min_window_frames) {
        return min_window_frames * frame_sec;
    }
    if (max_window_frames > 0 && frames > (double) max_window_frames) {
        return max_window_frames * frame_sec;
    }
    return default_sec;
}

struct LongFormCliPlan {
    bool enabled        = false;
    int  window_frames  = 0;  // encoder frames per window (center + 2 * context)
    int  context_frames = 0;  // encoder frames of shared context each side
    int  center_frames  = 0;  // committed encoder frames per window
    int  sub            = 0;  // subsampling factor (mel frames per encoder frame)
};

// Decides whether an n_mel_frames input needs the CLI's windowed encoder pass
// under `frames` (from long_form_frames_from_seconds). Inputs that fit within
// one window keep the single-pass path at their own length (enabled == false).
inline LongFormCliPlan resolve_cli_long_form_plan(const LongFormFrames & frames,
                                                  int subsampling_factor,
                                                  long long n_mel_frames) {
    LongFormCliPlan plan;
    if (frames.window_frames <= 0 || subsampling_factor <= 0) {
        return plan;
    }

    int context_frames = frames.context_frames > 0 ? frames.context_frames : 0;
    if (context_frames > frames.window_frames / 4) {
        context_frames = frames.window_frames / 4;
    }
    const int center_frames = frames.window_frames - 2 * context_frames;
    if (center_frames <= 0) {
        return plan;
    }
    if (n_mel_frames <= (long long) frames.window_frames * subsampling_factor) {
        return plan;
    }

    plan.enabled        = true;
    plan.window_frames  = frames.window_frames;
    plan.context_frames = context_frames;
    plan.center_frames  = center_frames;
    plan.sub            = subsampling_factor;
    return plan;
}

struct WindowTrim {
    int left_drop;   // leading encoder frames (left context) to discard
    int center_cnt;  // committed encoder frames to keep
};

// Maps a planned window (mel-frame units) onto the encoder frames the window
// actually produced (`t_enc_frames`): drop the left-context frames, keep the
// committed centre, drop the right-context frames -- except the final window,
// which keeps its whole tail after the left context. `sub` is the subsampling
// factor (mel frames per encoder frame). Results are clamped into
// [0, t_enc_frames] so per-window subsampling rounding at the boundaries can
// never index out of range.
inline WindowTrim compute_window_trim(const LongFormWindow & w,
                                      int t_enc_frames, int sub) {
    const int div = sub > 0 ? sub : 1;
    WindowTrim t;
    t.left_drop  = (w.center_start - w.window_start) / div;
    t.center_cnt = w.is_final ? (t_enc_frames - t.left_drop)
                              : (w.center_end - w.center_start) / div;
    if (t.left_drop < 0) t.left_drop = 0;
    if (t.left_drop > t_enc_frames) t.left_drop = t_enc_frames;
    if (t.center_cnt < 0) t.center_cnt = 0;
    if (t.center_cnt > t_enc_frames - t.left_drop) {
        t.center_cnt = t_enc_frames - t.left_drop;
    }
    return t;
}

// Appends `count` frames (each `stride` floats) starting at `start_frame` from
// `src` onto `dst`. Returns false without touching `dst` when the requested
// range would read past `src` -- callers must treat that as an error rather than
// silently advancing their frame count, otherwise a downstream decoder could
// read past the concatenated buffer.
inline bool append_committed_frames(std::vector<float> & dst,
                                    const std::vector<float> & src,
                                    int start_frame, int count, int stride) {
    if (count <= 0 || stride <= 0) {
        return true;
    }
    const size_t off = (size_t) start_frame * (size_t) stride;
    const size_t n   = (size_t) count * (size_t) stride;
    if (off + n > src.size()) {
        return false;
    }
    dst.insert(dst.end(), src.begin() + off, src.begin() + off + n);
    return true;
}

}  // namespace parakeet

// Model-free unit tests for the long-form encoder windowing arithmetic
// (src/long_form.h): the window planner, the window-size resolver, the seam
// trim, and the committed-frame append. No GGUF or ggml is needed, so these run
// on every checkout and are the primary regression gate for the bounded-memory
// long-audio fix.
//
// They assert the properties the fix depends on:
//   1. every window fed to the encoder is length-bounded (<= center + 2*ctx),
//      independent of total input length -- so peak encoder memory is bounded;
//   2. the committed centres are gap-free, non-overlapping and cover exactly
//      [0, n_units) -- so no audio is dropped or transcribed twice;
//   3. the resolved per-window frame ceiling never exceeds the encoder's trained
//      positional range (pos_emb_max_len), which is a hard ceiling over the
//      minimum-window floor;
//   4. trim + append stitch the per-window encoder outputs back into one
//      contiguous frame sequence with no gaps or duplicates at the seams;
//   5. the append refuses (rather than silently truncating) an out-of-range
//      request, so a broken invariant fails loud instead of reading past a buffer.
//
// The CLI's equal-length planner and its --window-sec / --overlap conversion
// add:
//   6. every window of an input longer than one window has the same length, so
//      the encoder sees a single input shape (one compute graph);
//   7. 0 % overlap reaches the planner as no context (-1 in the EngineOptions
//      encoding, never 0, which would mean "auto"); 50 % clamps to window/4;
//      windows below 256 encoder frames (~20.5 s) or above pos_emb_max_len are
//      rejected;
//   8. the default window used without --window-sec is fitted to the model's
//      range and never makes the conversion fail.
//
// Exit 0 on success; non-zero (with a FAIL line per broken invariant) otherwise.

#include "long_form.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using parakeet::LongFormCliPlan;
using parakeet::LongFormFrames;
using parakeet::LongFormWindow;
using parakeet::WindowTrim;
using parakeet::append_committed_frames;
using parakeet::compute_window_trim;
using parakeet::fit_default_window_sec;
using parakeet::long_form_frames_from_seconds;
using parakeet::plan_long_form_windows;
using parakeet::plan_long_form_windows_asymmetric;
using parakeet::resolve_cli_long_form_plan;
using parakeet::resolve_long_form_window_frames;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

// Every invariant the planner must uphold for a valid (n, center, ctx) triple.
void check_plan(int n, int center, int ctx) {
    const std::string tag = "plan n=" + std::to_string(n) +
                            " center=" + std::to_string(center) +
                            " ctx=" + std::to_string(ctx);
    const std::vector<LongFormWindow> ws = plan_long_form_windows(n, center, ctx);

    expect(!ws.empty(), tag + ": expected at least one window");
    if (ws.empty()) return;

    expect(ws.front().center_start == 0, tag + ": first centre must start at 0");
    expect(ws.back().center_end == n, tag + ": last centre must end at n_units");

    long long covered = 0;
    for (size_t i = 0; i < ws.size(); ++i) {
        const LongFormWindow & w = ws[i];
        const std::string wtag = tag + " window[" + std::to_string(i) + "]";

        expect(w.center_start < w.center_end, wtag + ": empty committed centre");
        expect(w.window_start >= 0, wtag + ": window_start < 0");
        expect(w.window_start + w.window_len <= n, wtag + ": window runs past n_units");
        expect(w.window_start <= w.center_start, wtag + ": window_start after centre");
        expect(w.window_start + w.window_len >= w.center_end,
               wtag + ": window ends before centre");
        expect(w.window_len <= center + 2 * ctx, wtag + ": window length exceeds bound");
        expect(w.is_final == (i + 1 == ws.size()), wtag + ": is_final mismatch");
        if (i + 1 < ws.size()) {
            expect(w.center_end == ws[i + 1].center_start,
                   wtag + ": centres not contiguous (gap or overlap at seam)");
        }
        covered += (long long) (w.center_end - w.center_start);
    }
    expect(covered == n, tag + ": committed centres do not cover [0, n) exactly");
}

// resolve_long_form_window_frames() maps a requested window (0 = auto) plus the
// model's pos_emb_max_len onto the effective per-window frame ceiling. The
// load-bearing property is that pos_emb_max_len is a HARD ceiling that wins over
// the minimum-window floor: no window may exceed the encoder's trained
// positional range, even when that range is smaller than the floor.
void check_window_resolution() {
    constexpr int kAuto = 3750;
    constexpr int kMin  = 256;

    // Auto (requested == 0): the generous default, capped by pos_emb_max_len.
    expect(resolve_long_form_window_frames(0, 0, kAuto, kMin) == kAuto,
           "resolve: auto with unknown pos_emb keeps the default");
    expect(resolve_long_form_window_frames(0, 5000, kAuto, kMin) == kAuto,
           "resolve: auto under a large pos_emb keeps the default");
    expect(resolve_long_form_window_frames(0, 3000, kAuto, kMin) == 3000,
           "resolve: auto is capped down to pos_emb_max_len");

    // The fix: a pos_emb_max_len below the floor still wins, rather than being
    // bumped back up to kMin (which would exceed the trained positional range).
    expect(resolve_long_form_window_frames(0, 200, kAuto, kMin) == 200,
           "resolve: tiny pos_emb ceiling wins over the min-window floor");
    expect(resolve_long_form_window_frames(0, kMin, kAuto, kMin) == kMin,
           "resolve: pos_emb exactly at the floor is honoured");

    // Explicit request: floored to kMin, and also clamped to the ceiling.
    expect(resolve_long_form_window_frames(1024, 5000, kAuto, kMin) == 1024,
           "resolve: explicit request within range is untouched");
    expect(resolve_long_form_window_frames(100, 5000, kAuto, kMin) == kMin,
           "resolve: explicit request below the floor is raised to it");
    expect(resolve_long_form_window_frames(6000, 5000, kAuto, kMin) == 5000,
           "resolve: explicit request above pos_emb is capped to it");
    expect(resolve_long_form_window_frames(1000, 100, kAuto, kMin) == 100,
           "resolve: pos_emb ceiling wins even when it is below the floor");
}

// Drive the real trim + append over a synthetic encoder output and assert the
// stitched frames are exactly [0, 1, ... T_total-1] -- i.e. no frame is dropped
// or duplicated at any seam. Each window's synthetic encoder frame carries its
// own global frame index, so contiguity of the stitched values is a direct
// check of compute_window_trim + append_committed_frames.
//
// Units are mel frames; `sub` mel frames map to one encoder frame. center/ctx
// and n are chosen as multiples of sub so the subsampling divides exactly and
// the expected total is n / sub.
void check_stitch(int n_mel, int center_mel, int ctx_mel, int sub) {
    const std::string tag = "stitch n_mel=" + std::to_string(n_mel) +
                            " center_mel=" + std::to_string(center_mel) +
                            " ctx_mel=" + std::to_string(ctx_mel) +
                            " sub=" + std::to_string(sub);
    const std::vector<LongFormWindow> ws =
        plan_long_form_windows(n_mel, center_mel, ctx_mel);

    std::vector<float> stitched;
    for (const LongFormWindow & w : ws) {
        const int t_enc = w.window_len / sub;  // synthetic subsampling
        std::vector<float> win_enc((size_t) t_enc);
        const int base = w.window_start / sub;  // global encoder offset of the window
        for (int j = 0; j < t_enc; ++j) {
            win_enc[(size_t) j] = (float) (base + j);
        }
        const WindowTrim trim = compute_window_trim(w, t_enc, sub);
        const bool ok = append_committed_frames(stitched, win_enc,
                                                trim.left_drop, trim.center_cnt, 1);
        expect(ok, tag + ": append_committed_frames failed for a valid range");
    }

    expect((int) stitched.size() == n_mel / sub,
           tag + ": stitched frame count != n_mel / sub");
    for (size_t i = 0; i < stitched.size(); ++i) {
        if (stitched[i] != (float) i) {
            fail(tag + ": frame " + std::to_string(i) + " = " +
                 std::to_string((long long) stitched[i]) + " (gap/dup at seam)");
            break;
        }
    }
}

// append_committed_frames must reject an out-of-range request rather than
// silently appending nothing (which would let a caller over-count frames).
void check_append_bounds() {
    std::vector<float> src(10, 1.0f);  // 10 frames, stride 1
    std::vector<float> dst;
    expect(append_committed_frames(dst, src, 0, 10, 1), "append: full range should succeed");
    expect(dst.size() == 10, "append: full range appended wrong count");
    expect(!append_committed_frames(dst, src, 5, 10, 1),
           "append: out-of-range request must return false");
    expect(dst.size() == 10, "append: failed request must not modify dst");
    expect(append_committed_frames(dst, src, 0, 0, 1), "append: zero-count is a no-op success");
}

// The equal-length planner keeps every check_plan() coverage invariant, and
// every window is exactly min(center + 2 * ctx, n) long.
void check_plan_uniform(int n, int center, int ctx) {
    const std::string tag = "uniform n=" + std::to_string(n) +
                            " center=" + std::to_string(center) +
                            " ctx=" + std::to_string(ctx);
    const std::vector<LongFormWindow> ws =
        plan_long_form_windows_asymmetric(n, center, ctx, ctx);

    expect(!ws.empty(), tag + ": expected at least one window");
    if (ws.empty()) return;
    expect(ws.front().center_start == 0, tag + ": first centre must start at 0");
    expect(ws.back().center_end == n, tag + ": last centre must end at n_units");

    const int expected_len = std::min(center + 2 * std::max(ctx, 0), n);
    long long covered = 0;
    for (size_t i = 0; i < ws.size(); ++i) {
        const LongFormWindow & w = ws[i];
        const std::string wtag = tag + " window[" + std::to_string(i) + "]";
        expect(w.center_start < w.center_end, wtag + ": empty committed centre");
        expect(w.window_start >= 0, wtag + ": window_start < 0");
        expect(w.window_start + w.window_len <= n, wtag + ": window runs past n_units");
        expect(w.window_start <= w.center_start, wtag + ": window_start after centre");
        expect(w.window_start + w.window_len >= w.center_end,
               wtag + ": window ends before centre");
        expect(w.window_len == expected_len,
               wtag + ": window length " + std::to_string(w.window_len) +
               " != " + std::to_string(expected_len) + " (more than one encoder shape)");
        expect(w.is_final == (i + 1 == ws.size()), wtag + ": is_final mismatch");
        if (i + 1 < ws.size()) {
            expect(w.center_end == ws[i + 1].center_start,
                   wtag + ": centres not contiguous (gap or overlap at seam)");
        }
        covered += (long long) (w.center_end - w.center_start);
    }
    expect(covered == n, tag + ": committed centres do not cover [0, n) exactly");
}

// Stitching through the equal-length planner. When n_mel and the window are
// multiples of sub every window starts on the encoder-frame grid and the result
// is exact. Otherwise the final window, pulled left to full length, starts off
// the grid: the frames stay in order and the total stays within one frame of
// n_mel / sub, with at most one repeated frame at the final seam.
void check_stitch_uniform(int n_mel, int center_mel, int ctx_mel, int sub) {
    const std::string tag = "uniform stitch n_mel=" + std::to_string(n_mel) +
                            " center_mel=" + std::to_string(center_mel) +
                            " ctx_mel=" + std::to_string(ctx_mel);
    const std::vector<LongFormWindow> ws =
        plan_long_form_windows_asymmetric(n_mel, center_mel, ctx_mel, ctx_mel);

    std::vector<float> stitched;
    for (const LongFormWindow & w : ws) {
        const int t_enc = w.window_len / sub;
        std::vector<float> win_enc((size_t) t_enc);
        const int base = w.window_start / sub;
        for (int j = 0; j < t_enc; ++j) {
            win_enc[(size_t) j] = (float) (base + j);
        }
        const WindowTrim trim = compute_window_trim(w, t_enc, sub);
        expect(append_committed_frames(stitched, win_enc,
                                       trim.left_drop, trim.center_cnt, 1),
               tag + ": append_committed_frames failed for a valid range");
    }

    if (n_mel % sub == 0 && (center_mel + 2 * ctx_mel) % sub == 0) {
        expect((int) stitched.size() == n_mel / sub,
               tag + ": stitched frame count != n_mel / sub");
        for (size_t i = 0; i < stitched.size(); ++i) {
            if (stitched[i] != (float) i) {
                fail(tag + ": frame " + std::to_string(i) + " out of place (gap/dup at seam)");
                break;
            }
        }
        return;
    }

    const int n_enc = n_mel / sub;
    expect((int) stitched.size() >= n_enc && (int) stitched.size() <= n_enc + 1,
           tag + ": stitched frame count not within one frame of n_mel / sub");
    int repeats = 0;
    for (size_t i = 1; i < stitched.size(); ++i) {
        if (stitched[i] < stitched[i - 1]) {
            fail(tag + ": frames out of order at " + std::to_string(i));
            break;
        }
        if (stitched[i] == stitched[i - 1]) ++repeats;
        if (stitched[i] > stitched[i - 1] + 1.0f) {
            fail(tag + ": gap at " + std::to_string(i));
            break;
        }
    }
    expect(repeats <= 1, tag + ": more than one repeated frame");
}

// --window-sec / --overlap -> encoder frames, at the shipped 80 ms frame and
// TranscribePsy's pos_emb_max_len of 5000.
void check_seconds_conversion() {
    constexpr double kFrame = 160.0 * 8.0 / 16000.0;
    constexpr int    kMin   = 256;
    constexpr int    kMax   = 5000;

    auto conv = [&](double sec, double overlap, int max_frames, LongFormFrames & out) {
        out = LongFormFrames{};
        return long_form_frames_from_seconds(sec, overlap, kFrame, kMin, max_frames, out);
    };
    LongFormFrames f;

    expect(conv(60.0, 0.25, kMax, f) && f.window_frames == 750 && f.context_frames == 94,
           "seconds: 60 s / 25 % -> 750 / 94");
    expect(conv(60.0, 0.0, kMax, f) && f.window_frames == 750 && f.context_frames == -1,
           "seconds: 0 % overlap -> context -1 (no context), never 0 (auto)");
    expect(conv(60.0, 0.001, kMax, f) && f.context_frames == -1,
           "seconds: overlap rounding to 0 frames -> context -1");
    expect(conv(60.0, 0.5, kMax, f) && f.context_frames == 187,
           "seconds: 50 % clamps to window / 4 = 187");
    expect(conv(30.0, 0.25, kMax, f) && f.window_frames == 375 && f.context_frames == 47,
           "seconds: 30 s / 25 % -> 375 / 47");

    expect(conv(20.5, 0.25, kMax, f) && f.window_frames == 256,
           "seconds: 20.5 s is the shortest accepted window (256 frames)");
    expect(!conv(20.4, 0.25, kMax, f), "seconds: 20.4 s (255 frames) is rejected");
    expect(!conv(10.0, 0.25, kMax, f), "seconds: 10 s is rejected");
    expect(conv(400.0, 0.25, kMax, f) && f.window_frames == 5000,
           "seconds: 400 s (pos_emb_max_len) is accepted");
    expect(!conv(400.1, 0.25, kMax, f), "seconds: past pos_emb_max_len is rejected");
    expect(conv(1000.0, 0.25, 0, f) && f.window_frames == 12500,
           "seconds: max_window_frames <= 0 means no ceiling");

    expect(conv(0.0, 0.25, kMax, f) && f.window_frames == -1,
           "seconds: 0 s disables windowing");
    f = LongFormFrames{};
    expect(long_form_frames_from_seconds(0.0, 0.25, 0.0, kMin, kMax, f) && f.window_frames == -1,
           "seconds: 0 s disables windowing even without a usable frame length");
    expect(!long_form_frames_from_seconds(60.0, 0.25, 0.0, kMin, kMax, f),
           "seconds: a window needs a usable frame length");

    expect(!conv(-1.0, 0.25, kMax, f), "seconds: negative window is rejected");
    expect(!conv(60.0, -0.1, kMax, f), "seconds: negative overlap is rejected");
    expect(!conv(60.0, 0.6, kMax, f), "seconds: overlap above 0.5 is rejected");
    expect(!conv(std::nan(""), 0.25, kMax, f), "seconds: NaN window is rejected");
    expect(!conv(60.0, std::nan(""), kMax, f), "seconds: NaN overlap is rejected");
    expect(!conv(INFINITY, 0.25, kMax, f), "seconds: infinite window is rejected");
    expect(!conv(1e12, 0.25, 0, f), "seconds: a window past INT_MAX frames is rejected");
    expect(!conv(1e308, 0.25, 0, f), "seconds: a window overflowing to inf frames is rejected");

    LongFormFrames untouched;
    untouched.window_frames  = 123;
    untouched.context_frames = 45;
    expect(!long_form_frames_from_seconds(60.0, 0.9, kFrame, kMin, kMax, untouched) &&
           untouched.window_frames == 123 && untouched.context_frames == 45,
           "seconds: a rejected request leaves the output untouched");
}

// Without --window-sec the 60 s default is fitted to the model and must never
// make the conversion fail.
void check_default_window() {
    constexpr double kFrame = 160.0 * 8.0 / 16000.0;
    constexpr int    kMin   = 256;

    auto fitted_frames = [&](double frame_sec, int max_frames) {
        const double sec = fit_default_window_sec(60.0, frame_sec, kMin, max_frames);
        LongFormFrames f;
        if (!long_form_frames_from_seconds(sec, 0.25, frame_sec, kMin, max_frames, f)) {
            return -2;
        }
        return f.window_frames;
    };

    expect(fit_default_window_sec(60.0, kFrame, kMin, 5000) == 60.0,
           "default: 60 s is kept for TranscribePsy (pos_emb_max_len 5000)");
    expect(fitted_frames(kFrame, 5000) == 750, "default: 750 frames at 5000");
    expect(fitted_frames(kFrame, 0) == 750, "default: 750 frames with no ceiling");
    expect(fitted_frames(kFrame, 500) == 500, "default: shrunk to a 500-frame range");
    expect(fitted_frames(kFrame, 256) == 256, "default: shrunk to a 256-frame range");
    expect(fitted_frames(kFrame, 255) == -1, "default: range under 256 frames -> single pass");
    expect(fitted_frames(0.3, 5000) == 256, "default: raised to the 256-frame floor");
    expect(fitted_frames(0.0, 5000) == -1, "default: zero frame length -> single pass");
    expect(fitted_frames(std::nan(""), 5000) == -1, "default: NaN frame length -> single pass");
    expect(fitted_frames(INFINITY, 5000) == -1, "default: infinite frame length -> single pass");
    expect(fitted_frames(1e-12, 0) == -1, "default: window past INT_MAX frames -> single pass");
}

// The CLI plan: -1 context really is zero context, inputs that fit one window
// stay single-pass, and the 30 min case splits as expected.
void check_cli_plan() {
    constexpr int kSub = 8;

    LongFormFrames f;
    f.window_frames  = 750;
    f.context_frames = 94;
    expect(!resolve_cli_long_form_plan(f, kSub, 750LL * kSub).enabled,
           "cli plan: input of exactly one window stays single-pass");
    const LongFormCliPlan p = resolve_cli_long_form_plan(f, kSub, 750LL * kSub + 1);
    expect(p.enabled && p.window_frames == 750 && p.context_frames == 94 &&
           p.center_frames == 562 && p.sub == kSub,
           "cli plan: 60 s / 25 % -> centre 562");

    f.context_frames = -1;
    const LongFormCliPlan p0 = resolve_cli_long_form_plan(f, kSub, 100000);
    expect(p0.enabled && p0.context_frames == 0 && p0.center_frames == 750,
           "cli plan: context -1 -> no context, centre = whole window");

    f.context_frames = 187;
    expect(resolve_cli_long_form_plan(f, kSub, 100000).center_frames == 376,
           "cli plan: 50 % -> centre 376");

    f.window_frames = -1;
    expect(!resolve_cli_long_form_plan(f, kSub, 10000000).enabled,
           "cli plan: window -1 stays single-pass at any length");

    // 30 min of 16 kHz audio: 1 + 28,800,000 / 160 mel frames.
    const int n_mel = 1 + 30 * 60 * 16000 / 160;
    f.window_frames  = 750;
    f.context_frames = 94;
    const LongFormCliPlan p30 = resolve_cli_long_form_plan(f, kSub, n_mel);
    const std::vector<LongFormWindow> w25 = plan_long_form_windows_asymmetric(
        n_mel, p30.center_frames * kSub, p30.context_frames * kSub, p30.context_frames * kSub);
    expect(w25.size() == 41, "cli plan: 30 min at 60 s / 25 % -> 41 windows");
    expect(w25.size() == 41 &&
           w25[39].window_start == w25[40].window_start &&
           w25[39].window_len == w25[40].window_len,
           "cli plan: final centre (161 mel) under the context -> last two windows share a span");
    check_plan_uniform(n_mel, p30.center_frames * kSub, p30.context_frames * kSub);

    f.context_frames = -1;
    const LongFormCliPlan p30_0 = resolve_cli_long_form_plan(f, kSub, n_mel);
    const std::vector<LongFormWindow> w0 = plan_long_form_windows_asymmetric(
        n_mel, p30_0.center_frames * kSub, 0, 0);
    expect(w0.size() == 31, "cli plan: 30 min at 60 s / 0 % -> 31 windows");
    expect(w0.size() == 31 && w0[29].window_start != w0[30].window_start,
           "cli plan: 0 % overlap -> the last two windows have different spans");
    for (size_t i = 0; i + 1 < w0.size(); ++i) {
        expect(w0[i].window_start == w0[i].center_start && w0[i].window_len == 6000,
               "cli plan: 0 % overlap -> windows before the last one do not overlap");
    }
    check_plan_uniform(n_mel, p30_0.center_frames * kSub, 0);
}

}  // namespace

int main() {
    // Planner: single-window, exact multiples, remainders, degenerate context.
    check_plan(100, 1000, 100);
    check_plan(1000, 1000, 128);
    check_plan(1000, 250, 50);
    check_plan(4096, 512, 64);
    check_plan(1000, 300, 50);
    check_plan(5000, 700, 90);
    check_plan(1000, 200, 0);
    check_plan(500, 100, 100000);
    check_plan(1, 4096, 64);
    check_plan(4097, 4096, 8);

    // Planner: realistic long-form case (90 min of 16 kHz audio -> ~540k mel
    // frames at hop 160). The load-bearing assertion is that every window stays
    // bounded instead of the full single-pass length.
    {
        const int sub        = 8;
        const int center_mel = 3238 * sub;
        const int ctx_mel    = 256 * sub;
        const int n_mel      = 90 * 60 * 100;  // ~540,000 mel frames
        check_plan(n_mel, center_mel, ctx_mel);
        const std::vector<LongFormWindow> ws =
            plan_long_form_windows(n_mel, center_mel, ctx_mel);
        expect(ws.size() > 1, "long-form: expected the 90-min input to split into windows");
        const int bound = center_mel + 2 * ctx_mel;
        for (const LongFormWindow & w : ws) {
            expect(w.window_len <= bound, "long-form: a window exceeded the bounded size");
        }
    }

    // Window-size resolution: pos_emb_max_len is a hard ceiling over the floor.
    check_window_resolution();

    // Trim + append seam stitching (multiples of sub so subsampling is exact).
    check_stitch(2048, 256, 64, 8);   // several equal windows
    check_stitch(2048, 512, 128, 8);  // fewer, wider windows
    check_stitch(2560, 640, 128, 8);  // exact multiple, remainder-free
    check_stitch(4096, 256, 256, 8);  // context == center
    check_stitch(800, 3200, 64, 8);   // whole input fits one window

    check_append_bounds();

    // Equal-length planner: one window length per input, whatever the remainder.
    check_plan_uniform(100, 1000, 100);
    check_plan_uniform(1000, 250, 50);
    check_plan_uniform(1000, 300, 50);
    check_plan_uniform(5000, 700, 90);
    check_plan_uniform(1000, 200, 0);
    check_plan_uniform(4097, 4096, 8);
    check_plan_uniform(6001, 4496, 752);   // just over one 60 s / 25 % window
    check_plan_uniform(9000, 4496, 752);   // second-to-last window pulled left too
    check_plan_uniform(13457, 4496, 752);  // n not a multiple of 8

    check_stitch_uniform(2048, 256, 64, 8);
    check_stitch_uniform(2560, 640, 128, 8);
    check_stitch_uniform(48000, 4496, 752, 8);
    check_stitch_uniform(48001, 4496, 752, 8);  // final window off the frame grid
    check_stitch_uniform(13457, 4496, 752, 8);
    check_stitch_uniform(180001, 6000, 0, 8);   // 30 min, no overlap

    check_seconds_conversion();
    check_default_window();
    check_cli_plan();

    if (g_failures == 0) {
        std::printf("test-long-form-windows: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test-long-form-windows: %d failure(s)\n", g_failures);
    return 1;
}

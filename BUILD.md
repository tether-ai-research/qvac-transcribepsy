# Building parakeet.cpp for TranscribePsy

All builds use the ggml vendored in [`ggml/`](ggml/): tetherto/qvac-ext-ggml,
branch `speech`, commit `576c9735326a50ef1f2d54263ba4dac0d384d73d`. CMake picks
it up automatically (`PARAKEET_USE_SYSTEM_GGML` defaults to `OFF`), so a build
fetches nothing.

The Android and iOS builds that produced the published phone measurements used
this commit.

Audio input is 16 kHz mono WAV. The engine does not resample; convert first,
for example `ffmpeg -i input.m4a -ar 16000 -ac 1 audio.wav`.

The Arm ISA flags below matter for speed. `dotprod` and `i8mm` enable the int8
matrix kernels the published measurements use; without them ggml falls back to
baseline NEON and runs substantially slower. A binary built with them requires
a CPU that has both extensions (Apple A15 or later; on Android, Armv9 cores
such as Cortex-A710/X2 and later) and fails with an illegal-instruction error
on one that lacks them.

## Linux (x86-64 or arm64)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/parakeet` (CMake target `parakeet-cli`, output name
`parakeet`).

```bash
./build/parakeet --model transcribepsy-250m-eu-sub.gguf --wav audio.wav --threads 6
```

`--threads 0` uses every hardware thread; `--verbose` prints per-stage timings
to stderr. The transcript is printed to stdout.

Recordings of 60 s or longer are encoded in overlapping windows, so the
encoder's memory depends on the window rather than on the length of the
recording. `--window-sec S` (default 60, from 20.5 to 400) sets the window and
`--overlap F` (default 0.25, from 0 to 0.5) the fraction of each window shared
with its neighbours; `--window-sec 0` encodes the whole recording in one pass.
With `--verbose` the tool logs the window plan. The flags are ignored, with a
warning, when the Core ML encoder is active, since it windows long recordings
itself. [Long recordings](README.md#long-recordings) in the README gives the
recommended settings.

## macOS (Apple silicon)

The same commands build the command-line tool. Add `-DPARAKEET_COREML=ON` to
run the Core ML distribution on the Mac's Neural Engine:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPARAKEET_COREML=ON
cmake --build build -j
./build/parakeet --model transcribepsy-250m-eu-sub-coreml.gguf --wav audio.wav
```

The three Core ML files must sit together in one directory; see
[Core ML: file layout](#core-ml-file-layout-and-runtime-settings) below.

## Android (arm64-v8a)

Android NDK r27c, API level 28. `ANDROID_NDK=/path/to/ndk scripts/build-android.sh`
runs the commands below.

```bash
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm+fp16 \
  -DGGML_CPU_ALL_VARIANTS=OFF \
  -DGGML_BACKEND_DL=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENCL=OFF \
  -DGGML_OPENMP=OFF \
  -DPARAKEET_OPENMP=OFF \
  -DPARAKEET_COREML=OFF \
  -DPARAKEET_BUILD_EXECUTABLES=ON \
  -DPARAKEET_INSTALL=OFF
cmake --build build-android -j
```

Run the resulting `build-android/parakeet` on the device with the same
arguments as on Linux; the published measurements use `--threads 6`.

## iOS (arm64 device, static library)

Xcode command-line tools on macOS. This builds the engine as a static library
for an app to link; there is no iOS command-line binary. `scripts/build-ios.sh`
runs the same build and leaves `libparakeet_bundle.a` and a copy of the headers
in `build-ios/`.

```bash
ARCH=armv8.2-a+dotprod+i8mm+fp16
cmake -S . -B build-ios \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH="$ARCH" \
  -DCMAKE_C_FLAGS="-march=$ARCH" \
  -DCMAKE_CXX_FLAGS="-march=$ARCH" \
  -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DPARAKEET_OPENMP=OFF \
  -DPARAKEET_COREML=ON \
  -DPARAKEET_BUILD_EXECUTABLES=OFF \
  -DPARAKEET_INSTALL=OFF
cmake --build build-ios -j
libtool -static -o libparakeet_bundle.a $(find build-ios -name '*.a' | sort)
```

Link `libparakeet_bundle.a` into the app together with `CoreML.framework` and
`Foundation.framework`, and add `include/` to the header search path.

`-DPARAKEET_COREML=ON` is required for the Core ML path. Without it the Core ML
sources are not compiled and the engine never looks for the `.mlmodelc`
sidecars: with the full GGUF it silently runs the ggml encoder on the CPU
instead, and with the Core ML distribution's GGUF it refuses to load. The same
library serves the GGUF-only CPU path: with no sidecars next to the full GGUF
it uses ggml.

## Core ML: file layout and runtime settings

The Core ML models need iOS 18 or macOS 15 or later and a device with a Neural
Engine. The engine finds them by name next to the GGUF: it takes the GGUF file
name without its extension (dropping a trailing quantisation tag such as `-q8_0`, if there is
one) and appends `-encoder.mlmodelc` and `-decoder.mlmodelc`. Keep all three
files from the Core ML distribution in one directory, with these names:

```text
transcribepsy-250m-eu-sub-coreml.gguf
transcribepsy-250m-eu-sub-coreml-encoder.mlmodelc/
transcribepsy-250m-eu-sub-coreml-decoder.mlmodelc/
```

The GGUF in the Core ML distribution carries no encoder weights (17.3 MB); it
supplies the vocabulary, the mel filterbank, the decoder weights and the
configuration. The engine recognises such a file and runs the encoder on Core
ML without further settings. It cannot run on the CPU: without the encoder
sidecar, or on a build without `-DPARAKEET_COREML=ON`, loading fails with a
message saying so.

The sidecars also work next to the full GGUF from the GGUF distribution
(`transcribepsy-250m-eu-sub.gguf`) if they are renamed to match it:
`transcribepsy-250m-eu-sub-encoder.mlmodelc` and
`transcribepsy-250m-eu-sub-decoder.mlmodelc`. In that case set
`PARAKEET_COREML_ONLY=1` before creating the engine, so that the ggml copy of
the encoder weights, which the Neural Engine path never uses, is not loaded;
the transcripts are the same either way. If the names do not match, the full
GGUF silently runs on the CPU.

Every other setting defaults to the published configuration: 15 s segments,
6 s pause search, 16 frames of encoder context, and the batched decoder.

The mel spectrogram is computed by the engine on the CPU; it is not part of the
Core ML models. Decoding also runs inside the engine: the decoder model runs on
the Neural Engine, batched across segments. The caller passes audio in and
gets text back.

The engine has a C++ API, so a Swift app calls it through an Objective-C++
(`.mm`) wrapper:

```objc
#include <parakeet/engine.h>

parakeet::EngineOptions opts;
opts.model_gguf_path = "/path/to/transcribepsy-250m-eu-sub-coreml.gguf";
opts.n_threads = 6;
parakeet::Engine engine(opts);

parakeet::EngineResult r = engine.transcribe("/path/to/audio.wav");
NSString *text = [NSString stringWithUTF8String:r.text.c_str()];
```

A long recording is a single `transcribe()` call; the engine cuts it into
segments itself. For a batch of separate recordings, the next one's mel can be
computed on a worker thread while the current one is on the Neural Engine, so
the CPU and the Neural Engine work at the same time:

```cpp
#include <future>

std::future<parakeet::PrecomputedMel> next = std::async(std::launch::async,
    [&] { return engine.compute_mel_wav(files[0]); });
for (size_t i = 0; i < files.size(); ++i) {
    parakeet::PrecomputedMel mel = next.get();
    if (i + 1 < files.size())
        next = std::async(std::launch::async,
            [&, i] { return engine.compute_mel_wav(files[i + 1]); });
    parakeet::EngineResult r = engine.transcribe_mel(std::move(mel));
    // r.text
}
```

`compute_mel_wav` is the one `Engine` member that may run concurrently with a
transcription on the same engine; do not call `transcribe*` from two threads
at once. `transcribe_mel` returns exactly what `transcribe` would for the same
file.

## Tests

The unit tests need no model file and no audio:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPARAKEET_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build
```

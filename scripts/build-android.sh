#!/usr/bin/env bash
# Build the command-line tool for Android (arm64-v8a) with the flags behind
# the published phone measurements.
#
#   ANDROID_NDK=/path/to/android-ndk-r27c scripts/build-android.sh
#
# The binary lands in build-android/parakeet. Run it on the phone with
#
#   adb push build-android/parakeet transcribepsy-250m-eu-sub.gguf audio.wav /data/local/tmp/
#   adb shell 'cd /data/local/tmp && ./parakeet --model transcribepsy-250m-eu-sub.gguf --wav audio.wav --threads 6'
#
# The binary needs an Armv9 CPU (dotprod and i8mm); see BUILD.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NDK="${ANDROID_NDK:-${ANDROID_NDK_HOME:-}}"
[ -n "$NDK" ] || { echo "error: set ANDROID_NDK to the NDK directory (r27c was used)" >&2; exit 1; }
[ -f "$NDK/build/cmake/android.toolchain.cmake" ] || { echo "error: $NDK is not an Android NDK" >&2; exit 1; }
BUILD="${BUILD_DIR:-$ROOT/build-android}"
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
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
cmake --build "$BUILD" --target parakeet-cli -j "$jobs"

echo "Built $BUILD/parakeet"

#!/usr/bin/env bash
# Build the engine for iOS (arm64 device) as one static library for an app to
# link, with the Core ML path enabled.
#
#   scripts/build-ios.sh
#
# Output in build-ios/:
#   libparakeet_bundle.a   link it with CoreML.framework and Foundation.framework
#   include/               add it to the header search path
#
# Requires macOS with the Xcode command-line tools. The library needs an A15
# or later (dotprod and i8mm); see BUILD.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ "$(uname -s)" = Darwin ] || { echo "error: the iOS build needs macOS with Xcode" >&2; exit 1; }
BUILD="${BUILD_DIR:-$ROOT/build-ios}"
ARCH=armv8.2-a+dotprod+i8mm+fp16

cmake -S "$ROOT" -B "$BUILD/cmake" \
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
cmake --build "$BUILD/cmake" -j "$(sysctl -n hw.ncpu)"

libs=()
while IFS= read -r a; do libs+=("$a"); done < <(find "$BUILD/cmake" -name '*.a' | sort)
[ ${#libs[@]} -gt 0 ] || { echo "error: no static libraries were produced" >&2; exit 1; }
libtool -static -o "$BUILD/libparakeet_bundle.a" "${libs[@]}"
rm -rf "$BUILD/include"
mkdir -p "$BUILD/include"
cp -R "$ROOT/include/parakeet" "$BUILD/include/"

echo "Built $BUILD/libparakeet_bundle.a and $BUILD/include/"

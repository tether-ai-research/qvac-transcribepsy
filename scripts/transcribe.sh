#!/usr/bin/env bash
# Transcribe audio files with TranscribePsy in one command.
#
#   scripts/transcribe.sh [--coreml] [--threads N] [--window-sec S] [--overlap F]
#                         AUDIO [AUDIO...]
#
# On first use it builds the engine, downloads the model from the Hugging Face
# Hub into models/, and then prints one transcript per file. Any format ffmpeg
# can read is accepted; without ffmpeg, pass 16 kHz mono WAV.
#
#   --coreml        run on the Apple Neural Engine (macOS on Apple silicon)
#   --threads N     CPU threads (default 6)
#   --window-sec S  CPU only: encode recordings of S seconds or longer in
#                   overlapping windows of S seconds, so memory stays bounded
#                   (default 60, minimum 20.5, maximum 400; 0 = whole file at once)
#   --overlap F     CPU only: fraction of each window shared with its
#                   neighbours, 0 to 0.5 (default 0.25)
#
# Environment:
#   MODELS_DIR   where models are stored (default: <repo>/models)
#   GGUF_REPO    Hub repository of the GGUF build
#   COREML_REPO  Hub repository of the Core ML build
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
GGUF_REPO="${GGUF_REPO:-qvac/TranscribePsy-250M-EU-sub-gguf}"
COREML_REPO="${COREML_REPO:-qvac/TranscribePsy-250M-EU-sub-coreml}"

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "error: $*" >&2; exit 1; }

coreml=0
threads=6
window_args=()
files=()
while [ $# -gt 0 ]; do
  case "$1" in
    --coreml)     coreml=1 ;;
    --threads)    [ $# -ge 2 ] || die "--threads needs a value"; threads="$2"; shift ;;
    --window-sec) [ $# -ge 2 ] || die "--window-sec needs a value"; window_args+=(--window-sec "$2"); shift ;;
    --overlap)    [ $# -ge 2 ] || die "--overlap needs a value"; window_args+=(--overlap "$2"); shift ;;
    -h|--help)    usage 0 ;;
    -*)           die "unknown option $1 (see --help)" ;;
    *)            files+=("$1") ;;
  esac
  shift
done
[ ${#files[@]} -gt 0 ] || usage 1
for f in "${files[@]}"; do [ -f "$f" ] || die "no such file: $f"; done

if [ "$coreml" = 1 ] && [ "$(uname -s)" != Darwin ]; then
  die "--coreml needs macOS on Apple silicon; drop it to run on the CPU"
fi
if [ "$coreml" = 1 ] && [ ${#window_args[@]} -gt 0 ]; then
  echo "note: --window-sec/--overlap apply to the CPU build only; the Core ML encoder windows long recordings itself" >&2
  window_args=()
fi

# Build the command-line tool on first use. Later runs bring it up to date,
# which does nothing unless the sources changed.
if [ "$coreml" = 1 ]; then
  build="$ROOT/build-coreml"; cmake_extra=(-DPARAKEET_COREML=ON)
else
  build="$ROOT/build"; cmake_extra=()
fi
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
log="$build/build.log"
if [ ! -x "$build/parakeet" ]; then
  command -v cmake >/dev/null || die "cmake is required to build the engine"
  echo "Building the engine in $build ..." >&2
  mkdir -p "$build"
  if ! { cmake -S "$ROOT" -B "$build" -DCMAKE_BUILD_TYPE=Release ${cmake_extra[@]+"${cmake_extra[@]}"} \
         && cmake --build "$build" --target parakeet-cli -j "$jobs"; } >"$log" 2>&1; then
    tail -n 30 "$log" >&2
    die "the build failed; the full log is in $log"
  fi
elif command -v cmake >/dev/null && [ -f "$build/CMakeCache.txt" ]; then
  if [ -n "$(find "$ROOT/CMakeLists.txt" "$ROOT/cmake" "$ROOT/ggml" "$ROOT/include" "$ROOT/src" \
               -newer "$build/parakeet" -print -quit 2>/dev/null)" ]; then
    echo "Sources changed; rebuilding the engine in $build ..." >&2
  fi
  if ! cmake --build "$build" --target parakeet-cli -j "$jobs" >"$log" 2>&1; then
    tail -n 30 "$log" >&2
    die "rebuilding the engine failed; the full log is in $log (delete $build to start over)"
  fi
fi

# Download the model once.
hub_download() {  # repo dest [file...]
  local repo="$1" dest="$2"; shift 2
  if command -v hf >/dev/null; then
    hf download "$repo" "$@" --local-dir "$dest" >/dev/null
  elif command -v huggingface-cli >/dev/null; then
    huggingface-cli download "$repo" "$@" --local-dir "$dest" >/dev/null
  else
    die "the Hugging Face CLI (hf) is required to download the model: pipx install huggingface_hub, or pip install -U huggingface_hub in a virtual environment and activate it"
  fi
}
if [ "$coreml" = 1 ]; then
  model="$MODELS_DIR/coreml/transcribepsy-250m-eu-sub-coreml.gguf"
  if [ ! -f "$model" ] || [ ! -d "${model%.gguf}-encoder.mlmodelc" ] || [ ! -d "${model%.gguf}-decoder.mlmodelc" ]; then
    echo "Downloading $COREML_REPO ..." >&2
    hub_download "$COREML_REPO" "$MODELS_DIR/coreml"
  fi
else
  model="$MODELS_DIR/transcribepsy-250m-eu-sub.gguf"
  if [ ! -f "$model" ]; then
    echo "Downloading $GGUF_REPO ..." >&2
    hub_download "$GGUF_REPO" "$MODELS_DIR" transcribepsy-250m-eu-sub.gguf
  fi
fi
[ -f "$model" ] || die "model not found at $model"

# Transcribe.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
i=0
for f in "${files[@]}"; do
  i=$((i + 1))
  if command -v ffmpeg >/dev/null; then
    wav="$tmp/$i.wav"
    ffmpeg -nostdin -loglevel error -y -i "$f" -ar 16000 -ac 1 -c:a pcm_s16le "$wav" \
      || die "ffmpeg could not read $f"
  else
    case "$f" in
      *.wav|*.WAV) wav="$f" ;;
      *) die "$f is not a WAV file; install ffmpeg, or convert it to 16 kHz mono WAV first" ;;
    esac
  fi
  if ! text="$("$build/parakeet" --model "$model" --wav "$wav" --threads "$threads" \
                 ${window_args[@]+"${window_args[@]}"} 2>"$tmp/err")"; then
    grep -iE 'error|failed' "$tmp/err" >&2 || tail -n 20 "$tmp/err" >&2
    die "transcription failed for $f"
  fi
  if [ ${#files[@]} -eq 1 ]; then
    printf '%s\n' "$text"
  else
    printf '%s: %s\n' "$f" "$text"
  fi
done

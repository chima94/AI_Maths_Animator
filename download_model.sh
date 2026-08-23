#!/usr/bin/env bash
# Download the submission's public GGUF model from Hugging Face.
# The pinned revision and SHA-256 make evaluation reproducible.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/qwen_manim_animation_16bit.Q4_K_M.gguf"
PARTIAL_FILE="$MODEL_FILE.partial"
MODEL_REVISION="91a4ff76dacebc59f954698831e3ec1afc89135f"
MODEL_URL="https://huggingface.co/Chimanwakis/qwen_manim_animation_q4_k_m_v5/resolve/$MODEL_REVISION/qwen2.5-coder-3b-instruct.Q4_K_M.gguf"
EXPECTED_SHA256="74f3523c47193a67183ceee512087e38aa615848ff56402e8d6355144217a40a"

file_sha256() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum > /dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "error: neither sha256sum nor shasum was found" >&2
    return 1
  fi
}

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  ACTUAL_SHA256="$(file_sha256 "$MODEL_FILE")"
  if [[ "$ACTUAL_SHA256" == "$EXPECTED_SHA256" ]]; then
    echo "model already present and verified at $MODEL_FILE — skipping download"
    exit 0
  fi
  echo "existing model checksum does not match; downloading a verified copy" >&2
fi

echo "downloading public model (~1.80 GiB) to $MODEL_FILE"
if command -v curl > /dev/null 2>&1; then
  curl -L --fail --retry 3 --retry-all-errors --progress-bar -o "$PARTIAL_FILE" "$MODEL_URL"
elif command -v wget > /dev/null 2>&1; then
  wget --show-progress -O "$PARTIAL_FILE" "$MODEL_URL"
else
  echo "error: neither curl nor wget was found" >&2
  exit 1
fi

ACTUAL_SHA256="$(file_sha256 "$PARTIAL_FILE")"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "error: downloaded model failed SHA-256 verification" >&2
  echo "expected: $EXPECTED_SHA256" >&2
  echo "actual:   $ACTUAL_SHA256" >&2
  exit 1
fi

mv "$PARTIAL_FILE" "$MODEL_FILE"
echo "done: $MODEL_FILE"

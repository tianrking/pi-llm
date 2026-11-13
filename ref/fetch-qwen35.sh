#!/bin/sh
# Download the tested text GGUF, pin its revision and verify the LFS SHA-256.
set -eu
cd "$(dirname "$0")/.."
model=models/qwen3.5-2b-q4_0.gguf
sha=91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6
url=https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/8de6479d2743924f9dc499e3654d4e51ea0d4b9d/Qwen_Qwen3.5-2B-Q4_0.gguf
mkdir -p models
if [ -f "$model" ]; then
    printf '%s  %s\n' "$sha" "$model" | sha256sum -c -
    exit
fi
curl -fL --retry 3 --connect-timeout 20 -C - -o "$model.part" "$url"
printf '%s  %s\n' "$sha" "$model.part" | sha256sum -c -
mv "$model.part" "$model"
printf 'Installed %s (1,296,764,000 bytes).\n' "$model"

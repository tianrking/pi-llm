#!/bin/bash
# Build the pinned llama.cpp reference and fetch the tested Qwen3.5 text model.
set -eu
cd "$(dirname "$0")/.."
ref/fetch-qwen35.sh
[ -x ref/venv/bin/cmake ] || python3 -m venv ref/venv
ref/venv/bin/pip -q install cmake
revision=5676dfbe01e7ebe0387913680eee0ed39bf4a363
if [ ! -d ref/llama.cpp ]; then
    git init ref/llama.cpp
    git -C ref/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp
    git -C ref/llama.cpp fetch --depth 1 origin "$revision"
    git -C ref/llama.cpp checkout --detach "$revision"
fi
if [ "$(git -C ref/llama.cpp rev-parse HEAD)" != "$revision" ]; then
    echo "Reference checkout differs from $revision; keep local changes and choose a separate reference directory." >&2
    exit 1
fi
cd ref/llama.cpp
git log -1 --format='llama.cpp commit %h %cd'
../venv/bin/cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DCMAKE_C_FLAGS="-mcpu=cortex-a72" -DCMAKE_CXX_FLAGS="-mcpu=cortex-a72" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
../venv/bin/cmake --build build -j3 --target llama-completion llama-bench
echo SETUP_DONE

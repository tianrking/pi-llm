#!/bin/bash
# Run the same ChatML prompt through llama.cpp and pillm with greedy decoding.
# usage: ref/compare.sh "question" [threads] [n_tokens]
cd "$(dirname "$0")/.."
M=${MODEL:-models/qwen3.5-2b-q4_0.gguf}
Q=${1:-What is the capital of France?}
T=${2:-4}
N=${3:-64}
P="<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
$Q<|im_end|>
<|im_start|>assistant
"
if ./ggufdump "$M" | grep -q 'general.architecture.*"qwen35"'; then
    P="$P<think>

</think>

"
fi
echo "=== llama.cpp ($T threads)"
ref/llama.cpp/build/bin/llama-completion -m "$M" -p "$P" -n $N -t $T --temp 0 --repeat-penalty 1.05 --flash-attn off -c 4096 -no-cnv --no-display-prompt \
    --no-warmup 2> ref/last-llama.log
echo
grep -E "eval time|load time" ref/last-llama.log | sed 's/^[^:]*: *//'
echo "=== pillm ($T threads)"
./pillm -m "$M" -p "$P" --raw -n $N -t $T --temp 0

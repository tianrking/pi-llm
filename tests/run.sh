#!/bin/bash
# Regression tests. usage: tests/run.sh [model.gguf]   (or: make test)
# Needs a model; compares with llama.cpp when ref/llama.cpp has been built.
cd "$(dirname "$0")/.."
M=${1:-${MODEL:-models/qwen3.5-2b-q4_0.gguf}}
[ -f "$M" ] || { echo "no model at $M (see ref/setup.sh)"; exit 2; }
hybrid=0
./ggufdump "$M" | grep -q 'general.architecture.*"qwen35"' && hybrid=1
fail=0
ok()  { echo "ok    $1"; }
bad() { echo "FAIL  $1"; fail=1; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

Q="What is the capital of France? Answer in one short sentence."
P="<|im_start|>user
$Q<|im_end|>
<|im_start|>assistant
"
if [ "$hybrid" = 1 ]; then P="$P<think>

</think>

"; fi

bench/expcheck > /dev/null;                           check "vector exp matches libm" "[ $? = 0 ]"

a=$(./pillm -m "$M" -p "$P" --raw --temp 0 -n 12 2>/dev/null)
check "greedy answer mentions Paris" 'echo "$a" | grep -q Paris'

b=$(./pillm -m "$M" -p "$P" --raw --temp 0 -n 12 2>/dev/null)
check "greedy decoding is deterministic" '[ "$a" = "$b" ]'

c=$(./pillm -m "$M" -p "$P" --raw --temp 0 -n 12 -t 1 -tb 1 2>/dev/null)
check "same answer with one thread" '[ "$a" = "$c" ]'

e=$(./pillm -m "$M" -p "$P" --raw --temp 0 -n 6 --exact-head 2>/dev/null)
check "two-pass output layer agrees with the exact one" 'case "$a" in "$e"*) true;; *) false;; esac'

s1=$(./pillm -m "$M" -p "$Q" --seed 11 -n 12 2>/dev/null); s2=$(./pillm -m "$M" -p "$Q" --seed 11 -n 12 2>/dev/null)
check "sampling repeats with the same seed" '[ -n "$s1" ] && [ "$s1" = "$s2" ]'

long=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' $(seq 1 30))
./pillm -m "$M" -c 64 -p "$long" -n 2 > /dev/null 2> tests/.err; rc=$?
check "oversized prompt is refused cleanly" '[ $rc = 1 ] && grep -q "does not fit" tests/.err'

printf 'Hi.\nTell me a story.\nAnother one.\n' | ./pillm -m "$M" -c 120 -n 50 --temp 0 > /dev/null 2> tests/.err; rc=$?
check "chat survives a full context" '[ $rc = 0 ] && grep -q "dropped the oldest" tests/.err'

./pillm -m "$M" -n 2 --temp 0 -p "<|foo|> <| 😀 $(printf 'a%.0s' $(seq 1 150)) １２３ 　'll" > /dev/null 2>&1
check "odd input does not crash" "[ $? = 0 ]"

# server
PORT=18473
./pillm -m "$M" --serve $PORT -c 1024 > /dev/null 2>&1 &
pid=$!
for i in $(seq 1 50); do curl -s -m 1 localhost:$PORT/health > /dev/null && break; sleep 0.2; done
check "server answers /health" '[ "$(curl -s -m 5 localhost:$PORT/health)" = "{\"status\":\"ok\"}" ]'
check "server serves the chat page" 'curl -s -m 5 localhost:$PORT/ | grep -q "<title>pillm</title>"'
j=$(curl -s -m 200 localhost:$PORT/v1/chat/completions -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":12,\"temperature\":0}")
check "chat completion returns valid JSON mentioning Paris" 'echo "$j" | python3 -c "import json,sys; j=json.load(sys.stdin); sys.exit(0 if \"Paris\" in j[\"choices\"][0][\"message\"][\"content\"] else 1)"'
st=$(curl -s -m 200 localhost:$PORT/v1/chat/completions -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":6,\"temperature\":0,\"stream\":true}")
if [ "$hybrid" = 1 ]; then
    check "streaming ends with [DONE] and rebuilds recurrent state" 'echo "$st" | grep -q "data: \[DONE\]" && echo "$st" | grep -q "\"cached_tokens\":0"'
else
    check "streaming ends with [DONE] and reuses the cached prompt" 'echo "$st" | grep -q "data: \[DONE\]" && echo "$st" | grep -q "\"cached_tokens\":[1-9]"'
fi
check "bad request gets 400" '[ "$(curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:$PORT/v1/chat/completions -d "{}")" = 400 ]'
check "server is still alive" 'curl -s -m 5 localhost:$PORT/health > /dev/null'
kill $pid 2> /dev/null; wait $pid 2> /dev/null

L=ref/llama.cpp/build/bin/llama-completion
if [ -x $L ]; then
    r=$($L -m "$M" -p "$P" -n 12 -c 512 -t 2 --temp 0 --repeat-penalty 1.05 --flash-attn off -no-cnv --no-display-prompt --no-warmup 2>/dev/null | sed 's/ \[end of text\]//')
    check "greedy answer equals llama.cpp's" '[ "$(echo $a)" = "$(echo $r)" ]'
else
    echo "skip  llama.cpp comparison (ref/llama.cpp not built)"
fi
rm -f tests/.err
[ $fail = 0 ] && echo "all tests passed" || echo "some tests FAILED"
exit $fail

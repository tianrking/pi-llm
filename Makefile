CC = gcc
CFLAGS = -O3 -mcpu=cortex-a72 -fopenmp -Wall -Wextra -Wno-unused-parameter
MODEL ?= models/qwen3.5-2b-q4_0.gguf
all: pillm ggufdump bench/limits bench/cycles bench/expcheck
pillm: src/pillm.c src/server.h src/gguf.c src/gguf.h src/kernels.h src/qwen35.h src/unicode.h
	$(CC) $(CFLAGS) -o $@ src/pillm.c src/gguf.c -lm
ggufdump: src/ggufdump.c src/gguf.c src/gguf.h
	$(CC) $(CFLAGS) -o $@ src/ggufdump.c src/gguf.c
clean:
	rm -f tests/numerics pillm ggufdump bench/limits bench/cycles bench/expcheck
bench/limits: bench/limits.c src/kernels.h
	$(CC) $(CFLAGS) -o $@ bench/limits.c
bench/cycles: bench/cycles.c src/kernels.h
	$(CC) $(CFLAGS) -o $@ bench/cycles.c
bench/expcheck: bench/expcheck.c src/kernels.h
	$(CC) $(CFLAGS) -o $@ bench/expcheck.c -lm
tests/numerics: tests/numerics.c src/pillm.c src/qwen35.h src/server.h src/unicode.h src/kernels.h src/gguf.c src/gguf.h
	$(CC) $(CFLAGS) -o $@ tests/numerics.c src/gguf.c -lm
check: pillm tests/numerics
	./tests/numerics
	python3 tests/legacy-smoke.py
test: all check
	tests/run.sh "$(MODEL)"
	@if ./ggufdump "$(MODEL)" | grep -q 'general.architecture.*"qwen35"'; then python3 tests/qwen35.py "$(MODEL)"; fi
.PHONY: all clean check test

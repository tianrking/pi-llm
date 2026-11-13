// limits — where is the ceiling: memory or CPU?
//
// For each working-set size (fits L1 / fits L2 / far larger than cache) and each
// thread count, measures how many GB/s of weight bytes go through:
//   read   : the cheapest possible NEON pass (load + xor). Pure memory speed.
//   q4     : the real Q4_0 x Q8 dot-product kernel, rows of 896 inputs.
//   q8     : the real Q8_0 x Q8 dot-product kernel.
//   q4x4/q8x4 : the four-rows-at-a-time versions of the same kernels.
// In-cache numbers are what the CPU can do; out-of-cache numbers are what DRAM allows.
#include "../src/kernels.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N_IN 896
static volatile float sink_f;
static volatile uint64_t sink_u;

static uint64_t pass_read(const uint8_t *b, size_t n) {
    uint8x16_t a0 = vdupq_n_u8(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 64 <= n; i += 64) {
        a0 = veorq_u8(a0, vld1q_u8(b + i));      a1 = veorq_u8(a1, vld1q_u8(b + i + 16));
        a2 = veorq_u8(a2, vld1q_u8(b + i + 32)); a3 = veorq_u8(a3, vld1q_u8(b + i + 48));
    }
    return vgetq_lane_u64(vreinterpretq_u64_u8(veorq_u8(veorq_u8(a0, a1), veorq_u8(a2, a3))), 0);
}
static float pass_q4(const uint8_t *b, size_t n, const int8_t *xq, const float *xd) {
    size_t row = N_IN / QK * 18; float s = 0;
    for (size_t o = 0; o + row <= n; o += row) s += dot_q4(b + o, xq, xd, N_IN / QK);
    return s;
}
static float pass_q4x4(const uint8_t *b, size_t n, const int8_t *xq, const float *xd, const int16_t *negc) {
    size_t row = N_IN / QK * 18; float y[4], s = 0;
    for (size_t o = 0; o + 4 * row <= n; o += 4 * row) { dot4_q4(b + o, row, xq, xd, negc, N_IN / QK, y); s += y[0] + y[3]; }
    return s;
}
static float pass_q8x4(const uint8_t *b, size_t n, const int8_t *xq, const float *xd) {
    size_t row = N_IN / QK * 34; float y[4], s = 0;
    for (size_t o = 0; o + 4 * row <= n; o += 4 * row) { dot4_q8(b + o, row, xq, xd, N_IN / QK, y); s += y[0] + y[3]; }
    return s;
}
static float pass_q8(const uint8_t *b, size_t n, const int8_t *xq, const float *xd) {
    size_t row = N_IN / QK * 34; float s = 0;
    for (size_t o = 0; o + row <= n; o += row) s += dot_q8(b + o, xq, xd, N_IN / QK);
    return s;
}

// Returns total GB/s across all threads. Each thread has a private buffer of `size` bytes.
static double run(int mode, size_t size, int threads, double seconds) {
    double total = 0;
    #pragma omp parallel num_threads(threads) reduction(+:total)
    {
        uint8_t *b = aligned_alloc(64, size);
        for (size_t i = 0; i < size; i++) b[i] = (uint8_t)(i * 2654435761u >> 24);
        int8_t xq[N_IN]; float xd[N_IN / QK];
        for (int i = 0; i < N_IN; i++) xq[i] = (int8_t)(i * 37 % 255 - 127);
        for (int i = 0; i < N_IN / QK; i++) xd[i] = 0.01f;
        int16_t negc[N_IN / QK * 8]; q4_prep(xq, N_IN / QK, negc);
        #pragma omp barrier
        double t0 = omp_get_wtime(), t; size_t bytes = 0;
        do {
            if (mode == 0) sink_u = pass_read(b, size);
            else if (mode == 1) sink_f = pass_q4(b, size, xq, xd);
            else if (mode == 2) sink_f = pass_q8(b, size, xq, xd);
            else if (mode == 3) sink_f = pass_q4x4(b, size, xq, xd, negc);
            else sink_f = pass_q8x4(b, size, xq, xd);
            bytes += size;
        } while ((t = omp_get_wtime()) - t0 < seconds);
        total += bytes / (t - t0) / 1e9;
        free(b);
    }
    return total;
}

int main(int argc, char **argv) {
    double secs = argc > 1 ? atof(argv[1]) : 1.0;
    static const struct { const char *name; size_t size; } ws[] = {
        { "L1   (16 KB/thread)", 16 << 10 }, { "L2  (128 KB/thread)", 128 << 10 }, { "DRAM (192 MB/thread)", 192 << 20 } };
    static const char *modes[] = { "read", "q4", "q8", "q4x4", "q8x4" };
    int only = argc > 2 ? atoi(argv[2]) : -1;       // optional: restrict to one working set
    printf("GB/s of weight bytes, all threads combined\n");
    printf("%-22s %-5s %8s %8s %8s %8s\n", "working set", "pass", "1 thr", "2 thr", "3 thr", "4 thr");
    for (int w = 0; w < 3; w++)
        for (int m = 0; m < 5; m++) {
            if (only >= 0 && w != only) continue;
            printf("%-22s %-5s", ws[w].name, modes[m]);
            for (int t = 1; t <= 4; t++) { printf(" %8.2f", run(m, ws[w].size, t, secs)); fflush(stdout); }
            printf("\n");
        }
    return 0;
}

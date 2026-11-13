// cycles — true CPU cycles per kernel iteration, read from the PMU, so the result
// does not depend on the clock speed, thermal throttling or preemption.
#define _GNU_SOURCE
#include "../src/kernels.h"
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static int ev(int type, int cfg) { struct perf_event_attr a = {0}; a.type = type; a.size = sizeof a; a.config = cfg; a.exclude_kernel = 1; a.exclude_hv = 1;
    return syscall(SYS_perf_event_open, &a, 0, -1, -1, 0); }
static uint64_t rdc(int fd) { uint64_t v = 0; if (read(fd, &v, 8) != 8) return 0; return v; }
static volatile float sink; static volatile uint64_t sinku;
int main(void) {
    int fc = ev(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES), fi = ev(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    if (fc < 0) { perror("perf_event_open"); return 1; }
    enum { NB = 28, ROWS = 16 };
    uint8_t *w = aligned_alloc(64, 65536); for (int i = 0; i < 65536; i++) w[i] = i * 131 >> 3;
    int8_t xq[NB * 32]; float xd[NB]; int16_t negc[NB * 8]; float y[4];
    for (int i = 0; i < NB * 32; i++) xq[i] = i * 37 % 255 - 127;
    for (int i = 0; i < NB; i++) xd[i] = 0.01f;
    q4_prep(xq, NB, negc);
    int reps = 200000;
    for (int k = 0; k < 5; k++) {
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t c0 = rdc(fc), i0 = rdc(fi);
        if (k == 0) for (int r = 0; r < reps; r++) { dot4_q4(w + (r % 4) * 2016, 504, xq, xd, negc, NB, y); sink = y[0]; }
        if (k == 1) for (int r = 0; r < reps; r++) { dot4_q8(w + (r % 4) * 3808, 952, xq, xd, NB, y); sink = y[0]; }
        if (k == 2) for (int r = 0; r < reps * 4; r++) sink = dot_q4(w + (r % 16) * 504, xq, xd, NB);
        if (k == 3) for (int r = 0; r < reps * 4; r++) sink = dot_q8(w + (r % 16) * 952, xq, xd, NB);
        if (k == 4) for (int r = 0; r < reps; r++) { uint8x16_t a = vdupq_n_u8(0); for (int i = 0; i < 2048; i += 16) a = veorq_u8(a, vld1q_u8(w + i)); sinku = vgetq_lane_u64(vreinterpretq_u64_u8(a), 0); }
        uint64_t c = rdc(fc) - c0, in = rdc(fi) - i0; clock_gettime(CLOCK_MONOTONIC, &t1);
        double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        const char *nm[] = {"dot4_q4", "dot4_q8", "dot_q4 ", "dot_q8 ", "load+eor"};
        double units = k == 4 ? reps * 128.0 : reps * 4.0 * NB;       // row-blocks (or 16-byte loads)
        printf("%s  %.1f cycles, %.1f instr per %s | effective clock %.0f MHz\n", nm[k], c / units, in / units, k == 4 ? "16-byte load" : "row-block", c / s / 1e6);
    }
}

// kernels.h — the inner loops, written for the Cortex-A72 (ARMv8.0 NEON, no SDOT).
#ifndef KERNELS_H
#define KERNELS_H
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define QK 32                       // values per quantisation block
typedef __fp16 f16;

static inline float rd_f16(const uint8_t *p) { f16 h; memcpy(&h, p, 2); return (float)h; }

// Quantise n floats (n % 32 == 0) into int8 blocks with one fp16-rounded scale per block.
static inline void quantize_q8(const float *x, int n, int8_t *q, float *d) {
    for (int b = 0; b < n / QK; b++, x += QK, q += QK) {
        float32x4_t mx = vdupq_n_f32(0);
        for (int i = 0; i < QK; i += 4) mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(x + i)));
        float s = vmaxvq_f32(mx) / 127.0f, id = s ? 1.0f / s : 0.0f;
        d[b] = (float)(f16)s;
        for (int i = 0; i < QK; i += 8) {
            int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i), id));
            int32x4_t c = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i + 4), id));
            vst1_s8(q + i, vmovn_s16(vcombine_s16(vmovn_s32(a), vmovn_s32(c))));
        }
    }
}

// One Q4_0 weight row (nb blocks of 18 bytes) · quantised activations.
// The A72 has no SDOT, so: widening multiply to int16, add (cannot overflow:
// |(nibble-8) * q| <= 8*127, four terms per lane), pairwise-widen to int32.
static inline float dot_q4(const uint8_t *w, const int8_t *xq, const float *xd, int nb) {
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const int8x16_t off = vdupq_n_s8(8);
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++, w += 18, xq += QK) {
        uint8x16_t nib = vld1q_u8(w + 2);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(nib, m4)), off);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(nib, 4)), off);
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16);
        int16x8_t s = vmull_s8(vget_low_s8(lo), vget_low_s8(x0));
        s = vmlal_high_s8(s, lo, x0);
        s = vmlal_s8(s, vget_low_s8(hi), vget_low_s8(x1));
        s = vmlal_high_s8(s, hi, x1);
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(vpaddlq_s16(s)), rd_f16(w) * xd[b]);
    }
    return vaddvq_f32(acc);
}

// One Q8_0 weight row (nb blocks of 34 bytes) · quantised activations.
static inline float dot_q8(const uint8_t *w, const int8_t *xq, const float *xd, int nb) {
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++, w += 34, xq += QK) {
        int8x16_t w0 = vld1q_s8((const int8_t *)w + 2), w1 = vld1q_s8((const int8_t *)w + 18);
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16);
        int32x4_t s = vpaddlq_s16(vmull_s8(vget_low_s8(w0), vget_low_s8(x0)));
        s = vpadalq_s16(s, vmull_high_s8(w0, x0));
        s = vpadalq_s16(s, vmull_s8(vget_low_s8(w1), vget_low_s8(x1)));
        s = vpadalq_s16(s, vmull_high_s8(w1, x1));
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), rd_f16(w) * xd[b]);
    }
    return vaddvq_f32(acc);
}

// exp(x) for four floats: range reduction to 2^n * exp(b) with |b| <= ln2/2 and a
// degree-5 polynomial, after the single-precision routine in Arm's optimized-routines.
// Relative error stays below 1e-6 (see bench/expcheck.c).
static inline float32x4_t v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vmaxvq_u32(c)) return vfmaq_f32(k, j, k);
    // |n| > 126: the result would overflow or underflow the simple scaling; split it in two
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// g[i] = silu(g[i]) * u[i], n a multiple of 4.   silu(x) = x / (1 + exp(-x))
static inline void silu_mul(float *g, const float *u, int n) {
    const float32x4_t one = vdupq_n_f32(1);
    for (int i = 0; i < n; i += 4) {
        float32x4_t x = vld1q_f32(g + i);
        vst1q_f32(g + i, vmulq_f32(vdivq_f32(x, vaddq_f32(one, v_expf(vnegq_f32(x)))), vld1q_f32(u + i)));
    }
}

// ---- four rows at a time ---------------------------------------------------
// The single-row kernels above spend much of their time on per-block overhead:
// reloading the activations, converting one scale, one float accumulate. Doing
// four weight rows against the same activation block shares all of that, and
// reduces the four integer sums to one vector whose lanes are the four rows.

// PREFETCH: on this SoC a plain sequential read tops out near 3.1 GB/s; asking for
// lines 0.5-4 KB ahead lifts that by 10-30%. While working on four rows the kernels
// request the four rows after them, which is that far ahead for every matrix shape here.
#ifndef PILLM_NO_PREFETCH
#define PREFETCH(p) __builtin_prefetch((p), 0, 0)
#else
#define PREFETCH(p) ((void)0)
#endif

// Per activation block: -8 * (pairwise lane sums). Starting the multiply-accumulate
// chain from this removes the "nibble - 8" subtraction from the inner loop.
static inline void q4_prep(const int8_t *xq, int nb, int16_t *negc) {
    for (int b = 0; b < nb; b++, xq += QK, negc += 8) {
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16);
        int16x8_t c = vaddq_s16(vaddl_s8(vget_low_s8(x0), vget_high_s8(x0)), vaddl_s8(vget_low_s8(x1), vget_high_s8(x1)));
        vst1q_s16(negc, vnegq_s16(vshlq_n_s16(c, 3)));
    }
}

// Per activation block: scale * sum of the 32 quantised values (for Q4_1's offset term).
static inline void xsum_prep(const int8_t *xq, const float *xd, int nb, float *xs) {
    for (int b = 0; b < nb; b++, xq += QK)
        xs[b] = xd[b] * (float)(vaddlvq_s8(vld1q_s8(xq)) + vaddlvq_s8(vld1q_s8(xq + 16)));
}

// Q6_K stores four 32-value streams in each 128-value half-block.
static inline void q6_unpack16(const uint8_t *w, int half, int part, int8x16_t q[4]) {
    const uint8_t *ql = w + half * 64 + part * 16, *qh = w + 128 + half * 32 + part * 16;
    uint8x16_t l0 = vld1q_u8(ql), l1 = vld1q_u8(ql + 32), high = vld1q_u8(qh);
    const uint8x16_t mask = vdupq_n_u8(15), two = vdupq_n_u8(3);
    const int8x16_t off = vdupq_n_s8(32);
    q[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, mask), vshlq_n_u8(vandq_u8(high, two), 4))), off);
    q[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, mask), vshlq_n_u8(vandq_u8(vshrq_n_u8(high, 2), two), 4))), off);
    q[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(high, 4), two), 4))), off);
    q[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vshrq_n_u8(high, 6), 4))), off);
}

static inline float q6_dot16(int8x16_t q, const float *x) {
    int16x8_t lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_s8(vget_high_s8(q));
    float32x4_t a = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(x));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(x + 4));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(x + 8));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(x + 12));
    return vaddvq_f32(a);
}

static inline float dot_q6_f32(const uint8_t *w, const float *x, int n) {
    float sum = 0;
    for (int b = 0; b < n / 256; b++, w += 210, x += 256) {
        float d = rd_f16(w + 208);
        const int8_t *sc = (const int8_t *)w + 192;
        for (int h = 0; h < 2; h++)
            for (int j = 0; j < 2; j++) {
                int8x16_t q[4]; q6_unpack16(w, h, j, q);
                for (int k = 0; k < 4; k++)
                    sum += d * sc[h * 8 + j + 2 * k] * q6_dot16(q[k], x + h * 128 + j * 16 + k * 32);
            }
    }
    return sum;
}

// Q6_K uses one float activation scale per 256 values (Q8_K).
static inline void quantize_q8k(const float *x, int n, int8_t *q, float *d) {
    for (int b = 0; b < n / 256; b++, x += 256, q += 256) {
        float peak = 0, amax = 0;
        for (int i = 0; i < 256; i++) if (fabsf(x[i]) > amax) { amax = fabsf(x[i]); peak = x[i]; }
        float inv = peak ? -127.0f / peak : 0;
        d[b] = peak ? 1.0f / inv : 0;
        for (int i = 0; i < 256; i += 8) {
            int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i), inv));
            int32x4_t c = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i + 4), inv));
            vst1_s8(q + i, vmovn_s16(vcombine_s16(vmovn_s32(a), vmovn_s32(c))));
        }
    }
}

static inline float dot_q6_q8k(const uint8_t *w, const int8_t *x, const float *xd, int n) {
    float sum = 0;
    for (int b = 0; b < n / 256; b++, w += 210, x += 256) {
        const int8_t *sc = (const int8_t *)w + 192;
        int32_t acc = 0;
        for (int h = 0; h < 2; h++)
            for (int j = 0; j < 2; j++) {
                int8x16_t q[4]; q6_unpack16(w, h, j, q);
                for (int k = 0; k < 4; k++) {
                    int8x16_t v = vld1q_s8(x + h * 128 + j * 16 + k * 32);
                    int16x8_t prod = vmull_s8(vget_low_s8(q[k]), vget_low_s8(v));
                    prod = vmlal_high_s8(prod, q[k], v);
                    acc += sc[h * 8 + j + 2 * k] * vaddlvq_s16(prod);
                }
            }
        sum += rd_f16(w + 208) * xd[b] * acc;
    }
    return sum;
}

static inline float32x4_t scales4(const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3, float xd) {
    uint16x4_t h = vld1_dup_u16((const uint16_t *)w0);
    h = vld1_lane_u16((const uint16_t *)w1, h, 1);
    h = vld1_lane_u16((const uint16_t *)w2, h, 2);
    h = vld1_lane_u16((const uint16_t *)w3, h, 3);
    return vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(h)), xd);
}

// y[0..3] = four consecutive Q4_0 rows (stride bytes apart) · activations.
static inline void dot4_q4(const uint8_t *w, size_t stride, const int8_t *xq, const float *xd,
                           const int16_t *negc, int nb, float *y) {
    const uint8_t *w0 = w, *w1 = w + stride, *w2 = w1 + stride, *w3 = w2 + stride;
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const uint8_t *pf = w3 + stride;            // the next four rows, see PREFETCH below
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++, w0 += 18, w1 += 18, w2 += 18, w3 += 18, xq += QK, negc += 8, pf += 72) {
        PREFETCH(pf); PREFETCH(pf + 36);
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16);
        int8x8_t x0l = vget_low_s8(x0), x1l = vget_low_s8(x1);
        int16x8_t c = vld1q_s16(negc);
#define Q4ROW(s, wp) { \
        uint8x16_t nib = vld1q_u8(wp + 2); \
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(nib, m4)), hi = vreinterpretq_s8_u8(vshrq_n_u8(nib, 4)); \
        s = vmlal_s8(c, vget_low_s8(lo), x0l); s = vmlal_high_s8(s, lo, x0); \
        s = vmlal_s8(s, vget_low_s8(hi), x1l); s = vmlal_high_s8(s, hi, x1); }
        int16x8_t s0, s1, s2, s3;
        Q4ROW(s0, w0) Q4ROW(s1, w1) Q4ROW(s2, w2) Q4ROW(s3, w3)
#undef Q4ROW
        // each lane is within +-4064, so two levels of pairwise int16 adds cannot overflow
        int32x4_t t = vpaddlq_s16(vpaddq_s16(vpaddq_s16(s0, s1), vpaddq_s16(s2, s3)));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(t), scales4(w0, w1, w2, w3, xd[b]));
    }
    vst1q_f32(y, acc);
}

// y[0..3] = four consecutive Q4_1 rows · activations. A Q4_1 block is 20 bytes: fp16
// scale d, fp16 offset m, 32 unsigned nibbles; a weight is d * nibble + m. So a block
// contributes d * sum(nibble * x) + m * sum(x); xs[b] holds sum(x) already scaled.
static inline void dot4_q41(const uint8_t *w, size_t stride, const int8_t *xq, const float *xd,
                            const float *xs, int nb, float *y) {
    const uint8_t *w0 = w, *w1 = w + stride, *w2 = w1 + stride, *w3 = w2 + stride;
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const uint8_t *pf = w3 + stride;
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++, w0 += 20, w1 += 20, w2 += 20, w3 += 20, xq += QK, pf += 80) {
        PREFETCH(pf); PREFETCH(pf + 40);
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16);
        int8x8_t x0l = vget_low_s8(x0), x1l = vget_low_s8(x1);
#define Q41ROW(s, wp) { \
        uint8x16_t nib = vld1q_u8(wp + 4); \
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(nib, m4)), hi = vreinterpretq_s8_u8(vshrq_n_u8(nib, 4)); \
        s = vmull_s8(vget_low_s8(lo), x0l); s = vmlal_high_s8(s, lo, x0); \
        s = vmlal_s8(s, vget_low_s8(hi), x1l); s = vmlal_high_s8(s, hi, x1); }
        int16x8_t s0, s1, s2, s3;
        Q41ROW(s0, w0) Q41ROW(s1, w1) Q41ROW(s2, w2) Q41ROW(s3, w3)
#undef Q41ROW
        // each lane is within +-7620, so two levels of pairwise int16 adds reach at most 30480
        int32x4_t t = vpaddlq_s16(vpaddq_s16(vpaddq_s16(s0, s1), vpaddq_s16(s2, s3)));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(t), scales4(w0, w1, w2, w3, xd[b]));
        acc = vaddq_f32(acc, scales4(w0 + 2, w1 + 2, w2 + 2, w3 + 2, xs[b]));
    }
    vst1q_f32(y, acc);
}

// y[0..3] = four consecutive Q8_0 rows · activations.
static inline void dot4_q8(const uint8_t *w, size_t stride, const int8_t *xq, const float *xd, int nb, float *y) {
    const uint8_t *w0 = w, *w1 = w + stride, *w2 = w1 + stride, *w3 = w2 + stride;
    const uint8_t *pf = w3 + stride;
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++, w0 += 34, w1 += 34, w2 += 34, w3 += 34, xq += QK, pf += 136) {
        PREFETCH(pf); PREFETCH(pf + 48); PREFETCH(pf + 96);
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16);
        int8x8_t x0l = vget_low_s8(x0), x1l = vget_low_s8(x1);
#define Q8ROW(t, wp) { \
        int8x16_t a = vld1q_s8((const int8_t *)wp + 2), e = vld1q_s8((const int8_t *)wp + 18); \
        int16x8_t p = vmlal_high_s8(vmull_s8(vget_low_s8(a), x0l), a, x0);   /* 2 * 127 * 127 fits int16 */ \
        int16x8_t r = vmlal_high_s8(vmull_s8(vget_low_s8(e), x1l), e, x1); \
        t = vpadalq_s16(vpaddlq_s16(p), r); }
        int32x4_t t0, t1, t2, t3;
        Q8ROW(t0, w0) Q8ROW(t1, w1) Q8ROW(t2, w2) Q8ROW(t3, w3)
#undef Q8ROW
        int32x4_t t = vpaddq_s32(vpaddq_s32(t0, t1), vpaddq_s32(t2, t3));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(t), scales4(w0, w1, w2, w3, xd[b]));
    }
    vst1q_f32(y, acc);
}

#endif

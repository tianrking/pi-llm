// pillm — a from-scratch LLM inference engine for the Raspberry Pi 4 (BCM2711, Cortex-A72).
// Stage 1: correct, readable, NEON integer kernels, one token at a time.
// Supports qwen2, llama and text-only qwen35 GGUF models.
//
// Quantised activations and fp16 K/V follow the reference GGUF computation paths.
// Qwen3.5 also uses Q8_K activations and fp32 recurrent state.
#define _GNU_SOURCE
#include "gguf.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "kernels.h"

#define MAXL 64
#define BMAX 16                     // tokens per batch when reading a prompt
#define NCAND 256                   // candidate tokens handed to the sampler

// ---------------------------------------------------------------- model
typedef struct { const uint8_t *w; int type; } wt;      // a quantised matrix: data and GGML type

typedef struct {
    const float *attn_norm, *ffn_norm, *bq, *bk, *bv;   // biases are null when the model has none
    wt wq, wk, wv, wo, gate, up, down;
    int recurrent;
    wt qkv, z, alpha, beta;
    const float *q_norm, *k_norm, *conv_w, *dt, *a, *ssm_norm;
    float *conv_state, *ssm_state;
} layer;

typedef struct {
    char arch[32];
    int n_layer, d, ff, n_head, n_kv_head, hd, kvd, n_vocab, n_ctx;
    int hybrid, n_rot, conv_kernel, ssm_dim, ssm_heads;
    float *qg, *qkv, *z, *alpha, *beta, *conv;
    const float *quant_x;
    int rope_neox;                  // RoPE pairs element i with i + hd/2 (Qwen2) instead of i + 1 (Llama)
    float eps, rope_base;
    wt tok_embd, out;               // out is tok_embd itself when the file has no output.weight
    size_t out_row;                 // bytes per row of out
    const float *out_norm;
    layer L[MAXL];
    f16 *kc, *vc;                   // [layer][pos][kvd]
    // scratch, sized for BMAX tokens at once
    float *x, *h, *q, *k, *v, *att, *o, *g, *u, *logits, *xd, *xs, *p, *rc, *rs;
    int8_t *xq; int16_t *negc;
    int nt_gen, nt_batch;           // threads for one token (memory-bound) and for batches (CPU-bound)
    int two_pass;                   // output layer: shortlist with the Q4_0 embedding matrix, then exact rows
    int n_cand, cand_id[NCAND];     // the NCAND most likely next tokens ...
    float cand_logit[NCAND];        // ... and their exact logits
} model;

static gguf G;

static void die(const char *fmt, const char *a) { fprintf(stderr, "pillm: "); fprintf(stderr, fmt, a); fprintf(stderr, "\n"); exit(1); }

static int blk_bytes(int type) { return type == GGML_Q4_0 ? 18 : type == GGML_Q4_1 ? 20 : type == GGML_Q8_0 ? 34 : 0; }

static const float *f32_tensor(const char *name, int required) {
    const gguf_tensor *t = gguf_tensor_find(&G, name);
    if (!t) { if (required) die("missing tensor %s", name); return 0; }
    if (t->type != GGML_F32) die("tensor %s is not F32", name);
    return (const float *)t->data;
}

// A weight matrix the fast kernels can run: Q4_0, Q4_1 or Q8_0, rows a multiple of four.
static wt weight(const char *name) {
    const gguf_tensor *t = gguf_tensor_find(&G, name);
    if (!t) die("missing tensor %s", name);
    if (!blk_bytes(t->type)) die("tensor %s: unsupported quantisation (pillm runs Q4_0, Q4_1 and Q8_0 weights)", name);
    if (t->dim[0] % QK || t->dim[1] % 4) die("tensor %s: unsupported shape", name);
    return (wt){ t->data, (int)t->type };
}

// Dequantise one row of a Q4_0 / Q8_0 / Q6_K matrix into w[d].
static void row_f32(wt W, int d, size_t r, float *w) {
    if (W.type == GGML_Q4_0) {
        const uint8_t *p = W.w + r * (d / QK) * 18;
        for (int b = 0; b < d / QK; b++, p += 18, w += QK) {
            float s = rd_f16(p);
            for (int j = 0; j < 16; j++) { w[j] = ((p[2 + j] & 15) - 8) * s; w[16 + j] = ((p[2 + j] >> 4) - 8) * s; }
        }
    } else if (W.type == GGML_Q8_0) {
        const uint8_t *p = W.w + r * (d / QK) * 34;
        for (int b = 0; b < d / QK; b++, p += 34, w += QK) {
            float s = rd_f16(p);
            for (int j = 0; j < QK; j++) w[j] = (int8_t)p[2 + j] * s;
        }
    } else {
        // Q6_K: 256 weights per 210-byte block: 128 bytes of low nibbles, 64 bytes of
        // high bit-pairs, 16 int8 sub-block scales, one fp16 scale.
        const uint8_t *p = W.w + r * (d / 256) * 210;
        for (int blk = 0; blk < d / 256; blk++, p += 210) {
            const uint8_t *ql = p, *qh = p + 128; const int8_t *sc = (const int8_t *)p + 192;
            float s = rd_f16(p + 208);
            for (int n = 0; n < 2; n++, w += 128, ql += 64, qh += 32, sc += 8)
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = ((ql[l] & 15) | ((qh[l] & 3) << 4)) - 32, q2 = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32, q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    w[l] = s * sc[is] * q1; w[l + 32] = s * sc[is + 2] * q2;
                    w[l + 64] = s * sc[is + 4] * q3; w[l + 96] = s * sc[is + 6] * q4;
                }
        }
    }
}

static float dot_f32(const float *a, const float *b, int n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    for (int i = 0; i < n; i += 8) { s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i)); s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)); }
    return vaddvq_f32(vaddq_f32(s0, s1));
}

// Many models share one matrix between the input embedding and the output layer, and
// the file may store it twice at different precisions. Confirm that on a few rows.
static int tied_embeddings(const model *m) {
    float a[m->d], b[m->d];
    for (int k = 0; k < 8; k++) {
        size_t r = (size_t)m->n_vocab * k / 8 + 100;
        row_f32(m->tok_embd, m->d, r, a); row_f32(m->out, m->d, r, b);
        if (dot_f32(a, b, m->d) < 0.98f * sqrtf(dot_f32(a, a, m->d) * dot_f32(b, b, m->d))) return 0;
    }
    return 1;
}

static uint64_t hp_u(const model *m, const char *key) { char k[96]; snprintf(k, sizeof k, "%s.%s", m->arch, key); return gguf_u(&G, k, 0); }
static float hp_f(const model *m, const char *key, float dflt) { char k[96]; snprintf(k, sizeof k, "%s.%s", m->arch, key); return gguf_f(&G, k, dflt); }

static void qwen35_init(model *m, int n_ctx);

static void model_init(model *m, int n_ctx) {
    const gguf_kv *ka = gguf_find(&G, "general.architecture");
    if (!ka || ka->type != GT_STR) die("%s", "no general.architecture in the file");
    const uint8_t *c = ka->val; gstr an = gguf_arr_str(ka, 0, &c);
    snprintf(m->arch, sizeof m->arch, "%.*s", (int)an.n, an.p);
    // Same block layout (RMSNorm, grouped-query attention with RoPE, SwiGLU); they differ in
    // RoPE pairing and in whether Q/K/V have biases, which the tensor list tells us.
    if (!strcmp(m->arch, "qwen35")) { qwen35_init(m, n_ctx); return; }
    if (!strcmp(m->arch, "qwen2")) m->rope_neox = 1;
    else if (!strcmp(m->arch, "llama")) m->rope_neox = 0;
    else die("architecture '%s' is not supported (pillm runs qwen2, llama and qwen35)", m->arch);

    m->n_layer = hp_u(m, "block_count");
    m->d = hp_u(m, "embedding_length");
    m->ff = hp_u(m, "feed_forward_length");
    m->n_head = hp_u(m, "attention.head_count");
    m->n_kv_head = hp_u(m, "attention.head_count_kv");
    if (!m->n_kv_head) m->n_kv_head = m->n_head;
    m->eps = hp_f(m, "attention.layer_norm_rms_epsilon", 1e-5f);
    m->rope_base = hp_f(m, "rope.freq_base", 10000.0f);
    if (!m->n_layer || m->n_layer > MAXL || !m->n_head || m->d % QK || m->ff % QK || m->d % m->n_head ||
        m->n_head % m->n_kv_head || (m->d / m->n_head) % 4) die("%s", "unsupported model dimensions");
    m->hd = m->d / m->n_head;
    m->n_rot = m->hd;
    m->kvd = m->hd * m->n_kv_head;
    m->n_ctx = n_ctx;

    const gguf_tensor *te = gguf_tensor_find(&G, "token_embd.weight");
    if (!te || (te->type != GGML_Q4_0 && te->type != GGML_Q8_0)) die("%s", "token_embd.weight: need Q4_0 or Q8_0");
    m->n_vocab = te->dim[1];
    m->tok_embd = (wt){ te->data, (int)te->type };
    const gguf_tensor *ow = gguf_tensor_find(&G, "output.weight");
    if (!ow) { m->out = m->tok_embd; m->two_pass = 0; }                     // tied and stored once
    else if (blk_bytes(ow->type) || (ow->type == GGML_Q6_K && m->d % 256 == 0)) m->out = (wt){ ow->data, (int)ow->type };
    else die("%s", "output.weight: need Q4_0, Q8_0 or Q6_K");
    m->out_row = m->out.type == GGML_Q6_K ? (size_t)(m->d / 256) * 210 : (size_t)(m->d / QK) * blk_bytes(m->out.type);
    if (m->n_vocab % 4) die("%s", "vocabulary size must be a multiple of four");
    m->out_norm = f32_tensor("output_norm.weight", 1);
    for (int i = 0; i < m->n_layer; i++) {
        char n[64]; layer *l = &m->L[i];
#define NAME(suffix) (snprintf(n, sizeof n, "blk.%d." suffix, i), n)
        l->attn_norm = f32_tensor(NAME("attn_norm.weight"), 1); l->ffn_norm = f32_tensor(NAME("ffn_norm.weight"), 1);
        l->bq = f32_tensor(NAME("attn_q.bias"), 0); l->bk = f32_tensor(NAME("attn_k.bias"), 0); l->bv = f32_tensor(NAME("attn_v.bias"), 0);
        l->wq = weight(NAME("attn_q.weight")); l->wk = weight(NAME("attn_k.weight")); l->wv = weight(NAME("attn_v.weight"));
        l->wo = weight(NAME("attn_output.weight"));
        l->gate = weight(NAME("ffn_gate.weight")); l->up = weight(NAME("ffn_up.weight")); l->down = weight(NAME("ffn_down.weight"));
#undef NAME
    }
    // The shortlist needs a cheaper copy of the output matrix: a separate Q4_0 embedding.
    m->two_pass = m->two_pass && ow && m->tok_embd.type == GGML_Q4_0 && m->out.type != GGML_Q4_0 && tied_embeddings(m);
    size_t kv = (size_t)m->n_layer * n_ctx * m->kvd;
    m->kc = malloc(kv * sizeof(f16)); m->vc = malloc(kv * sizeof(f16));
    if (!m->kc || !m->vc) die("%s", "out of memory for the K/V cache; use a smaller -c");
#define A(f, n) m->f = aligned_alloc(64, ((size_t)(n) * sizeof *m->f + 63) & ~63ull)
    A(x, BMAX * m->d); A(h, BMAX * m->d); A(q, BMAX * m->d); A(k, BMAX * m->kvd); A(v, BMAX * m->kvd);
    A(att, BMAX * m->d); A(o, BMAX * m->d); A(g, BMAX * m->ff); A(u, BMAX * m->ff); A(logits, m->n_vocab);
    A(xd, BMAX * m->ff / QK); A(xs, BMAX * m->ff / QK); A(xq, BMAX * m->ff); A(negc, BMAX * m->ff / QK * 8);
    A(p, (size_t)omp_get_max_threads() * (n_ctx + 4)); A(rc, BMAX * m->hd / 2); A(rs, BMAX * m->hd / 2);
#undef A
}

// ---------------------------------------------------------------- kernels
// Four rows of any supported type against one activation vector.
static inline void dot4(int type, const uint8_t *w, size_t stride, const int8_t *xq, const float *xd,
                        const int16_t *negc, const float *xs, int nb, float *y) {
    if (type == GGML_Q4_0) dot4_q4(w, stride, xq, xd, negc, nb, y);
    else if (type == GGML_Q4_1) dot4_q41(w, stride, xq, xd, xs, nb, y);
    else dot4_q8(w, stride, xq, xd, nb, y);
}

// Y[b][n_out] = W[n_out][n_in] · X[b] for the B activation vectors last passed to
// quant(), optional bias.
//
// Rows go four at a time. For a batch, the loops are arranged so that everything
// the inner kernel touches fits the A72's 32 KB L1: the four weight rows are cut
// into tiles of TILE blocks, and each tile meets the activations of CHUNK tokens
// before moving on. The weights still come from memory only once per batch.
#define TILE 28                     // blocks of weights per tile: 4 rows x 504 bytes
#define CHUNK 8                     // tokens per inner pass: 8 x 1.5 KB of activations
static void matvec(const model *m, wt W, int n_in, int n_out, int B, const float *bias, float *y, int ldy, int nt) {
    if (W.type == GGML_Q6_K && m->hybrid) {
        int8_t q[(size_t)B * n_in]; float scales[(size_t)B * (n_in / 256)];
        for (int b = 0; b < B; b++) quantize_q8k(m->quant_x + (size_t)b * n_in, n_in,
                                                q + (size_t)b * n_in, scales + b * (n_in / 256));
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int r = 0; r < n_out; r++)
            for (int b = 0; b < B; b++)
                y[(size_t)b * ldy + r] = dot_q6_q8k(W.w + (size_t)r * (n_in / 256) * 210,
                    q + (size_t)b * n_in, scales + b * (n_in / 256), n_in) + (bias ? bias[r] : 0);
        return;
    }
    if (W.type == GGML_F32 || W.type == GGML_Q6_K) {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int r = 0; r < n_out; r++)
            for (int b = 0; b < B; b++) {
                const float *x = m->quant_x + (size_t)b * n_in;
                float v = W.type == GGML_F32 ? dot_f32((const float *)W.w + (size_t)r * n_in, x, n_in)
                    : dot_q6_f32(W.w + (size_t)r * (n_in / 256) * 210, x, n_in);
                y[(size_t)b * ldy + r] = v + (bias ? bias[r] : 0);
            }
        return;
    }
    int nb = n_in / QK, bs = blk_bytes(W.type); size_t row = (size_t)nb * bs;
    const int8_t *xq = m->xq; const float *xd = m->xd, *xs = m->xs; const int16_t *negc = m->negc;
    // Hybrid state must see the same accumulation order during prefill and decode.
    if (B == 1 && !m->hybrid) {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int r = 0; r < n_out / 4; r++) dot4(W.type, W.w + 4 * r * row, row, xq, xd, negc, xs, nb, y + 4 * r);
    } else {
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int r = 0; r < n_out / 4; r++) {
            const uint8_t *w = W.w + 4 * r * row;
            for (int b0 = 0; b0 < B; b0 += CHUNK) {
                int bn = B - b0 < CHUNK ? B - b0 : CHUNK;
                float32x4_t acc[CHUNK];
                for (int t = 0; t < nb; t += TILE) {
                    int tn = nb - t < TILE ? nb - t : TILE;
                    for (int b = 0; b < bn; b++) {
                        float part[4]; size_t o = (size_t)(b0 + b) * nb + t;
                        dot4(W.type, w + t * bs, row, xq + o * QK, xd + o, negc + o * 8, xs + o, tn, part);
                        acc[b] = t ? vaddq_f32(acc[b], vld1q_f32(part)) : vld1q_f32(part);
                    }
                }
                for (int b = 0; b < bn; b++) vst1q_f32(y + (size_t)(b0 + b) * ldy + 4 * r, acc[b]);
            }
        }
    }
    if (bias) for (int b = 0; b < B; b++) { float *yb = y + (size_t)b * ldy; for (int r = 0; r < n_out; r++) yb[r] += bias[r]; }
}

static void rmsnorm(const float *x, const float *w, int n, float eps, float *y) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * s * w[i];
}

// Rotary position encoding. The angles depend only on the position, so the sines and
// cosines are worked out once per token (rope_table) and reused by every layer and head.
static void rope_table(float *cs, float *sn, int hd, int pos, float base) {
    float scale = powf(base, -2.0f / hd), theta = (float)pos;
    for (int i = 0; i < hd / 2; i++, theta *= scale) { cs[i] = cosf(theta); sn[i] = sinf(theta); }
}

// neox pairs element i with i + hd/2, otherwise with i + 1.
static void rope(float *x, int n_heads, int hd, const float *cs, const float *sn, int neox) {
    int half = hd / 2;
    for (int h = 0; h < n_heads; h++, x += hd)
        for (int i = 0; i < half; i++) {
            float *pa = neox ? x + i : x + 2 * i, *pb = neox ? x + i + half : x + 2 * i + 1, a = *pa, b = *pb;
            *pa = a * cs[i] - b * sn[i]; *pb = a * sn[i] + b * cs[i];
        }
}

// ---------------------------------------------------------------- forward
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// Time spent per stage, printed with --prof.
enum { P_EMBED, P_ATT_MAT, P_ATT, P_FFN_MAT, P_FFN, P_HEAD, P_N };
static const char *prof_name[P_N] = { "embed+norm", "attn matrices", "attention", "ffn matrices", "ffn silu+quant", "output head" };
static double prof[P_N], prof_t;
#define STAGE(i) { double t_ = now(); prof[i] += t_ - prof_t; prof_t = t_; }

// Quantise B vectors of n floats (stride n) into the shared activation buffers.
static void quant(model *m, const float *x, int n, int B) {
    m->quant_x = x;
    for (int b = 0; b < B; b++) {
        quantize_q8(x + (size_t)b * n, n, m->xq + (size_t)b * n, m->xd + b * (n / QK));
        q4_prep(m->xq + (size_t)b * n, n / QK, m->negc + b * (n / QK) * 8);
        xsum_prep(m->xq + (size_t)b * n, m->xd + b * (n / QK), n / QK, m->xs + b * (n / QK));
    }
}

// One attention head for one token: q (hd floats) against cache slots 0..pos.
static void attn_head(const model *m, const f16 *kc, const f16 *vc, const float *q, int pos, int kh, float *p, float *out) {
    int hd = m->hd, kvd = m->kvd;
    float inv = 1.0f / sqrtf((float)hd), mx = -INFINITY, sum = 0;
    f16 q16[hd];
    for (int i = 0; i < hd; i++) q16[i] = q[i];
    for (int t = 0; t <= pos; t++) {
        const f16 *kt = kc + (size_t)t * kvd + kh * hd;
        float32x4_t a = vdupq_n_f32(0);
        for (int i = 0; i < hd; i += 4) a = vmlaq_f32(a, m->hybrid ? vld1q_f32(q + i) : vcvt_f32_f16(vld1_f16(q16 + i)), vcvt_f32_f16(vld1_f16(kt + i)));
        p[t] = vaddvq_f32(a) * inv;
        if (p[t] > mx) mx = p[t];
    }
    float32x4_t vs = vdupq_n_f32(0), vmx = vdupq_n_f32(mx);
    for (int t = pos + 1; t < ((pos + 4) & ~3); t++) p[t] = -INFINITY;      // pad to a multiple of 4: exp gives 0
    for (int t = 0; t <= pos; t += 4) { float32x4_t e = v_expf(vsubq_f32(vld1q_f32(p + t), vmx)); vst1q_f32(p + t, e); vs = vaddq_f32(vs, e); }
    sum = vaddvq_f32(vs);
    float32x4_t acc[hd / 4];
    for (int i = 0; i < hd / 4; i++) acc[i] = vdupq_n_f32(0);
    for (int t = 0; t <= pos; t++) {
        const f16 *vt = vc + (size_t)t * kvd + kh * hd;
        float w = m->hybrid ? p[t] / sum : (float)(f16)(p[t] / sum);
        for (int i = 0; i < hd / 4; i++) acc[i] = vmlaq_n_f32(acc[i], vcvt_f32_f16(vld1_f16(vt + 4 * i)), w);
    }
    for (int i = 0; i < hd / 4; i++) vst1q_f32(out + 4 * i, acc[i]);
}

#include "qwen35.h"

// Run B tokens (B <= BMAX). They occupy K/V cache slots slot..slot+B-1 and carry
// positions pos..pos+B-1; the two differ once old context has been dropped (see
// shift_context). With want_logits, the candidates for the next token are filled in.
static void forward(model *m, const int *tokens, int B, int slot, int pos, int want_logits) {
    int d = m->d, hd = m->hd, kvd = m->kvd, ff = m->ff, nt = B > 1 ? m->nt_batch : m->nt_gen;
    prof_t = now();

    for (int b = 0; b < B; b++) {
        row_f32(m->tok_embd, d, tokens[b], m->x + b * d);                   // embedding rows
        rope_table(m->rc + b * hd / 2, m->rs + b * hd / 2, m->n_rot, pos + b, m->rope_base);
    }

    for (int li = 0; li < m->n_layer; li++) {
        const layer *l = &m->L[li];
        f16 *kc = m->kc + ((size_t)li * m->n_ctx) * kvd, *vc = m->vc + ((size_t)li * m->n_ctx) * kvd;

        // --- attention
        for (int b = 0; b < B; b++) rmsnorm(m->x + b * d, l->attn_norm, d, m->eps, m->h + b * d);
        quant(m, m->h, d, B);
        STAGE(P_EMBED)
        if (m->hybrid) {
            qwen35_attention(m, l, B, li, slot, pos, nt);
        } else {
            matvec(m, l->wq, d, d, B, l->bq, m->q, d, nt);
            matvec(m, l->wk, d, kvd, B, l->bk, m->k, kvd, nt);
            matvec(m, l->wv, d, kvd, B, l->bv, m->v, kvd, nt);
            STAGE(P_ATT_MAT)
            for (int b = 0; b < B; b++) {                                       // all keys/values of the batch go in first
                rope(m->q + b * d, m->n_head, hd, m->rc + b * hd / 2, m->rs + b * hd / 2, m->rope_neox);
                rope(m->k + b * kvd, m->n_kv_head, hd, m->rc + b * hd / 2, m->rs + b * hd / 2, m->rope_neox);
                f16 *kd = kc + (size_t)(slot + b) * kvd, *vd = vc + (size_t)(slot + b) * kvd;
                for (int i = 0; i < kvd; i++) { kd[i] = m->k[b * kvd + i]; vd[i] = m->v[b * kvd + i]; }
            }
            int group = m->n_head / m->n_kv_head;
            #pragma omp parallel for schedule(dynamic) num_threads(nt) if(B > 1)
            for (int i = 0; i < B * m->n_head; i++) {
                int b = i / m->n_head, h = i % m->n_head;
                attn_head(m, kc, vc, m->q + b * d + h * hd, slot + b, h / group,
                          m->p + (size_t)omp_get_thread_num() * (m->n_ctx + 4), m->att + b * d + h * hd);
            }
            quant(m, m->att, d, B);
            STAGE(P_ATT)
            matvec(m, l->wo, d, d, B, 0, m->o, d, nt);
            for (int i = 0; i < B * d; i++) m->x[i] += m->o[i];
            STAGE(P_ATT_MAT)

        }

        // --- feed forward: down( silu(gate(h)) * up(h) )
        for (int b = 0; b < B; b++) rmsnorm(m->x + b * d, l->ffn_norm, d, m->eps, m->h + b * d);
        quant(m, m->h, d, B);
        STAGE(P_EMBED)
        matvec(m, l->gate, d, ff, B, 0, m->g, ff, nt);
        matvec(m, l->up, d, ff, B, 0, m->u, ff, nt);
        STAGE(P_FFN_MAT)
        silu_mul(m->g, m->u, B * ff);
        quant(m, m->g, ff, B);
        STAGE(P_FFN)
        matvec(m, l->down, ff, d, B, 0, m->o, d, nt);
        for (int i = 0; i < B * d; i++) m->x[i] += m->o[i];
        STAGE(P_FFN_MAT)
    }

    if (!want_logits) return;
    rmsnorm(m->x + (B - 1) * d, m->out_norm, d, m->eps, m->h);
    quant(m, m->h, d, 1);
    STAGE(P_EMBED)
    // Output layer. Only the few most likely tokens matter, and reading all of the
    // full-precision matrix (the largest tensor) for them is the single biggest memory
    // cost per token. When the file also has a Q4_0 copy of the same matrix, rank the
    // vocabulary with that, then compute exact logits for the NCAND best. The true top
    // 20 were always inside the approximate top 46 in testing, far inside NCAND.
    if (m->two_pass) matvec(m, m->tok_embd, d, m->n_vocab, 1, 0, m->logits, m->n_vocab, m->nt_gen);
    else matvec(m, m->out, d, m->n_vocab, 1, 0, m->logits, m->n_vocab, m->nt_gen);
    int n = 0, lo = 0;                                                      // lo = index of the smallest kept
    for (int i = 0; i < m->n_vocab; i++) {
        float v = m->logits[i];
        if (n < NCAND) { m->cand_id[n] = i; m->cand_logit[n] = v; if (v < m->cand_logit[lo]) lo = n; n++; continue; }
        if (v <= m->cand_logit[lo]) continue;
        m->cand_id[lo] = i; m->cand_logit[lo] = v;
        for (int j = 0; j < NCAND; j++) if (m->cand_logit[j] < m->cand_logit[lo]) lo = j;
    }
    m->n_cand = n;
    if (m->two_pass)
        for (int j = 0; j < n; j++) {
            if (m->out.type == GGML_Q8_0) m->cand_logit[j] = dot_q8(m->out.w + m->cand_id[j] * m->out_row, m->xq, m->xd, d / QK);
            else { float w[d]; row_f32(m->out, d, m->cand_id[j], w); m->cand_logit[j] = dot_f32(w, m->h, d); }
        }
    STAGE(P_HEAD)
}

// ---------------------------------------------------------------- tokenizer (GPT-2 style byte-level BPE)
enum { PRE_QWEN2, PRE_QWEN35, PRE_LLAMA3, PRE_GPT2, PRE_SMOLLM };   // pre-tokenizer rule sets, named as in llama.cpp
typedef struct { gstr *tok; int32_t *type; gstr *merge; uint32_t *th, *mh; int n_tok, n_merge, pre, n_special, *special; uint32_t b2u[256]; int16_t u2b[1024]; } vocab;
#define HBITS 20
#define HSIZE (1u << HBITS)

static uint32_t hash(const char *p, size_t n) { uint32_t h = 2166136261u; while (n--) h = (h ^ (uint8_t)*p++) * 16777619u; return h; }
static void hput(uint32_t *tab, const gstr *arr, uint32_t idx) {
    uint32_t i = hash(arr[idx].p, arr[idx].n) & (HSIZE - 1);
    while (tab[i]) i = (i + 1) & (HSIZE - 1);
    tab[i] = idx + 1;
}
static int hget(const uint32_t *tab, const gstr *arr, const char *p, size_t n) {
    uint32_t i = hash(p, n) & (HSIZE - 1);
    for (; tab[i]; i = (i + 1) & (HSIZE - 1)) {
        const gstr *s = &arr[tab[i] - 1];
        if (s->n == n && !memcmp(s->p, p, n)) return tab[i] - 1;
    }
    return -1;
}

static void vocab_init(vocab *v) {
    const gguf_kv *kt = gguf_find(&G, "tokenizer.ggml.tokens"), *km = gguf_find(&G, "tokenizer.ggml.merges"),
                  *ky = gguf_find(&G, "tokenizer.ggml.token_type");
    if (!kt || !km || !ky) die("%s", "the file has no BPE tokenizer (tokens, merges, token types)");
    const gguf_kv *kp = gguf_find(&G, "tokenizer.ggml.pre");
    char pre[48] = "";
    if (kp && kp->type == GT_STR) { const uint8_t *c = kp->val; gstr p = gguf_arr_str(kp, 0, &c); snprintf(pre, sizeof pre, "%.*s", (int)p.n, p.p); }
    if (!strcmp(pre, "qwen2")) v->pre = PRE_QWEN2;
    else if (!strcmp(pre, "qwen35")) v->pre = PRE_QWEN35;
    else if (!strcmp(pre, "llama-bpe") || !strcmp(pre, "llama3")) v->pre = PRE_LLAMA3;
    else if (!strcmp(pre, "smollm")) v->pre = PRE_SMOLLM;
    else if (!strcmp(pre, "gpt-2") || !strcmp(pre, "gpt2")) v->pre = PRE_GPT2;
    else die("pre-tokenizer '%s' is not supported (pillm knows qwen2, qwen35, llama-bpe, smollm, gpt-2)", pre);
    v->n_tok = kt->count; v->n_merge = km->count;
    v->tok = malloc(sizeof(gstr) * v->n_tok); v->merge = malloc(sizeof(gstr) * v->n_merge);
    v->th = calloc(HSIZE, 4); v->mh = calloc(HSIZE, 4);
    v->type = malloc(4 * v->n_tok); memcpy(v->type, ky->val, 4 * v->n_tok);
    v->special = malloc(sizeof(int) * v->n_tok);
    for (int i = 0; i < v->n_tok; i++) if (v->type[i] == 3 || v->type[i] == 4) v->special[v->n_special++] = i;
    const uint8_t *c = kt->val;
    for (int i = 0; i < v->n_tok; i++) { v->tok[i] = gguf_arr_str(kt, i, &c); hput(v->th, v->tok, i); }
    c = km->val;
    for (int i = 0; i < v->n_merge; i++) { v->merge[i] = gguf_arr_str(km, i, &c); hput(v->mh, v->merge, i); }
    // GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100...
    int n = 0;
    memset(v->u2b, -1, sizeof v->u2b);
    for (int b = 0; b < 256; b++) {
        int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
        v->b2u[b] = keep ? b : 256 + n++;
        v->u2b[v->b2u[b]] = b;
    }
}

static int utf8_put(char *o, uint32_t c) { if (c < 0x80) { o[0] = c; return 1; } o[0] = 0xc0 | (c >> 6); o[1] = 0x80 | (c & 63); return 2; }

// Decode a token into raw bytes (control tokens give nothing). Returns the length.
#define TOK_MAX 512
static int tok_bytes(const vocab *v, int id, char *out) {
    if (v->type[id] != 1 && v->type[id] != 4 && v->type[id] != 6) return 0;
    const uint8_t *p = (const uint8_t *)v->tok[id].p, *e = p + v->tok[id].n;
    int n = 0;
    while (p < e && n < TOK_MAX) {
        uint32_t c = *p < 0x80 ? *p : ((*p & 0x1f) << 6) | (p[1] & 0x3f);
        p += *p < 0x80 ? 1 : 2;
        if (c < 1024 && v->u2b[c] >= 0) out[n++] = v->u2b[c];
    }
    return n;
}

// Qwen3.5 includes combining marks in words; older pre-tokenizers do not.
enum { C_L, C_N, C_S, C_P, C_M };
#include "unicode.h"
static int cls_kind(const uint8_t *s, int *len, int marks) {
    uint32_t cp = s[0]; *len = 1;
    int n = cp < 0x80 ? 1 : cp >= 0xc2 && cp < 0xe0 ? 2 : cp >= 0xe0 && cp < 0xf0 ? 3 : cp >= 0xf0 && cp < 0xf5 ? 4 : 0;
    if (!n) return C_P;
    if (n > 1) {
        cp &= (1u << (7 - n)) - 1;
        for (int i = 1; i < n; i++) {
            if (!s[i] || (s[i] & 0xc0) != 0x80) return C_P;
            cp = (cp << 6) | (s[i] & 63);
        }
        if (cp < (n == 2 ? 0x80u : n == 3 ? 0x800u : 0x10000u) || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return C_P;
        *len = n;
    }
    int lo = 0, hi = sizeof unicode_ranges / sizeof *unicode_ranges;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (cp < unicode_ranges[mid].first) hi = mid;
        else if (cp > unicode_ranges[mid].last) lo = mid + 1;
        else {
            int kind = unicode_ranges[mid].kind;
            return kind == C_M ? (marks ? C_L : C_P) : kind;
        }
    }
    return C_P;
}
static int cls(const uint8_t *s, int *len) { return cls_kind(s, len, 0); }
static int is_nl(uint8_t c) { return c == '\n' || c == '\r'; }

// Length in bytes of the next pre-token starting at s (n bytes remain). These are
// hand-written matchers for the regular expressions llama.cpp uses.
//
// qwen2 / llama3:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,k} | ?[^\s\p{L}\p{N}]+[\r\n]* |
//   \s*[\r\n]+ | \s+(?!\S) | \s+            with k = 1 for qwen2 and 3 for llama3
static int pretok_qwen(const uint8_t *s, int n, int max_digits, int marks) {
    int l, c = cls_kind(s, &l, marks);
    if (s[0] == '\'' && n >= 2) {                                           // contractions
        uint8_t a = s[1] | 32, b = n >= 3 ? s[2] | 32 : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }
    {                                                                       // [^\r\n\p{L}\p{N}]?\p{L}+
        int i = 0, l2;
        if (c != C_L && c != C_N && !is_nl(s[0])) i = l;
        int j = i;
        while (j < n && cls_kind(s + j, &l2, marks) == C_L) j += l2;
        if (j > i) return j;
    }
    if (c == C_N) {                                                         // \p{N}{1,k}
        int j = 0, k = 0, l2;
        while (j < n && k < max_digits && cls_kind(s + j, &l2, marks) == C_N) { j += l2; k++; }
        return j;
    }
    {                                                                       //  ?[^\s\p{L}\p{N}]+[\r\n]*
        int i = s[0] == ' ' ? 1 : 0, j = i, l2;
        while (j < n && cls_kind(s + j, &l2, marks) == C_P) j += l2;
        if (j > i) { while (j < n && is_nl(s[j])) j++; return j; }
    }
    int end = 0, last_nl = 0, l2;                                           // whitespace run
    while (end < n && cls_kind(s + end, &l2, marks) == C_S) { end += l2; if (is_nl(s[end - 1])) last_nl = end; }
    if (last_nl) return last_nl;                                            // \s*[\r\n]+
    if (end == n || end == 1) return end;                                   // \s+(?!\S) at end of text, or \s+
    return end - 1;                                                         // \s+(?!\S): leave one space for the next word
}

// gpt-2:   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S)
// smollm:  the same, after every digit has first been split off as its own piece.
// Text no alternative matches (a lone newline before a word, say) is a piece by itself.
static int pretok_gpt2(const uint8_t *s, int n, int split_digits) {
    int l, c = cls(s, &l);
    if (s[0] == '\'' && n >= 2) {
        if (n >= 3 && ((s[1] == 'r' && s[2] == 'e') || (s[1] == 'v' && s[2] == 'e') || (s[1] == 'l' && s[2] == 'l'))) return 3;
        if (s[1] == 's' || s[1] == 't' || s[1] == 'm' || s[1] == 'd') return 2;
    }
    if (split_digits && c == C_N) return l;
    int i = s[0] == ' ' ? 1 : 0, l2;
    if (i < n) {                                                            //  ?X+ for X in letters, digits, other
        int k = cls(s + i, &l2);
        if (k != C_S && !(split_digits && k == C_N)) {
            int j = i;
            while (j < n && cls(s + j, &l2) == k) j += l2;
            return j;
        }
    }
    int end = 0, last = 0, chars = 0;                                       // whitespace run; last = its final character
    while (end < n && cls(s + end, &l2) == C_S) { last = end; end += l2; chars++; }
    if (end == n || chars == 1) return end;
    if (split_digits && cls(s + end, &l2) == C_N) return end;               // the digit was cut off: the run ends its fragment
    return last;                                                            // \s+(?!\S) backs off one character
}

static int pretok(const vocab *v, const uint8_t *s, int n) {
    switch (v->pre) {
        case PRE_QWEN2: return pretok_qwen(s, n, 1, 0);
        case PRE_QWEN35: return pretok_qwen(s, n, 1, 1);
        case PRE_LLAMA3: return pretok_qwen(s, n, 3, 0);
        case PRE_SMOLLM: return pretok_gpt2(s, n, 1);
        default: return pretok_gpt2(s, n, 0);
    }
}

static int bpe(const vocab *v, const uint8_t *s, int n, int *out, int max) {
    char *buf = malloc((size_t)n * 2 + 1), *key = malloc((size_t)n * 4 + 2);
    int *off = malloc(((size_t)n + 1) * sizeof(int)), ns = n, o = 0;
    if (!buf || !key || !off) { free(buf); free(key); free(off); return -1; }
    for (int i = 0; i < n; i++) { off[i] = o; o += utf8_put(buf + o, v->b2u[s[i]]); }
    off[n] = o;
    for (;;) {                                                              // merge the best-ranked adjacent pair
        int best = -1, bi = -1;
        for (int i = 0; i + 1 < ns; i++) {
            int a = off[i + 1] - off[i], b = off[i + 2] - off[i + 1];
            memcpy(key, buf + off[i], a); key[a] = ' '; memcpy(key + a + 1, buf + off[i + 1], b);
            int r = hget(v->mh, v->merge, key, a + 1 + b);
            if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
        }
        if (bi < 0) break;
        memmove(off + bi + 1, off + bi + 2, sizeof(int) * (ns - bi - 1));
        ns--;
    }
    int k = 0;
    if (ns > max) { free(buf); free(key); free(off); return -1; }
    for (int i = 0; i < ns; i++) {
        int id = hget(v->th, v->tok, buf + off[i], off[i + 1] - off[i]);
        if (id >= 0) out[k++] = id;
    }
    free(buf); free(key); free(off);
    return k;
}

// Added tokens such as <think> need not use the <|...|> spelling.
static int special_at(const vocab *v, const uint8_t *s, int n) {
    int id = -1; size_t longest = 0;
    for (int i = 0; i < v->n_special; i++) {
        int t = v->special[i]; gstr word = v->tok[t];
        if (word.n > longest && word.n <= (uint64_t)n && word.p[0] == s[0] && !memcmp(s, word.p, word.n)) {
            id = t; longest = word.n;
        }
    }
    return id;
}

static int tokenize(const vocab *v, const char *text, int *out, int max) {
    const uint8_t *s = (const uint8_t *)text;
    int n = strlen(text), k = 0;
    for (int i = 0; i < n;) {
        if (k == max) return -1;
        int special = special_at(v, s + i, n - i);
        if (special >= 0) { out[k++] = special; i += v->tok[special].n; continue; }
        int len = pretok(v, s + i, n - i);
        if (len < 1 || len > n - i) return -1;
        for (int j = 1; j < len; j++) if (special_at(v, s + i + j, n - i - j) >= 0) { len = j; break; }
        int made = bpe(v, s + i, len, out + k, max - k);
        if (made < 0) return -1;
        k += made;
        i += len;
    }
    return k;
}

// ---------------------------------------------------------------- main

// ---------------------------------------------------------------- sampling
typedef struct { float temp, top_p, repeat; int top_k; uint64_t rng; } sampler;

static float rnd(sampler *s) {                                              // xorshift64*, uniform in [0,1)
    s->rng ^= s->rng >> 12; s->rng ^= s->rng << 25; s->rng ^= s->rng >> 27;
    return (float)((s->rng * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

// Pick the next token from the model's candidates. hist holds the tokens so far
// (for the repetition penalty).
static int sample(sampler *s, model *m, const int *hist, int n_hist) {
    float *lg = m->cand_logit; int *cid = m->cand_id, nc = m->n_cand;
    if (s->temp <= 0) {                                                     // greedy: pure argmax
        int best = 0;
        for (int i = 1; i < nc; i++) if (lg[i] > lg[best]) best = i;
        return cid[best];
    }
    if (s->repeat != 1.0f)
        for (int i = 0; i < nc; i++)
            for (int j = n_hist > 64 ? n_hist - 64 : 0; j < n_hist; j++)
                if (hist[j] == cid[i]) { lg[i] = lg[i] > 0 ? lg[i] / s->repeat : lg[i] * s->repeat; break; }
    int k = s->top_k < 1 ? 1 : s->top_k > nc ? nc : s->top_k;
    int id[NCAND]; float val[NCAND]; int n = 0;
    for (int i = 0; i < nc; i++) {                                          // keep the k largest, sorted
        if (n == k && lg[i] <= val[n - 1]) continue;
        int j = n < k ? n++ : k - 1;
        for (; j > 0 && val[j - 1] < lg[i]; j--) { val[j] = val[j - 1]; id[j] = id[j - 1]; }
        val[j] = lg[i]; id[j] = cid[i];
    }
    float sum = 0, p[NCAND];
    for (int i = 0; i < n; i++) { p[i] = expf((val[i] - val[0]) / s->temp); sum += p[i]; }
    float cut = s->top_p * sum, acc = 0; int keep = 0;
    while (keep < n) { acc += p[keep++]; if (acc >= cut) break; }           // nucleus
    float r = rnd(s) * acc;
    for (int i = 0; i < keep; i++) { r -= p[i]; if (r <= 0) return id[i]; }
    return id[keep - 1];
}

// ---------------------------------------------------------------- driver
// Chat formats, picked from the template text stored in the model file.
typedef struct { const char *name, *marker, *bos, *sys, *user, *end, *msg, *gen; } chatfmt;   // msg takes role, content
static const chatfmt QWEN35_CHATFMT = {
    "qwen35", "<|im_start|>", "", "<|im_start|>system\n%s<|im_end|>\n",
    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_end|>\n",
    "<|im_start|>%s\n%s<|im_end|>\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n"
};
static const chatfmt CHATFMT[] = {
    { "chatml", "<|im_start|>", "", "<|im_start|>system\n%s<|im_end|>\n", "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", "<|im_end|>\n",
      "<|im_start|>%s\n%s<|im_end|>\n", "<|im_start|>assistant\n" },
    { "llama3", "<|start_header_id|>", "<|begin_of_text|>", "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|>",
      "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>",
      "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", "<|start_header_id|>assistant<|end_header_id|>\n\n" },
};

typedef struct {
    model m; vocab v; sampler s;
    const chatfmt *fmt;
    int *hist;                      // tokens in the cache, by slot
    int n, pos;                     // slots in use; position of the next token (never decreases)
    int keep, can_shift;            // leading slots that are never dropped; chat may drop old context
    int stop[8], n_stop;
    int hit_stop;                   // the last generate() ended on an end-of-turn token, not on the limit
    const char *sys, *model_name;
    double t_prompt, t_gen; int n_prompt, n_gen;
} session;

// Make room in a full context by dropping the older half of the conversation (the
// first `keep` slots, the system prompt, stay). Keys were rotated by absolute position
// and attention only sees position differences, so the surviving entries stay valid
// as they are: positions keep counting up and nothing is recomputed.
static int feed(session *S, const int *toks, int n);

static void reset_session(session *S) {
    S->n = S->pos = 0;
    if (S->m.hybrid) qwen35_reset(&S->m);
}

static void shift_context(session *S) {
    model *m = &S->m;
    int drop = (S->n - S->keep) / 2, rest = S->n - S->keep - drop;
    for (int l = 0; l < m->n_layer; l++) {
        f16 *kc = m->kc + ((size_t)l * m->n_ctx + S->keep) * m->kvd, *vc = m->vc + ((size_t)l * m->n_ctx + S->keep) * m->kvd;
        memmove(kc, kc + (size_t)drop * m->kvd, (size_t)rest * m->kvd * sizeof(f16));
        memmove(vc, vc + (size_t)drop * m->kvd, (size_t)rest * m->kvd * sizeof(f16));
    }
    memmove(S->hist + S->keep, S->hist + S->keep + drop, sizeof(int) * rest);
    S->n -= drop;
    if (m->hybrid) {
        int n = S->n;
        reset_session(S);
        feed(S, S->hist, n);
    }
    fprintf(stderr, "\n[context full: dropped the oldest %d tokens]\n", drop);
}

// Feed tokens through the model; candidates are only computed after the last one.
static int feed(session *S, const int *toks, int n) {
    int room = S->m.n_ctx - (S->can_shift ? S->keep : S->n);
    if (n < 1 || n > room) return -1;
    while (S->n + n > S->m.n_ctx) shift_context(S);
    double t0 = now();
    for (int i = 0; i < n; i += BMAX) {
        int B = n - i < BMAX ? n - i : BMAX;
        memmove(S->hist + S->n, toks + i, sizeof(int) * B);
        forward(&S->m, toks + i, B, S->n, S->pos, i + B == n);
        S->n += B; S->pos += B;
    }
    S->t_prompt += now() - t0; S->n_prompt += n;
    return 0;
}

typedef int (*emit_fn)(void *ctx, const char *bytes, int n);                // returns nonzero to stop generating
static int emit_stdout(void *ctx, const char *b, int n) { (void)ctx; fwrite(b, 1, n, stdout); fflush(stdout); return 0; }

// Generate until end-of-turn or the limit, handing each token's bytes to emit.
// Returns the number of tokens produced.
static int generate(session *S, int limit, int show_tokens, emit_fn emit, void *ctx) {
    double t0 = now(); int n = 0;
    S->hit_stop = 0;
    while (n < limit) {
        int t = sample(&S->s, &S->m, S->hist, S->n);
        int stop = 0;
        for (int i = 0; i < S->n_stop; i++) stop |= t == S->stop[i];
        if (stop) { S->hit_stop = 1; break; }
        if (show_tokens) fprintf(stderr, "[%d]", t);
        char piece[TOK_MAX]; int len = tok_bytes(&S->v, t, piece);
        n++;
        if (emit(ctx, piece, len)) break;
        if (S->n == S->m.n_ctx) { if (!S->can_shift || S->n - S->keep < 2) break; shift_context(S); }
        S->hist[S->n] = t;
        forward(&S->m, &t, 1, S->n++, S->pos++, 1);
    }
    S->t_gen += now() - t0; S->n_gen += n;
    return n;
}

#include "server.h"

static void usage(void) {
    fprintf(stderr,
        "usage: pillm -m model.gguf [options]\n"
        "  -p TEXT      ask one question and exit\n"
        "  -i           interactive chat (default when -p is absent)\n"
        "  --serve PORT serve an OpenAI-compatible API and a chat page over HTTP\n"
        "  --host ADDR  address to listen on (default 127.0.0.1; 0.0.0.0 for the whole network)\n"
        "  -s TEXT      system prompt\n"
        "  -n N         max tokens per reply (default 512)\n"
        "  -c N         context size in tokens (default 4096)\n"
        "  -t N         threads while generating (default 2: two cores saturate this SoC's memory)\n"
        "  -tb N        threads while reading the prompt (default 4: batches are CPU-bound)\n"
        "  --temp F     temperature, 0 = greedy (default 0.7)\n"
        "  --top-k N    (default 20, max 256)    --top-p F  (default 0.8)    --repeat F  (default 1.05)\n"
        "  --exact-head score the whole vocabulary without candidate shortlisting\n"
        "  --seed N     random seed (default: from the clock)\n"
        "  --raw        -p is the full prompt text, no chat template\n"
        "  --tokens     print token ids    --prof  print per-stage timing\n");
}

static session S;

int main(int argc, char **argv) {
    const char *path = 0, *prompt = 0, *sys = 0;
    int n_predict = 512, raw = 0, show_tokens = 0, n_ctx = 4096, show_prof = 0, threads = 2, threads_batch = 4, port = 0;
    const char *host = "127.0.0.1";
    S.s = (sampler){ .temp = 0.7f, .top_p = 0.8f, .repeat = 1.05f, .top_k = 20, .rng = 0 };
    S.m.two_pass = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *nx = i + 1 < argc ? argv[i + 1] : 0;
#define OPT(name) (!strcmp(a, name) && nx && ++i)
        if (OPT("-m")) path = nx;
        else if (OPT("-p")) prompt = nx;
        else if (OPT("-s")) sys = nx;
        else if (OPT("-n")) n_predict = atoi(nx);
        else if (OPT("-c")) n_ctx = atoi(nx);
        else if (OPT("-t")) threads = atoi(nx);
        else if (OPT("-tb")) threads_batch = atoi(nx);
        else if (OPT("--serve")) port = atoi(nx);
        else if (OPT("--host")) host = nx;
        else if (OPT("--temp")) S.s.temp = atof(nx);
        else if (OPT("--top-k")) S.s.top_k = atoi(nx);
        else if (OPT("--top-p")) S.s.top_p = atof(nx);
        else if (OPT("--repeat")) S.s.repeat = atof(nx);
        else if (OPT("--seed")) S.s.rng = strtoull(nx, 0, 10);
        else if (!strcmp(a, "-i")) prompt = 0;
        else if (!strcmp(a, "--raw")) raw = 1;
        else if (!strcmp(a, "--exact-head")) S.m.two_pass = 0;
        else if (!strcmp(a, "--tokens")) show_tokens = 1;
        else if (!strcmp(a, "--prof")) show_prof = 1;
        else { usage(); return 1; }
#undef OPT
    }
    if (!path || n_ctx < 64 || threads < 1 || threads_batch < 1) { usage(); return 1; }
    if (!S.s.rng) S.s.rng = (uint64_t)(now() * 1e9) | 1;
    omp_set_num_threads(threads > threads_batch ? threads : threads_batch);
    S.m.nt_gen = threads; S.m.nt_batch = threads_batch;

    double t0 = now();
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(path); return 1; }
    const uint8_t *file = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (file == MAP_FAILED) { perror("mmap"); return 1; }
    if (gguf_parse(&G, file, st.st_size)) die("%s: not a GGUF file pillm can read", path);
    model_init(&S.m, n_ctx);
    vocab_init(&S.v);
    S.hist = malloc(sizeof(int) * n_ctx);
    S.stop[S.n_stop++] = gguf_u(&G, "tokenizer.ggml.eos_token_id", 0);
    static const char *stops[] = { "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>" };
    for (int i = 0; i < 4; i++) { int id = hget(S.v.th, S.v.tok, stops[i], strlen(stops[i])); if (id >= 0) S.stop[S.n_stop++] = id; }
    const gguf_kv *kt = gguf_find(&G, "tokenizer.chat_template");
    if (kt && kt->type == GT_STR) {
        const uint8_t *c = kt->val; gstr t = gguf_arr_str(kt, 0, &c);
        for (unsigned i = 0; i < sizeof CHATFMT / sizeof *CHATFMT && !S.fmt; i++)
            if (memmem(t.p, t.n, CHATFMT[i].marker, strlen(CHATFMT[i].marker))) S.fmt = &CHATFMT[i];
    }
    if (S.m.hybrid) S.fmt = &QWEN35_CHATFMT;
    if (!S.fmt && !raw) die("%s", "this model's chat format is not supported (pillm knows ChatML and Llama 3); use --raw with -p");
    if (!sys) {
        const gguf_kv *kn = gguf_find(&G, "general.name");
        const uint8_t *c = kn ? kn->val : 0; gstr nm = kn && kn->type == GT_STR ? gguf_arr_str(kn, 0, &c) : (gstr){ "", 0 };
        sys = memmem(nm.p, nm.n, "wen", 3) ? "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." : "You are a helpful assistant.";
    }
    double t_load = now() - t0;

    S.sys = sys;
    S.model_name = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
    if (port) {
        if (!S.fmt) die("%s", "--serve needs a model with a supported chat format");
        fprintf(stderr, "loaded %s in %.2fs\n", S.model_name, now() - t0);
        return serve(&S, host, port, n_predict);
    }

    static char text[32768], line[16384];
    static int toks[8192];
    if (prompt) {                                                           // one question
        if (raw) snprintf(text, sizeof text, "%s", prompt);
        else {
            int o = snprintf(text, sizeof text, "%s", S.fmt->bos);
            o += snprintf(text + o, sizeof text - o, S.fmt->sys, sys);
            snprintf(text + o, sizeof text - o, S.fmt->user, prompt);
        }
        int n = tokenize(&S.v, text, toks, 8192);
        if (show_tokens) { for (int i = 0; i < n; i++) fprintf(stderr, "%d ", toks[i]); fprintf(stderr, "\n"); }
        if (feed(&S, toks, n)) { fprintf(stderr, "prompt does not fit the context (%d tokens, -c %d)\n", n, n_ctx); return 1; }
        if (show_prof)
            for (int i = 0; i < P_N; i++) fprintf(stderr, "  prompt  %-16s %6.1f ms/token\n", prof_name[i], prof[i] / n * 1e3);
        memset(prof, 0, sizeof prof);
        generate(&S, n_predict, show_tokens, emit_stdout, 0);
        printf("\n");
    } else {                                                                // chat
        fprintf(stderr, "pillm chat. Empty line or Ctrl-D to quit.\n");
        if (!S.fmt) die("%s", "chat needs a supported chat format; use -p with --raw");
        int o = snprintf(text, sizeof text, "%s", S.fmt->bos);
        snprintf(text + o, sizeof text - o, S.fmt->sys, sys);
        int n = tokenize(&S.v, text, toks, 8192);
        if (feed(&S, toks, n) || n > n_ctx / 2) die("%s", "the system prompt does not fit; use a larger -c");
        S.keep = S.n; S.can_shift = 1;
        memset(prof, 0, sizeof prof);
        for (;;) {
            fprintf(stderr, "\n> "); fflush(stderr);
            if (!fgets(line, sizeof line, stdin)) break;
            line[strcspn(line, "\r\n")] = 0;
            if (!line[0]) break;
            snprintf(text, sizeof text, S.fmt->user, line);
            n = tokenize(&S.v, text, toks, 8192);
            if (feed(&S, toks, n)) { fprintf(stderr, "that message is too long for the context (%d tokens, -c %d)\n", n, n_ctx); continue; }
            generate(&S, n_predict, show_tokens, emit_stdout, 0);
            printf("\n");
            n = tokenize(&S.v, S.fmt->end, toks, 8192);
            feed(&S, toks, n);
        }
    }

    fprintf(stderr, "\nload %.2fs | prompt %d tok, %.2f tok/s | gen %d tok, %.2f tok/s | %d threads\n", t_load,
            S.n_prompt, S.n_prompt / (S.t_prompt + 1e-9), S.n_gen, S.n_gen / (S.t_gen + 1e-9), threads);
    if (show_prof && S.n_gen)
        for (int i = 0; i < P_N; i++) fprintf(stderr, "  gen     %-16s %6.1f ms/token\n", prof_name[i], prof[i] / S.n_gen * 1e3);
    return 0;
}

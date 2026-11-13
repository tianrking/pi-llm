// Text-only dense Qwen3.5: gated full attention and recurrent Gated DeltaNet.
// Included after the shared matrix, normalisation and attention helpers.
#ifndef PILLM_QWEN35_H
#define PILLM_QWEN35_H

static const gguf_tensor *q35_tensor(const char *name, int in, int out) {
    const gguf_tensor *t = gguf_tensor_find(&G, name);
    if (!t) die("missing tensor %s", name);
    if (t->dim[0] != (uint64_t)in || t->dim[1] != (uint64_t)out || t->dim[2] != 1 || t->dim[3] != 1)
        die("tensor %s: wrong shape for qwen35", name);
    return t;
}

static const float *q35_f32(const char *name, int in, int out) {
    const gguf_tensor *t = q35_tensor(name, in, out);
    if (t->type != GGML_F32) die("tensor %s: expected F32", name);
    return (const float *)t->data;
}

static wt q35_weight(const char *name, int in, int out) {
    const gguf_tensor *t = q35_tensor(name, in, out);
    if ((!blk_bytes(t->type) && t->type != GGML_F32 && t->type != GGML_Q6_K) ||
        (t->type == GGML_Q6_K && in % 256)) die("tensor %s: unsupported quantisation", name);
    return (wt){ t->data, (int)t->type };
}

static void *q35_alloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) die("%s", "out of memory for qwen35 state; use a smaller -c");
    return p;
}

static void qwen35_init(model *m, int n_ctx) {
    m->hybrid = 1;
    int total = hp_u(m, "block_count"), mtp = hp_u(m, "nextn_predict_layers");
    m->n_layer = total - mtp;
    m->d = hp_u(m, "embedding_length"); m->ff = hp_u(m, "feed_forward_length");
    m->n_head = hp_u(m, "attention.head_count"); m->n_kv_head = hp_u(m, "attention.head_count_kv");
    m->hd = hp_u(m, "attention.key_length"); m->n_rot = hp_u(m, "rope.dimension_count");
    m->eps = hp_f(m, "attention.layer_norm_rms_epsilon", 1e-6f);
    m->rope_base = hp_f(m, "rope.freq_base", 10000000.0f);
    m->conv_kernel = hp_u(m, "ssm.conv_kernel"); m->ssm_dim = hp_u(m, "ssm.state_size");
    m->ssm_heads = hp_u(m, "ssm.time_step_rank");
    int interval = hp_u(m, "full_attention_interval");
    if (total < 1 || total > MAXL || mtp < 0 || mtp >= total || m->d < QK || m->d > 8192 || m->d % QK ||
        m->ff < m->d || m->ff > 65536 || m->ff % QK || m->n_head < 1 || m->n_head > 64 ||
        m->hd < 8 || m->hd > 256 || m->hd % 8 || m->n_head * m->hd != m->d ||
        m->n_kv_head < 1 || m->n_kv_head > m->n_head || m->n_head % m->n_kv_head ||
        hp_u(m, "attention.value_length") != (uint64_t)m->hd || m->n_rot < 4 || m->n_rot > m->hd || m->n_rot % 4 ||
        m->ssm_dim < 8 || m->ssm_dim > 256 || m->ssm_dim % 8 || m->ssm_heads < 1 || m->ssm_heads > 64 ||
        m->ssm_dim * m->ssm_heads != m->d || hp_u(m, "ssm.group_count") != (uint64_t)m->ssm_heads ||
        hp_u(m, "ssm.inner_size") != (uint64_t)m->d || interval < 1 ||
        m->conv_kernel < 2 || m->conv_kernel > 16 || m->eps <= 0 || !isfinite(m->eps) || m->rope_base <= 0)
        die("%s", "unsupported qwen35 dimensions (need the dense 2B layout)");
    m->kvd = m->hd * m->n_kv_head; m->n_ctx = n_ctx;
    const gguf_tensor *te = gguf_tensor_find(&G, "token_embd.weight");
    if (!te || te->dim[1] < 4 || te->dim[1] > 1000000 || te->dim[1] % 4) die("%s", "invalid qwen35 vocabulary");
    m->n_vocab = te->dim[1];
    m->tok_embd = q35_weight("token_embd.weight", m->d, m->n_vocab);
    if (m->tok_embd.type != GGML_Q4_0 && m->tok_embd.type != GGML_Q8_0 && m->tok_embd.type != GGML_Q6_K)
        die("%s", "qwen35 embeddings need Q4_0, Q8_0 or Q6_K");
    m->out = gguf_tensor_find(&G, "output.weight") ? q35_weight("output.weight", m->d, m->n_vocab) : m->tok_embd;
    m->two_pass = 0;
    m->out_norm = q35_f32("output_norm.weight", m->d, 1);
    for (int i = 0; i < m->n_layer; i++) {
        layer *l = &m->L[i]; char name[96];
#define Q35_NAME(suffix) (snprintf(name, sizeof name, "blk.%d." suffix, i), name)
        l->attn_norm = q35_f32(Q35_NAME("attn_norm.weight"), m->d, 1);
        l->ffn_norm = q35_f32(Q35_NAME("post_attention_norm.weight"), m->d, 1);
        l->gate = q35_weight(Q35_NAME("ffn_gate.weight"), m->d, m->ff);
        l->up = q35_weight(Q35_NAME("ffn_up.weight"), m->d, m->ff);
        l->down = q35_weight(Q35_NAME("ffn_down.weight"), m->ff, m->d);
        l->recurrent = (i + 1) % interval != 0;
        if (l->recurrent) {
            if (gguf_tensor_find(&G, Q35_NAME("attn_qkv_gate.weight")))
                l->qkv = q35_weight(name, m->d, 4 * m->d);
            else {
                l->qkv = q35_weight(Q35_NAME("attn_qkv.weight"), m->d, 3 * m->d);
                l->z = q35_weight(Q35_NAME("attn_gate.weight"), m->d, m->d);
            }
            if (gguf_tensor_find(&G, Q35_NAME("ssm_beta_alpha.weight")))
                l->alpha = q35_weight(name, m->d, 2 * m->ssm_heads);
            else {
                l->alpha = q35_weight(Q35_NAME("ssm_alpha.weight"), m->d, m->ssm_heads);
                l->beta = q35_weight(Q35_NAME("ssm_beta.weight"), m->d, m->ssm_heads);
            }
            l->conv_w = q35_f32(Q35_NAME("ssm_conv1d.weight"), m->conv_kernel, 3 * m->d);
            l->dt = q35_f32(Q35_NAME("ssm_dt.bias"), m->ssm_heads, 1);
            l->a = q35_f32(Q35_NAME("ssm_a"), m->ssm_heads, 1);
            l->ssm_norm = q35_f32(Q35_NAME("ssm_norm.weight"), m->ssm_dim, 1);
            l->wo = q35_weight(Q35_NAME("ssm_out.weight"), m->d, m->d);
            l->conv_state = q35_alloc((size_t)(m->conv_kernel - 1) * 3 * m->d, sizeof(float));
            l->ssm_state = q35_alloc((size_t)m->ssm_heads * m->ssm_dim * m->ssm_dim, sizeof(float));
        } else {
            l->wq = q35_weight(Q35_NAME("attn_q.weight"), m->d, 2 * m->d);
            l->wk = q35_weight(Q35_NAME("attn_k.weight"), m->d, m->kvd);
            l->wv = q35_weight(Q35_NAME("attn_v.weight"), m->d, m->kvd);
            l->wo = q35_weight(Q35_NAME("attn_output.weight"), m->d, m->d);
            l->q_norm = q35_f32(Q35_NAME("attn_q_norm.weight"), m->hd, 1);
            l->k_norm = q35_f32(Q35_NAME("attn_k_norm.weight"), m->hd, 1);
        }
#undef Q35_NAME
    }
    size_t kv = (size_t)m->n_layer * n_ctx * m->kvd;
    m->kc = q35_alloc(kv, sizeof(f16)); m->vc = q35_alloc(kv, sizeof(f16));
#define Q35_A(f, n) m->f = q35_alloc((size_t)(n), sizeof *m->f)
    Q35_A(x, BMAX * m->d); Q35_A(h, BMAX * m->d); Q35_A(q, BMAX * m->d);
    Q35_A(k, BMAX * m->kvd); Q35_A(v, BMAX * m->kvd);
    Q35_A(att, BMAX * m->d); Q35_A(o, BMAX * m->d); Q35_A(g, BMAX * m->ff); Q35_A(u, BMAX * m->ff);
    Q35_A(logits, m->n_vocab); Q35_A(xd, BMAX * m->ff / QK); Q35_A(xs, BMAX * m->ff / QK);
    Q35_A(xq, BMAX * m->ff); Q35_A(negc, BMAX * m->ff / QK * 8);
    Q35_A(p, omp_get_max_threads() * (n_ctx + 4)); Q35_A(rc, BMAX * m->hd / 2); Q35_A(rs, BMAX * m->hd / 2);
    Q35_A(qg, BMAX * 2 * m->d); Q35_A(qkv, BMAX * 4 * m->d); Q35_A(z, BMAX * m->d);
    Q35_A(alpha, BMAX * 2 * m->ssm_heads); Q35_A(beta, BMAX * m->ssm_heads); Q35_A(conv, BMAX * 3 * m->d);
#undef Q35_A
}

static void qwen35_reset(model *m) {
    for (int i = 0; i < m->n_layer; i++) if (m->L[i].recurrent) {
        memset(m->L[i].conv_state, 0, (size_t)(m->conv_kernel - 1) * 3 * m->d * sizeof(float));
        memset(m->L[i].ssm_state, 0, (size_t)m->ssm_heads * m->ssm_dim * m->ssm_dim * sizeof(float));
    }
}

static float q35_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float q35_silu(float x) { return x * q35_sigmoid(x); }
static float q35_softplus(float x) { return fmaxf(x, 0) + log1pf(expf(-fabsf(x))); }

static void q35_l2(float *x, int n, float eps) {
    float s = 1.0f / sqrtf(dot_f32(x, x, n) + eps);
    for (int i = 0; i < n; i++) x[i] *= s;
}

// S is stored as contiguous key vectors, one row for each value channel.
static void q35_delta(float *S, const float *q, const float *k, const float *v, int n, float decay, float beta, float *out) {
    float scale = 1.0f / sqrtf(n);
    for (int j = 0; j < n; j++) {
        float *row = S + (size_t)j * n;
        for (int i = 0; i < n; i += 4) vst1q_f32(row + i, vmulq_n_f32(vld1q_f32(row + i), decay));
        float delta = (v[j] - dot_f32(row, k, n)) * beta;
        for (int i = 0; i < n; i += 4) vst1q_f32(row + i, vfmaq_n_f32(vld1q_f32(row + i), vld1q_f32(k + i), delta));
        out[j] = dot_f32(row, q, n) * scale;
    }
}

static void q35_linear(model *m, const layer *l, int B, int nt) {
    int d = m->d, heads = m->ssm_heads, hd = m->ssm_dim, width = l->z.w ? 3 * d : 4 * d;
    matvec(m, l->qkv, d, width, B, 0, m->qkv, width, nt);
    if (l->z.w) matvec(m, l->z, d, d, B, 0, m->z, d, nt);
    else for (int b = 0; b < B; b++) memcpy(m->z + b * d, m->qkv + b * width + 3 * d, d * sizeof(float));
    if (l->beta.w) {
        matvec(m, l->alpha, d, heads, B, 0, m->alpha, heads, nt);
        matvec(m, l->beta, d, heads, B, 0, m->beta, heads, nt);
    } else {
        matvec(m, l->alpha, d, 2 * heads, B, 0, m->alpha, 2 * heads, nt);
        for (int b = 0; b < B; b++) {
            memcpy(m->beta + b * heads, m->alpha + b * 2 * heads, heads * sizeof(float));
            memmove(m->alpha + b * heads, m->alpha + b * 2 * heads + heads, heads * sizeof(float));
        }
    }
    STAGE(P_ATT_MAT)
    int hist = m->conv_kernel - 1;
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int c = 0; c < 3 * d; c++) {
        float *state = l->conv_state + (size_t)c * hist;
        const float *w = l->conv_w + (size_t)c * m->conv_kernel;
        for (int b = 0; b < B; b++) {
            float x = m->qkv[b * width + c], y = x * w[hist];
            for (int i = 0; i < hist; i++) y += state[i] * w[i];
            memmove(state, state + 1, (hist - 1) * sizeof(float)); state[hist - 1] = x;
            m->conv[b * 3 * d + c] = q35_silu(y);
        }
    }
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int h = 0; h < heads; h++) {
        float *state = l->ssm_state + (size_t)h * hd * hd;
        for (int b = 0; b < B; b++) {
            float *q = m->conv + b * 3 * d + h * hd, *k = q + d, *v = k + d;
            q35_l2(q, hd, m->eps); q35_l2(k, hd, m->eps);
            float decay = expf(l->a[h] * q35_softplus(m->alpha[b * heads + h] + l->dt[h]));
            float *out = m->att + b * d + h * hd;
            q35_delta(state, q, k, v, hd, decay, q35_sigmoid(m->beta[b * heads + h]), out);
            rmsnorm(out, l->ssm_norm, hd, m->eps, out);
            for (int i = 0; i < hd; i++) out[i] *= q35_silu(m->z[b * d + h * hd + i]);
        }
    }
    STAGE(P_ATT)
}

static void q35_full(model *m, const layer *l, int B, int li, int slot, int nt) {
    int d = m->d, hd = m->hd, kvd = m->kvd;
    matvec(m, l->wq, d, 2 * d, B, 0, m->qg, 2 * d, nt);
    matvec(m, l->wk, d, kvd, B, 0, m->k, kvd, nt);
    matvec(m, l->wv, d, kvd, B, 0, m->v, kvd, nt);
    STAGE(P_ATT_MAT)
    f16 *kc = m->kc + (size_t)li * m->n_ctx * kvd, *vc = m->vc + (size_t)li * m->n_ctx * kvd;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < m->n_head; h++) {
            float *q = m->q + b * d + h * hd;
            rmsnorm(m->qg + b * 2 * d + h * 2 * hd, l->q_norm, hd, m->eps, q);
            rope(q, 1, m->n_rot, m->rc + b * hd / 2, m->rs + b * hd / 2, 1);
        }
        for (int h = 0; h < m->n_kv_head; h++) {
            float *k = m->k + b * kvd + h * hd;
            rmsnorm(k, l->k_norm, hd, m->eps, k);
            rope(k, 1, m->n_rot, m->rc + b * hd / 2, m->rs + b * hd / 2, 1);
        }
        for (int i = 0; i < kvd; i++) {
            kc[(size_t)(slot + b) * kvd + i] = m->k[b * kvd + i];
            vc[(size_t)(slot + b) * kvd + i] = m->v[b * kvd + i];
        }
    }
    int group = m->n_head / m->n_kv_head;
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int i = 0; i < B * m->n_head; i++) {
        int b = i / m->n_head, h = i % m->n_head;
        float *out = m->att + b * d + h * hd;
        attn_head(m, kc, vc, m->q + b * d + h * hd, slot + b, h / group,
                  m->p + (size_t)omp_get_thread_num() * (m->n_ctx + 4), out);
        const float *gate = m->qg + b * 2 * d + h * 2 * hd + hd;
        for (int j = 0; j < hd; j++) out[j] *= q35_sigmoid(gate[j]);
    }
    STAGE(P_ATT)
}

static void qwen35_attention(model *m, const layer *l, int B, int li, int slot, int pos, int nt) {
    (void)pos;
    if (l->recurrent) q35_linear(m, l, B, nt);
    else q35_full(m, l, B, li, slot, nt);
    quant(m, m->att, m->d, B);
    matvec(m, l->wo, m->d, m->d, B, 0, m->o, m->d, nt);
    for (int i = 0; i < B * m->d; i++) m->x[i] += m->o[i];
    STAGE(P_ATT_MAT)
}
#endif

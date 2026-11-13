// Independent scalar checks for the packed Q6_K and recurrent state kernels.
#define main pillm_cli_main
#include "../src/pillm.c"
#undef main
#include <assert.h>

static void close_enough(double got, double want, double tolerance) {
    if (!isfinite(got) || fabs(got - want) > tolerance * (1 + fabs(want))) {
        fprintf(stderr, "got %.9g, expected %.9g\n", got, want);
        exit(1);
    }
}

static void test_q6(void) {
    enum { N = 1024 };
    uint8_t w[N / 256 * 210]; float x[N], expanded[N], xd[N / 256]; int8_t xq[N];
    memset(w, 0, sizeof w);
    for (int b = 0; b < N / 256; b++) {
        uint8_t *p = w + b * 210;
        f16 d = 0.03125f * (b + 1); memcpy(p + 208, &d, 2);
        for (int j = 0; j < 16; j++) p[192 + j] = (uint8_t)(int8_t)((j * 31 + b * 7) % 255 - 127);
        for (int i = 0; i < 256; i++) {
            int q = (i * 17 + b * 13) % 64 - 32, u = q + 32;
            int half = i / 128, k = i % 128, stream = k / 32, col = k % 32;
            p[half * 64 + (stream & 1) * 32 + col] |= (u & 15) << (stream >= 2 ? 4 : 0);
            p[128 + half * 32 + col] |= (u >> 4) << (2 * stream);
            expanded[b * 256 + i] = (float)d * (int8_t)p[192 + i / 16] * q;
            x[b * 256 + i] = sinf((b * 256 + i) * 0.73f) * 0.02f;
        }
    }
    double ref = 0;
    for (int i = 0; i < N; i++) ref += (double)expanded[i] * x[i];
    close_enough(dot_q6_f32(w, x, N), ref, 2e-5);
    quantize_q8k(x, N, xq, xd);
    ref = 0;
    for (int b = 0; b < N / 256; b++) {
        float peak = 0;
        for (int i = 0; i < 256; i++) if (fabsf(x[b * 256 + i]) > fabsf(peak)) peak = x[b * 256 + i];
        float inv = -127.0f / peak;
        close_enough(xd[b], 1.0f / inv, 1e-7);
        for (int i = 0; i < 256; i++) {
            assert(xq[b * 256 + i] == (int)nearbyintf(x[b * 256 + i] * inv));
            ref += (double)expanded[b * 256 + i] * xd[b] * xq[b * 256 + i];
        }
    }
    close_enough(dot_q6_q8k(w, xq, xd, N), ref, 2e-5);
    memset(x, 0, sizeof x); quantize_q8k(x, N, xq, xd);
    for (int i = 0; i < N; i++) assert(xq[i] == 0);
    assert(dot_q6_q8k(w, xq, xd, N) == 0);
    puts("ok    Q6_K float and Q8_K dots match independently encoded weights");
}

static void test_delta(void) {
    enum { D = 8 };
    float state[D * D] = {0}, q[D], k[D], v[D], out[D]; double ref[D * D] = {0};
    for (int t = 0; t < 5; t++) {
        for (int i = 0; i < D; i++) {
            q[i] = sinf(i + t * .3f); k[i] = cosf(i * .7f + t); v[i] = sinf(i * 1.3f - t);
        }
        q35_l2(q, D, 1e-6f); q35_l2(k, D, 1e-6f);
        float decay = .71f + t * .03f, beta = .13f + t * .11f;
        q35_delta(state, q, k, v, D, decay, beta, out);
        for (int j = 0; j < D; j++) {
            double pred = 0;
            for (int i = 0; i < D; i++) { ref[j * D + i] *= decay; pred += ref[j * D + i] * k[i]; }
            double delta = (v[j] - pred) * beta, y = 0;
            for (int i = 0; i < D; i++) { ref[j * D + i] += delta * k[i]; y += ref[j * D + i] * q[i]; }
            close_enough(out[j], y / sqrt(D), 2e-6);
        }
        for (int i = 0; i < D * D; i++) close_enough(state[i], ref[i], 2e-6);
    }
    puts("ok    recurrent delta state and output match a double-precision scalar recurrence");
}

static void test_pretok(void) {
    int len;
    assert(cls_kind((const uint8_t *)"😀", &len, 1) == C_P && len == 4);
    assert(cls_kind((const uint8_t *)"３", &len, 1) == C_N && len == 3);
    assert(cls_kind((const uint8_t *)"\xcc\x81", &len, 1) == C_L && len == 2);
    assert(cls_kind((const uint8_t *)"\xcc\x81", &len, 0) == C_P);
    assert(pretok_qwen((const uint8_t *)"cafe\xcc\x81!", 7, 1, 1) == 6);
    assert(pretok_qwen((const uint8_t *)"123", 3, 1, 1) == 1);
    close_enough(q35_softplus(100), 100, 1e-7);
    close_enough(q35_softplus(-100), 0, 1e-7);
    puts("ok    Qwen3.5 marks, numbers, emoji and stable gates");
}

static size_t state_size(const model *m) {
    size_t n = 0;
    for (int l = 0; l < m->n_layer; l++) if (m->L[l].recurrent)
        n += (size_t)m->ssm_heads * m->ssm_dim * m->ssm_dim + (size_t)(m->conv_kernel - 1) * 3 * m->d;
    return n;
}

static void capture_state(const model *m, float *out) {
    size_t matrix = (size_t)m->ssm_heads * m->ssm_dim * m->ssm_dim;
    size_t conv = (size_t)(m->conv_kernel - 1) * 3 * m->d;
    for (int l = 0; l < m->n_layer; l++) if (m->L[l].recurrent) {
        memcpy(out, m->L[l].ssm_state, matrix * sizeof(float)); out += matrix;
        memcpy(out, m->L[l].conv_state, conv * sizeof(float)); out += conv;
    }
}

static void compare_state(const model *m, const float *expected) {
    size_t n = state_size(m); float *actual = malloc(n * sizeof(float)); assert(actual);
    capture_state(m, actual);
    double diff = 0, norm = 0;
    for (size_t i = 0; i < n; i++) {
        assert(isfinite(actual[i]));
        diff += (double)(actual[i] - expected[i]) * (actual[i] - expected[i]);
        norm += (double)expected[i] * expected[i];
    }
    double error = sqrt(diff / (norm + 1e-30));
    fprintf(stderr, "state relative L2 error %.7g\n", error);
    if (error >= 1e-5) {
        size_t size = (size_t)m->ssm_heads * m->ssm_dim * m->ssm_dim + (size_t)(m->conv_kernel - 1) * 3 * m->d, offset = 0;
        for (int l = 0; l < m->n_layer; l++) if (m->L[l].recurrent) {
            double e = 0, v = 0;
            for (size_t j = 0; j < size; j++) { double d = actual[offset + j] - expected[offset + j]; e += d*d; v += (double)expected[offset+j]*expected[offset+j]; }
            fprintf(stderr, "layer %d relative error %.7g\n", l, sqrt(e/(v+1e-30))); offset += size;
        }
    }
    assert(error < 1e-5);
    free(actual);
}

static void test_model_state(const char *path) {
    int fd = open(path, O_RDONLY); struct stat st;
    assert(fd >= 0 && !fstat(fd, &st));
    const uint8_t *file = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(file != MAP_FAILED && !gguf_parse(&G, file, st.st_size));
    omp_set_num_threads(4); S.m.nt_gen = 2; S.m.nt_batch = 4;
    model_init(&S.m, 64); assert(S.m.hybrid);
    S.hist = calloc(64, sizeof(int)); assert(S.hist);
    const int tokens[] = {760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198};
    size_t count = state_size(&S.m); float *expected = malloc(count * sizeof(float)); assert(expected);
    assert(feed(&S, tokens, 16) == 0); capture_state(&S.m, expected);
    reset_session(&S);
    for (int i = 0; i < 16; i++) assert(feed(&S, tokens + i, 1) == 0);
    compare_state(&S.m, expected);
    puts("ok    batched and single-token recurrent/conv states agree");
    S.keep = 2; S.can_shift = 1; shift_context(&S);
    int retained[64], n = S.n;
    memcpy(retained, S.hist, n * sizeof(int)); capture_state(&S.m, expected);
    reset_session(&S);
    for (int i = 0; i < n; i++) assert(feed(&S, retained + i, 1) == 0);
    compare_state(&S.m, expected);
    puts("ok    context truncation matches a fresh replay of the retained tokens");
    free(expected);
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--state-check")) { test_model_state(argv[2]); return 0; }
    if (argc == 4 && !strcmp(argv[1], "--tokenize")) {
        int fd = open(argv[2], O_RDONLY); struct stat st;
        if (fd < 0 || fstat(fd, &st)) return 2;
        const uint8_t *file = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (file == MAP_FAILED || gguf_parse(&G, file, st.st_size)) return 2;
        vocab v = {0}; vocab_init(&v); int ids[8192];
        int n = tokenize(&v, argv[3], ids, 8192);
        if (n < 0) return 2;
        for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", ids[i]);
        puts(""); return 0;
    }
    if (argc != 1) return 2;
    test_q6(); test_delta(); test_pretok();
    return 0;
}

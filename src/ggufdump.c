// ggufdump — print the metadata and tensor table of a GGUF file.
#include "gguf.h"
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

static gguf g;

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: ggufdump model.gguf\n"); return 1; }
    int fd = open(argv[1], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(argv[1]); return 1; }
    const uint8_t *f = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (f == MAP_FAILED) { perror("mmap"); return 1; }
    int r = gguf_parse(&g, f, st.st_size);
    if (r) { fprintf(stderr, "parse error %d\n", r); return 1; }

    printf("GGUF v%u, %llu kv, %llu tensors\n", g.version, (unsigned long long)g.n_kv, (unsigned long long)g.n_tensors);
    for (uint64_t i = 0; i < g.n_kv; i++) {
        const gguf_kv *kv = &g.kv[i];
        printf("  %-44.*s ", (int)kv->key.n, kv->key.p);
        if (kv->type == GT_STR) {
            const uint8_t *c = kv->val; gstr s = gguf_arr_str(kv, 0, &c);
            printf("\"%.*s%s\"\n", (int)(s.n > 70 ? 70 : s.n), s.p, s.n > 70 ? "..." : "");
        } else if (kv->type == GT_ARR) {
            printf("[array type=%u count=%llu]\n", kv->elem_type, (unsigned long long)kv->count);
        } else if (kv->type == GT_F32) {
            printf("%g\n", gguf_f(&g, "", 0) + *(const float *)kv->val);
        } else {
            char key[128]; snprintf(key, sizeof key, "%.*s", (int)kv->key.n, kv->key.p);
            printf("%lld\n", (long long)gguf_u(&g, key, 0));
        }
    }
    uint64_t bytes_by_type[40] = {0}, count_by_type[40] = {0};
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const gguf_tensor *t = &g.t[i];
        uint64_t n = t->dim[0] * t->dim[1] * t->dim[2] * t->dim[3];
        uint64_t b = t->type == GGML_F32 ? n * 4 : t->type == GGML_F16 ? n * 2 :
                     t->type == GGML_Q4_0 ? n / 32 * 18 : t->type == GGML_Q4_1 ? n / 32 * 20 :
                     t->type == GGML_Q8_0 ? n / 32 * 34 : t->type == GGML_Q6_K ? n / 256 * 210 : 0;
        if (t->type < 40) { bytes_by_type[t->type] += b; count_by_type[t->type]++; }
        if (i < 14 || i + 3 > g.n_tensors)
            printf("  T %-34.*s type=%-2u [%llu x %llu] off=%llu\n", (int)t->name.n, t->name.p, t->type,
                   (unsigned long long)t->dim[0], (unsigned long long)t->dim[1],
                   (unsigned long long)(t->data - f));
    }
    for (int i = 0; i < 40; i++) if (count_by_type[i])
        printf("  type %d: %llu tensors, %.1f MB\n", i, (unsigned long long)count_by_type[i], bytes_by_type[i] / 1e6);
    return 0;
}

// gguf.h — minimal GGUF v3 reader. Works on a memory image of the file (mmap on
// Linux, a RAM buffer on bare metal). No libc dependency beyond memcmp/strlen.
#ifndef GGUF_H
#define GGUF_H
#include <stdint.h>
#include <stddef.h>

enum { GT_U8, GT_I8, GT_U16, GT_I16, GT_U32, GT_I32, GT_F32, GT_BOOL, GT_STR, GT_ARR, GT_U64, GT_I64, GT_F64 };
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q8_0 = 8, GGML_Q6_K = 14 };

typedef struct { const char *p; uint64_t n; } gstr;          // not NUL-terminated

typedef struct {
    gstr key;
    uint32_t type;        // GT_*
    uint32_t elem_type;   // for GT_ARR
    uint64_t count;       // for GT_ARR
    const uint8_t *val;   // points at the scalar / first array element
} gguf_kv;

typedef struct {
    gstr name;
    uint32_t n_dims;
    uint64_t dim[4];
    uint32_t type;        // GGML_*
    const uint8_t *data;
} gguf_tensor;

#define GGUF_MAX_KV 64
#define GGUF_MAX_TENSORS 512

typedef struct {
    uint32_t version;
    uint64_t n_kv, n_tensors;
    gguf_kv kv[GGUF_MAX_KV];
    gguf_tensor t[GGUF_MAX_TENSORS];
} gguf;

int gguf_parse(gguf *g, const uint8_t *file, uint64_t size);        // 0 on success
const gguf_kv *gguf_find(const gguf *g, const char *key);
const gguf_tensor *gguf_tensor_find(const gguf *g, const char *name);
uint64_t gguf_u(const gguf *g, const char *key, uint64_t dflt);     // any integer kv
float gguf_f(const gguf *g, const char *key, float dflt);
gstr gguf_arr_str(const gguf_kv *kv, uint64_t i, const uint8_t **cursor); // sequential walk

#endif

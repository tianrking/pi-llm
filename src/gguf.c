#include "gguf.h"
#include <string.h>

static uint32_t rd32(const uint8_t **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static uint64_t rd64(const uint8_t **p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static gstr rdstr(const uint8_t **p) { gstr s; s.n = rd64(p); s.p = (const char *)*p; *p += s.n; return s; }

static const uint8_t scalar_size[] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };

// Advance past one value of the given type.
static void skip(const uint8_t **p, uint32_t type) {
    if (type == GT_STR) { rdstr(p); return; }
    if (type == GT_ARR) {
        uint32_t et = rd32(p); uint64_t n = rd64(p);
        if (et == GT_STR) { while (n--) rdstr(p); }
        else *p += n * scalar_size[et];
        return;
    }
    *p += scalar_size[type];
}

int gguf_parse(gguf *g, const uint8_t *file, uint64_t size) {
    const uint8_t *p = file;
    if (size < 24 || memcmp(p, "GGUF", 4)) return -1;
    p += 4;
    g->version = rd32(&p);
    g->n_tensors = rd64(&p);
    g->n_kv = rd64(&p);
    if (g->version < 2 || g->n_kv > GGUF_MAX_KV || g->n_tensors > GGUF_MAX_TENSORS) return -2;

    for (uint64_t i = 0; i < g->n_kv; i++) {
        gguf_kv *kv = &g->kv[i];
        kv->key = rdstr(&p);
        kv->type = rd32(&p);
        kv->elem_type = 0; kv->count = 0;
        if (kv->type == GT_ARR) {
            const uint8_t *q = p;
            kv->elem_type = rd32(&q);
            kv->count = rd64(&q);
            kv->val = q;
        } else if (kv->type == GT_STR) {
            kv->val = p;            // points at the u64 length
        } else kv->val = p;
        skip(&p, kv->type);
    }

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->t[i];
        t->name = rdstr(&p);
        t->n_dims = rd32(&p);
        if (t->n_dims > 4) return -3;
        for (uint32_t d = 0; d < 4; d++) t->dim[d] = d < t->n_dims ? rd64(&p) : 1;
        t->type = rd32(&p);
        t->data = (const uint8_t *)(uintptr_t)rd64(&p);   // offset for now
    }

    uint64_t align = gguf_u(g, "general.alignment", 32);
    uint64_t base = ((uint64_t)(p - file) + align - 1) / align * align;
    for (uint64_t i = 0; i < g->n_tensors; i++)
        g->t[i].data = file + base + (uint64_t)(uintptr_t)g->t[i].data;
    return 0;
}

static int keyeq(gstr s, const char *k) { size_t n = strlen(k); return s.n == n && !memcmp(s.p, k, n); }

const gguf_kv *gguf_find(const gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++) if (keyeq(g->kv[i].key, key)) return &g->kv[i];
    return 0;
}

const gguf_tensor *gguf_tensor_find(const gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) if (keyeq(g->t[i].name, name)) return &g->t[i];
    return 0;
}

uint64_t gguf_u(const gguf *g, const char *key, uint64_t dflt) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv) return dflt;
    switch (kv->type) {
        case GT_U8: case GT_BOOL: return *kv->val;
        case GT_I8: return (uint64_t)(int64_t)*(const int8_t *)kv->val;
        case GT_U16: { uint16_t v; memcpy(&v, kv->val, 2); return v; }
        case GT_I16: { int16_t v; memcpy(&v, kv->val, 2); return (uint64_t)(int64_t)v; }
        case GT_U32: { uint32_t v; memcpy(&v, kv->val, 4); return v; }
        case GT_I32: { int32_t v; memcpy(&v, kv->val, 4); return (uint64_t)(int64_t)v; }
        case GT_U64: case GT_I64: { uint64_t v; memcpy(&v, kv->val, 8); return v; }
    }
    return dflt;
}

float gguf_f(const gguf *g, const char *key, float dflt) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv || kv->type != GT_F32) return dflt;
    float v; memcpy(&v, kv->val, 4); return v;
}

// Walk a string array in order: pass *cursor = kv->val before the first call.
gstr gguf_arr_str(const gguf_kv *kv, uint64_t i, const uint8_t **cursor) {
    (void)kv; (void)i;
    return rdstr(cursor);
}

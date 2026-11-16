// server.h — a small HTTP server: OpenAI-style chat completions and a built-in chat page.
// Included by pillm.c. One request at a time: the model can only do one thing at once.
//
//   GET  /                      chat page
//   GET  /health                {"status":"ok"}
//   GET  /v1/models             the loaded model
//   POST /v1/chat/completions   messages, max_tokens, temperature, top_p, top_k, seed, stream
//
// Requests carry the whole conversation each time. The part that matches what is
// already in the K/V cache is reused for qwen2/llama. Hybrid models restore one
// matching message-boundary checkpoint and replay the uncached suffix.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

// ---------------------------------------------------------------- JSON, just enough to read a request
static const char *js_ws(const char *p) { while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++; return p; }

// Skip one value; returns the position after it, or null on malformed input.
static const char *js_skip(const char *p) {
    p = js_ws(p);
    if (*p == '"') {
        for (p++; *p && *p != '"'; p++) if (*p == '\\' && p[1]) p++;
        return *p ? p + 1 : 0;
    }
    if (*p == '{' || *p == '[') {
        char open = *p, close = open == '{' ? '}' : ']';
        p = js_ws(p + 1);
        if (*p == close) return p + 1;
        for (;;) {
            if (open == '{') {
                if (!(p = js_skip(p))) return 0;
                p = js_ws(p);
                if (*p++ != ':') return 0;
            }
            if (!(p = js_skip(p))) return 0;
            p = js_ws(p);
            if (*p == ',') { p++; continue; }
            return *p == close ? p + 1 : 0;
        }
    }
    const char *s = p;
    while (*p && !strchr(",]} \n\r\t", *p)) p++;
    return p > s ? p : 0;
}

// The value stored under key in the object at obj, or null.
static const char *js_get(const char *obj, const char *key) {
    const char *p = js_ws(obj);
    size_t kl = strlen(key);
    if (*p != '{') return 0;
    p = js_ws(p + 1);
    while (*p == '"') {
        const char *k = p + 1, *e = js_skip(p);
        if (!e) return 0;
        int match = (size_t)(e - 1 - k) == kl && !memcmp(k, key, kl);
        p = js_ws(e);
        if (*p != ':') return 0;
        p = js_ws(p + 1);
        if (match) return p;
        if (!(p = js_skip(p))) return 0;
        p = js_ws(p);
        if (*p == ',') p = js_ws(p + 1);
    }
    return 0;
}

static int js_hex4(const char *p) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        int d = c >= '0' && c <= '9' ? c - '0' : (c | 32) >= 'a' && (c | 32) <= 'f' ? (c | 32) - 'a' + 10 : -1;
        if (d < 0) return -1;
        v = v * 16 + d;
    }
    return v;
}

// Decode the JSON string at p into UTF-8. Returns the length, or -1 if p is not a string.
// Output beyond max - 1 bytes is dropped.
static int js_str(const char *p, char *out, int max) {
    if (!p || *p != '"') return -1;
    int n = 0;
    for (p++; *p && *p != '"';) {
        unsigned c = (unsigned char)*p++;
        if (c == '\\') {
            c = (unsigned char)*p; if (!c) break;
            p++;
            if (c == 'n') c = '\n'; else if (c == 't') c = '\t'; else if (c == 'r') c = '\r';
            else if (c == 'b') c = '\b'; else if (c == 'f') c = '\f';
            else if (c == 'u') {
                int u = js_hex4(p);
                if (u < 0) break;
                p += 4;
                if (u >= 0xD800 && u < 0xDC00 && p[0] == '\\' && p[1] == 'u') {     // surrogate pair
                    int lo = js_hex4(p + 2);
                    if (lo >= 0xDC00 && lo < 0xE000) { u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00); p += 6; }
                }
                char b[4]; int k = 0;
                if (u < 0x80) b[k++] = u;
                else if (u < 0x800) { b[k++] = 0xC0 | (u >> 6); b[k++] = 0x80 | (u & 63); }
                else if (u < 0x10000) { b[k++] = 0xE0 | (u >> 12); b[k++] = 0x80 | ((u >> 6) & 63); b[k++] = 0x80 | (u & 63); }
                else { b[k++] = 0xF0 | (u >> 18); b[k++] = 0x80 | ((u >> 12) & 63); b[k++] = 0x80 | ((u >> 6) & 63); b[k++] = 0x80 | (u & 63); }
                for (int i = 0; i < k; i++) if (n < max - 1) out[n++] = b[i];
                continue;
            }
        }
        if (n < max - 1) out[n++] = c;
    }
    out[n] = 0;
    return n;
}

static double js_num(const char *p, double dflt) {
    return p && (*p == '-' || (*p >= '0' && *p <= '9')) ? strtod(p, 0) : dflt;
}

// Append s[0..n) to out as the inside of a JSON string. out needs 6n + 1 bytes.
static int js_escape(const char *s, int n, char *out) {
    int o = 0;
    for (int i = 0; i < n; i++) {
        unsigned char c = s[i];
        if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = c; }
        else if (c == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
        else if (c == '\r') { out[o++] = '\\'; out[o++] = 'r'; }
        else if (c == '\t') { out[o++] = '\\'; out[o++] = 't'; }
        else if (c < 0x20) o += sprintf(out + o, "\\u%04x", c);
        else out[o++] = c;
    }
    out[o] = 0;
    return o;
}

// ---------------------------------------------------------------- HTTP plumbing
static int send_all(int fd, const char *b, size_t n) {
    while (n) {
        ssize_t k = send(fd, b, n, MSG_NOSIGNAL);
        if (k <= 0) return -1;
        b += k; n -= k;
    }
    return 0;
}

static void http_send(int fd, int status, const char *type, const char *body, size_t len) {
    char h[256];
    const char *msg = status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found" : "Error";
    int n = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", status, msg, type, len);
    if (!send_all(fd, h, n)) send_all(fd, body, len);
}

static void http_error(int fd, int status, const char *message) {
    char b[1024], esc[6 * 120 + 1];
    js_escape(message, strlen(message) > 120 ? 120 : strlen(message), esc);
    int n = snprintf(b, sizeof b, "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}", esc);
    http_send(fd, status, "application/json", b, n);
}

// How many leading bytes of b[0..n) are whole UTF-8 characters. Tokens can end in the
// middle of a character, and JSON must not be sent with half of one.
static int utf8_whole(const char *b, int n) {
    int i = n, cont = 0;
    while (i > 0 && cont < 4 && ((unsigned char)b[i - 1] & 0xC0) == 0x80) { i--; cont++; }
    if (i == 0) return n;
    unsigned char c = b[i - 1];
    int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    return cont + 1 < need ? i - 1 : n;
}

// ---------------------------------------------------------------- chat completions
typedef struct {
    session *S; int fd, stream, failed;
    char id[32]; long created;
    char *acc; size_t acc_n, acc_cap;           // the whole reply, when not streaming
    char pend[TOK_MAX + 8]; int pend_n;         // bytes waiting for the rest of their character
} reply;

static void reply_text(reply *r, const char *b, int n) {
    if (!n) return;
    if (!r->stream) {
        if (r->acc_n + n + 1 > r->acc_cap) { r->acc_cap = (r->acc_cap + n) * 2 + 256; r->acc = realloc(r->acc, r->acc_cap); }
        memcpy(r->acc + r->acc_n, b, n); r->acc_n += n;
        return;
    }
    char esc[6 * (TOK_MAX + 8) + 1], out[sizeof esc + 320];
    js_escape(b, n, esc);
    int k = snprintf(out, sizeof out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
                     "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                     r->id, r->created, r->S->model_name, esc);
    if (send_all(r->fd, out, k)) r->failed = 1;
}

static int emit_http(void *ctx, const char *bytes, int n) {
    reply *r = ctx;
    memcpy(r->pend + r->pend_n, bytes, n); r->pend_n += n;
    int k = utf8_whole(r->pend, r->pend_n);
    reply_text(r, r->pend, k);
    memmove(r->pend, r->pend + k, r->pend_n - k); r->pend_n -= k;
    return r->failed;                                                       // the client went away: stop generating
}

// The text of a message's content: a string, or an array of {"type":"text","text":...} parts.
static int message_text(const char *v, char *out, int max) {
    if (!v) return -1;
    if (*v == '"') return js_str(v, out, max);
    if (*v != '[') return -1;
    int n = 0;
    for (const char *p = js_ws(v + 1); *p == '{';) {
        char type[16];
        if (js_str(js_get(p, "type"), type, sizeof type) < 0 || strcmp(type, "text")) return -1;
        int k = js_str(js_get(p, "text"), out + n, max - n);
        if (k < 0) return -1;
        n += k;
        if (!(p = js_skip(p))) return -1;
        p = js_ws(p);
        if (*p == ',') p = js_ws(p + 1);
    }
    out[n] = 0;
    return n;
}

// Keep one checkpoint before the assistant generation prefix. Later turns omit
// the previous empty <think> prefix, so a generation-end checkpoint cannot match.
static size_t prefix_state_size(const model *m) {
    size_t n = 0;
    for (int i = 0; i < m->n_layer; i++) if (m->L[i].recurrent)
        n += (size_t)m->ssm_heads * m->ssm_dim * m->ssm_dim + (size_t)(m->conv_kernel - 1) * 3 * m->d;
    return n;
}

static int prefix_init(session *S) {
    if (S->prefix_state && S->prefix_tokens) return 1;
    size_t count = prefix_state_size(&S->m);
    if (!count) return 0;
    float *state = malloc(count * sizeof(float));
    int *tokens = malloc((size_t)S->m.n_ctx * sizeof(int));
    if (!state || !tokens) { free(state); free(tokens); return 0; }
    S->prefix_state = state; S->prefix_tokens = tokens;
    return 1;
}

static void prefix_store(session *S) {
    const model *m = &S->m;
    size_t matrix = (size_t)m->ssm_heads * m->ssm_dim * m->ssm_dim;
    size_t conv = (size_t)(m->conv_kernel - 1) * 3 * m->d;
    float *p = S->prefix_state;
    for (int i = 0; i < m->n_layer; i++) if (m->L[i].recurrent) {
        memcpy(p, m->L[i].ssm_state, matrix * sizeof(float)); p += matrix;
        memcpy(p, m->L[i].conv_state, conv * sizeof(float)); p += conv;
    }
    memcpy(S->prefix_tokens, S->hist, S->n * sizeof(int));
    S->prefix_n = S->n;
}

static int prefix_restore(session *S, const int *tokens, int n) {
    int same = S->prefix_n;
    if (same < 1 || same > n || memcmp(tokens, S->prefix_tokens, same * sizeof(int))) return 0;
    model *m = &S->m;
    size_t matrix = (size_t)m->ssm_heads * m->ssm_dim * m->ssm_dim;
    size_t conv = (size_t)(m->conv_kernel - 1) * 3 * m->d;
    const float *p = S->prefix_state;
    for (int i = 0; i < m->n_layer; i++) if (m->L[i].recurrent) {
        memcpy(m->L[i].ssm_state, p, matrix * sizeof(float)); p += matrix;
        memcpy(m->L[i].conv_state, p, conv * sizeof(float)); p += conv;
    }
    // No request has overwritten this K/V prefix since the checkpoint was saved.
    memcpy(S->hist, S->prefix_tokens, same * sizeof(int));
    S->n = S->pos = same;
    return same;
}

static int prepare_hybrid_prompt(session *S, const int *tokens, int n) {
    int gen[128], ng = tokenize(&S->v, S->fmt->gen, gen, 128);
    if (S->no_prefix_cache || ng < 1 || ng >= n || memcmp(tokens + n - ng, gen, ng * sizeof(int)) || !prefix_init(S)) {
        reset_session(S);
        return feed(S, tokens, n) ? -1 : 0;
    }
    int messages = n - ng, same = prefix_restore(S, tokens, messages);
    if (!same) reset_session(S);
    if (messages > S->n && feed_tokens(S, tokens + S->n, messages - S->n, 0)) {
        reset_session(S); return -1;
    }
    prefix_store(S);
    if (feed(S, tokens + messages, ng)) { reset_session(S); return -1; }
    return same;
}

static void handle_chat(session *S, int fd, const char *body, int default_max) {
    const chatfmt *f = S->fmt;
    const char *msgs = js_get(body, "messages");
    if (!msgs || *msgs != '[') { http_error(fd, 400, "messages must be an array"); return; }

    size_t cap = 2 * strlen(body) + strlen(S->sys) + 4096, o;
    char *text = malloc(cap), *content = malloc(cap), role[32];
    int *toks = 0, n_msg = 0;
    reply r = { .S = S, .fd = fd };
    o = snprintf(text, cap, "%s", f->bos);
    for (const char *p = js_ws(msgs + 1); *p == '{'; n_msg++) {
        if (js_str(js_get(p, "role"), role, sizeof role) < 0 || message_text(js_get(p, "content"), content, cap) < 0 ||
            (strcmp(role, "system") && strcmp(role, "user") && strcmp(role, "assistant"))) {
            http_error(fd, 400, "each message needs a role (system, user, assistant) and text content"); goto done;
        }
        if (!n_msg && strcmp(role, "system")) o += snprintf(text + o, cap - o, f->sys, S->sys);    // default system prompt
        if (o < cap) o += snprintf(text + o, cap - o, f->msg, role, content);
        if (!(p = js_skip(p))) { http_error(fd, 400, "malformed JSON in messages"); goto done; }
        p = js_ws(p);
        if (*p == ',') p = js_ws(p + 1);
    }
    if (!n_msg) { http_error(fd, 400, "messages is empty"); goto done; }
    if (o < cap) o += snprintf(text + o, cap - o, "%s", f->gen);
    if (o >= cap) { http_error(fd, 400, "request too large"); goto done; }

    int ctx = S->m.n_ctx, max_toks = ctx + 128;
    toks = malloc(sizeof(int) * max_toks);
    int n = tokenize(&S->v, text, toks, max_toks);
    if (n < 1 || n > ctx - 16) {
        char e[96]; snprintf(e, sizeof e, "prompt is over %d tokens, context is %d", ctx - 16, ctx);
        http_error(fd, 400, e); goto done;
    }

    const char *mt = js_get(body, "max_tokens");
    if (!mt) mt = js_get(body, "max_completion_tokens");
    int limit = (int)js_num(mt, default_max);
    if (limit < 1) limit = 1;
    if (limit > ctx - n) limit = ctx - n;
    const char *sv = js_get(body, "stream");
    r.stream = sv && *sv == 't';
    sampler keep = S->s;
    S->s.temp = js_num(js_get(body, "temperature"), keep.temp);
    S->s.top_p = js_num(js_get(body, "top_p"), keep.top_p);
    S->s.top_k = (int)js_num(js_get(body, "top_k"), keep.top_k);
    S->s.repeat = js_num(js_get(body, "repeat_penalty"), keep.repeat);
    const char *seed = js_get(body, "seed");
    if (seed && *seed >= '0' && *seed <= '9') S->s.rng = strtoull(seed, 0, 10) | 1;

    int same = 0;
    double t0 = now();
    if (S->m.hybrid) same = prepare_hybrid_prompt(S, toks, n);
    else {
        if (!S->no_prefix_cache)
            while (same < S->n && same < n - 1 && S->hist[same] == toks[same]) same++;
        S->n = S->pos = same;
        if (feed(S, toks + same, n - same)) same = -1;
    }
    if (same < 0) {
        S->s = keep; http_error(fd, 500, "could not prepare prompt"); goto done;
    }
    double t1 = now();

    r.created = (long)time(0);
    snprintf(r.id, sizeof r.id, "chatcmpl-%lx%04x", r.created, (unsigned)(S->s.rng >> 20) & 0xffff);
    char out[1024];
    if (r.stream) {
        int k = snprintf(out, sizeof out, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
                         "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
                         "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
                         r.id, r.created, S->model_name);
        if (send_all(fd, out, k)) r.failed = 1;
    }
    int made = r.failed ? 0 : generate(S, limit, 0, emit_http, &r);
    double t2 = now();
    const char *finish = S->hit_stop ? "stop" : "length";
    uint64_t rng = S->s.rng;
    S->s = keep; S->s.rng = rng;

    char usage[256];
    snprintf(usage, sizeof usage, "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
             "\"timings\":{\"cached_tokens\":%d,\"prompt_per_second\":%.2f,\"predicted_per_second\":%.2f}",
             n, made, n + made, same, (n - same) / (t1 - t0 + 1e-9), made / (t2 - t1 + 1e-9));
    if (r.stream) {
        int k = snprintf(out, sizeof out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
                         "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],%s}\n\ndata: [DONE]\n\n",
                         r.id, r.created, S->model_name, finish, usage);
        if (!r.failed) send_all(fd, out, k);
    } else {
        char *esc = malloc(6 * r.acc_n + 1), *resp = malloc(6 * r.acc_n + 1024);
        js_escape(r.acc ? r.acc : "", r.acc_n, esc);
        int k = sprintf(resp, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"%s\","
                        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"%s\"}],%s}",
                        r.id, r.created, S->model_name, esc, finish, usage);
        http_send(fd, 200, "application/json", resp, k);
        free(esc); free(resp);
    }
    fprintf(stderr, "chat: %d prompt tokens (%d cached) at %.1f tok/s, %d generated at %.2f tok/s%s\n",
            n, same, (n - same) / (t1 - t0 + 1e-9), made, made / (t2 - t1 + 1e-9), r.failed ? ", client disconnected" : "");
done:
    free(text); free(content); free(toks); free(r.acc);
}

// ---------------------------------------------------------------- chat page
static const char PAGE[] =
"<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
"<title>pillm</title><style>"
":root{--bg:#f6f5f2;--fg:#1d1c1a;--mut:#6f6b64;--line:#dcd9d2;--me:#e4e9f2;--ai:#fff;--acc:#2f5fd0}"
"@media(prefers-color-scheme:dark){:root{--bg:#161719;--fg:#e9e7e2;--mut:#98948c;--line:#303236;--me:#253049;--ai:#1f2124;--acc:#8fb0ff}}"
"*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif;display:flex;flex-direction:column;height:100dvh}"
"header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}"
"header b{font-size:17px}header span{color:var(--mut);font-size:13px;font-variant-numeric:tabular-nums}"
"header button{margin-left:auto}"
"#log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}"
".m{max-width:min(46rem,92%);padding:10px 14px;border-radius:12px;white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid var(--line)}"
".u{align-self:flex-end;background:var(--me)}.a{align-self:flex-start;background:var(--ai)}"
"form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--line)}"
"textarea{flex:1;min-width:0;resize:none;height:3.4em;padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:var(--ai);color:inherit;font:inherit}"
"button{padding:6px 16px;border-radius:8px;border:1px solid var(--line);background:var(--ai);color:var(--acc);font:inherit;cursor:pointer}"
"button:disabled{opacity:.5;cursor:default}:focus-visible{outline:2px solid var(--acc);outline-offset:1px}"
"</style></head><body><header><b>pillm</b><span id=\"model\"></span><span id=\"stat\"></span><button id=\"new\" type=\"button\">\xe6\x96\xb0\xe5\xaf\xb9\xe8\xaf\x9d</button></header>"
"<div id=\"log\"></div><form id=\"f\"><textarea id=\"t\" placeholder=\"\xe8\xbe\x93\xe5\x85\xa5\xe6\xb6\x88\xe6\x81\xaf\xef\xbc\x8c\x45nter \xe5\x8f\x91\xe9\x80\x81\xef\xbc\x8cShift+Enter \xe6\x8d\xa2\xe8\xa1\x8c\" autofocus></textarea>"
"<button id=\"go\">\xe5\x8f\x91\xe9\x80\x81</button></form><script>"
"const $=id=>document.getElementById(id),log=$('log'),t=$('t'),go=$('go');let msgs=[],busy=false,ctl=null;"
"fetch('/v1/models').then(r=>r.json()).then(j=>$('model').textContent=j.data[0].id).catch(()=>{});"
"function add(c,s){const d=document.createElement('div');d.className='m '+c;d.textContent=s;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}"
"async function send(){if(busy){ctl&&ctl.abort();return}const q=t.value.trim();if(!q)return;t.value='';msgs.push({role:'user',content:q});add('u',q);"
"const el=add('a','');busy=true;go.textContent='\xe5\x81\x9c\xe6\xad\xa2';ctl=new AbortController();let out='';"
"try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},signal:ctl.signal,body:JSON.stringify({messages:msgs,stream:true})});"
"if(!r.ok){el.textContent='\xe5\x87\xba\xe9\x94\x99: '+(await r.text());msgs.pop()}else{const rd=r.body.getReader(),dec=new TextDecoder();let buf='';"
"for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;"
"while((i=buf.indexOf('\\n\\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+2);if(!line.startsWith('data: ')||line=='data: [DONE]')continue;"
"const j=JSON.parse(line.slice(6)),d=j.choices[0].delta.content;if(d){out+=d;el.textContent=out;log.scrollTop=log.scrollHeight}"
"if(j.timings)$('stat').textContent=j.timings.predicted_per_second.toFixed(2)+' token/s \xc2\xb7 '+j.usage.completion_tokens+' tokens'}}}}"
"catch(e){if(e.name!='AbortError')el.textContent=out+'\\n[\xe8\xbf\x9e\xe6\x8e\xa5\xe4\xb8\xad\xe6\x96\xad]'}"
"if(out)msgs.push({role:'assistant',content:out});busy=false;go.textContent='\xe5\x8f\x91\xe9\x80\x81';t.focus()}"
"$('f').onsubmit=e=>{e.preventDefault();send()};t.onkeydown=e=>{if(e.key=='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send()}};"
"$('new').onclick=()=>{if(busy)return;msgs=[];log.textContent='';$('stat').textContent='';t.focus()};"
"</script></body></html>";

// ---------------------------------------------------------------- request loop
#define REQ_MAX (4 << 20)

static void handle(session *S, int fd, int default_max) {
    size_t cap = 65536, n = 0;
    char *buf = malloc(cap + 1), *hdr_end = 0;
    for (;;) {                                                              // headers
        ssize_t k = recv(fd, buf + n, cap - n, 0);
        if (k <= 0) { free(buf); return; }
        n += k; buf[n] = 0;
        if ((hdr_end = strstr(buf, "\r\n\r\n"))) break;
        if (n == cap) { http_error(fd, 400, "headers too large"); free(buf); return; }
    }
    size_t head = hdr_end + 4 - buf, len = 0;
    *hdr_end = 0;
    const char *cl = strcasestr(buf, "\ncontent-length:");
    if (cl) len = strtoul(cl + 16, 0, 10);
    if (len > REQ_MAX) { http_error(fd, 400, "request too large"); free(buf); return; }
    if (head + len > cap) { cap = head + len; buf = realloc(buf, cap + 1); }
    while (n < head + len) {                                                // body
        ssize_t k = recv(fd, buf + n, head + len - n, 0);
        if (k <= 0) { free(buf); return; }
        n += k;
    }
    buf[head + len] = 0;
    const char *body = buf + head;

    char method[8] = "", path[128] = "";
    sscanf(buf, "%7s %127s", method, path);
    char *q = strchr(path, '?');
    if (q) *q = 0;
    if (!strcmp(method, "GET") && !strcmp(path, "/")) http_send(fd, 200, "text/html; charset=utf-8", PAGE, sizeof PAGE - 1);
    else if (!strcmp(method, "GET") && !strcmp(path, "/health")) http_send(fd, 200, "application/json", "{\"status\":\"ok\"}", 15);
    else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
        char b[512];
        int k = snprintf(b, sizeof b, "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"pillm\"}]}", S->model_name);
        http_send(fd, 200, "application/json", b, k);
    }
    else if (!strcmp(method, "POST") && !strcmp(path, "/v1/chat/completions")) handle_chat(S, fd, body, default_max);
    else http_error(fd, 404, "not found");
    free(buf);
}

static int serve(session *S, const char *host, int port, int default_max) {
    signal(SIGPIPE, SIG_IGN);
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port) };
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) die("--host %s is not an IPv4 address", host);
    if (ls < 0 || bind(ls, (struct sockaddr *)&a, sizeof a) || listen(ls, 8)) { perror("pillm: cannot listen"); return 1; }
    fprintf(stderr, "serving on http://%s:%d  (chat page at /, API at /v1/chat/completions)\n", host, port);
    for (;;) {
        int fd = accept(ls, 0, 0);
        if (fd < 0) continue;
        struct timeval tv = { .tv_sec = 15 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        handle(S, fd, default_max);
        close(fd);
    }
}

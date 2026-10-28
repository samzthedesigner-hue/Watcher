#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"

#ifndef WA_TEMP
#define WA_TEMP 0.6f
#endif
#ifndef WA_TOP_K
#define WA_TOP_K 40
#endif
#ifndef WA_TOP_P
#define WA_TOP_P 0.9f
#endif
#ifndef WA_REPEAT_PENALTY
#define WA_REPEAT_PENALTY 1.1f
#endif
#ifndef WA_MAX_TOKENS
#define WA_MAX_TOKENS 48
#endif
#ifndef WA_N_CTX
#define WA_N_CTX 1024
#endif
#ifndef WA_N_THREADS
#define WA_N_THREADS 4
#endif

struct H { llama_model* model; llama_context* ctx; };

static bool has_stop(const std::string& s) {
    return s.find("<|im_end|>") != std::string::npos ||
           s.find("<|im_start|>") != std::string::npos ||
           s.find("<|endoftext|>") != std::string::npos;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_whats_bot_LlamaHelper_loadModelNative(JNIEnv* env, jobject, jstring jpath) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.load_mode = LLAMA_LOAD_MODE_NONE;   // no mmap, load fully into RAM
    llama_model* m = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!m) return 0;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = WA_N_CTX;
    cp.n_batch = WA_N_CTX;
    cp.n_threads = WA_N_THREADS;
    cp.n_threads_batch = WA_N_THREADS;
    llama_context* c = llama_init_from_model(m, cp);
    if (!c) { llama_model_free(m); return 0; }
    return (jlong) new H{m, c};
}

JNIEXPORT jbyteArray JNICALL
Java_com_whats_bot_LlamaHelper_generateNative(JNIEnv* env, jobject, jlong ptr, jstring jprompt) {
    H* h = (H*) ptr;
    std::string out;
    if (h) {
        const char* p = env->GetStringUTFChars(jprompt, nullptr);
        std::string prompt(p);
        env->ReleaseStringUTFChars(jprompt, p);

        const llama_vocab* vocab = llama_model_get_vocab(h->model);
        llama_memory_clear(llama_get_memory(h->ctx), true);

        // parse_special=true so <|im_start|> / <|im_end|> become real tokens
        int n = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                nullptr, 0, true, true);
        int n_ctx = (int) llama_n_ctx(h->ctx);
        if (n > 0 && n + WA_MAX_TOKENS < n_ctx) {
            std::vector<llama_token> toks(n);
            if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                               toks.data(), n, true, true) >= 0) {

                llama_sampler* s = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(s, llama_sampler_init_penalties(
                        llama_vocab_n_tokens(vocab), 64, WA_REPEAT_PENALTY, 0.0f, 0.0f));
                llama_sampler_chain_add(s, llama_sampler_init_top_k(WA_TOP_K));
                llama_sampler_chain_add(s, llama_sampler_init_top_p(WA_TOP_P, 1));
                llama_sampler_chain_add(s, llama_sampler_init_temp(WA_TEMP));
                llama_sampler_chain_add(s, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

                llama_token tok = 0;
                llama_batch batch = llama_batch_get_one(toks.data(), n);
                for (int i = 0; i < WA_MAX_TOKENS; i++) {
                    if (llama_decode(h->ctx, batch) != 0) break;
                    tok = llama_sampler_sample(s, h->ctx, -1);
                    if (llama_vocab_is_eog(vocab, tok)) break;
                    char buf[256];
                    int len = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
                    if (len > 0) out.append(buf, len);
                    if (has_stop(out)) break;
                    if (out.size() > 1 && out.find('\n', 1) != std::string::npos) break;
                    batch = llama_batch_get_one(&tok, 1);
                }
                llama_sampler_free(s);
            }
        }
    }
    jbyteArray arr = env->NewByteArray((jsize) out.size());
    env->SetByteArrayRegion(arr, 0, (jsize) out.size(), (const jbyte*) out.data());
    return arr;
}

JNIEXPORT void JNICALL
Java_com_whats_bot_LlamaHelper_freeModelNative(JNIEnv*, jobject, jlong ptr) {
    H* h = (H*) ptr;
    if (!h) return;
    llama_free(h->ctx);
    llama_model_free(h->model);
    delete h;
}

}

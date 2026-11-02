#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>
#include <vector>

#include "llama.h"

#ifndef JNI_CLASS
#define JNI_CLASS "com/aegis/agent/LlamaBridge"
#endif

#define TAG "AgentLLM"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static std::atomic<bool> g_stop(false);
static int g_n_ctx = 4096;
static bool g_backend_ready = false;
static std::vector<llama_token> g_eog;

static void log_cb(ggml_log_level level, const char *text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        LOGE("%s", text);
    }
}

static void release_all() {
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_eog.clear();
}

static jboolean n_load(JNIEnv *env, jclass, jstring jpath, jint nCtx, jint nThreads) {
    release_all();
    if (!g_backend_ready) {
        llama_log_set(log_cb, nullptr);
        llama_backend_init();
        g_backend_ready = true;
    }
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.use_mmap = true;
    g_model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model) {
        LOGE("model load failed");
        return JNI_FALSE;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) nCtx;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = nThreads;
    cp.n_threads_batch = nThreads;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) {
        LOGE("context init failed");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }
    g_n_ctx = (int) llama_n_ctx(g_ctx);

    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    int nv = llama_vocab_n_tokens(vocab);
    for (int t = 0; t < nv; t++) {
        if (llama_vocab_is_eog(vocab, t)) {
            g_eog.push_back(t);
        }
    }
    return JNI_TRUE;
}

static std::string jstr(JNIEnv *env, jstring s) {
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    env->ReleaseStringUTFChars(s, c);
    return out;
}

static bool ends_sentence(const char *p, int len) {
    for (int i = 0; i < len; i++) {
        if (p[i] == '\n') {
            return true;
        }
    }
    if (len <= 0) {
        return false;
    }
    char c = p[len - 1];
    return c == '.' || c == '!' || c == '?';
}

/*
 * Generates a reply.
 *  minTokens: end-of-text is blocked until this many tokens exist.
 *  softMax:   after this many tokens, stop at the next sentence end.
 *  hardMax:   absolute cap.
 * Returns 0 on success, negative on error.
 */
static jint n_generate(JNIEnv *env, jclass, jobjectArray jroles, jobjectArray jcontents,
                       jint minTokens, jint softMax, jint hardMax, jfloat temperature,
                       jobject sink) {
    if (!g_model || !g_ctx) {
        return -1;
    }
    g_stop = false;

    // ---- chat messages ----
    jsize count = env->GetArrayLength(jroles);
    std::vector<std::string> roles(count), contents(count);
    std::vector<llama_chat_message> msgs(count);
    size_t total = 0;
    for (jsize i = 0; i < count; i++) {
        jstring r = (jstring) env->GetObjectArrayElement(jroles, i);
        jstring c = (jstring) env->GetObjectArrayElement(jcontents, i);
        roles[i] = jstr(env, r);
        contents[i] = jstr(env, c);
        env->DeleteLocalRef(r);
        env->DeleteLocalRef(c);
        total += roles[i].size() + contents[i].size();
    }
    for (jsize i = 0; i < count; i++) {
        msgs[i].role = roles[i].c_str();
        msgs[i].content = contents[i].c_str();
    }

    const char *tmpl = llama_model_chat_template(g_model, nullptr);
    std::vector<char> buf(total * 2 + 2048);
    int n = llama_chat_apply_template(tmpl ? tmpl : "chatml", msgs.data(), msgs.size(), true,
                                      buf.data(), (int32_t) buf.size());
    if (n > (int) buf.size()) {
        buf.resize(n + 16);
        n = llama_chat_apply_template(tmpl ? tmpl : "chatml", msgs.data(), msgs.size(), true,
                                      buf.data(), (int32_t) buf.size());
    }
    if (n < 0 && tmpl) {
        n = llama_chat_apply_template("chatml", msgs.data(), msgs.size(), true,
                                      buf.data(), (int32_t) buf.size());
        if (n > (int) buf.size()) {
            buf.resize(n + 16);
            n = llama_chat_apply_template("chatml", msgs.data(), msgs.size(), true,
                                          buf.data(), (int32_t) buf.size());
        }
    }
    if (n < 0) {
        return -2;
    }
    std::string prompt(buf.data(), n);

    // ---- tokenize ----
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    int nt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (nt <= 0) {
        return -3;
    }
    std::vector<llama_token> toks(nt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(),
                       (int32_t) toks.size(), true, true) < 0) {
        return -3;
    }
    if ((int) toks.size() + 32 >= g_n_ctx) {
        return -4;
    }
    int room = g_n_ctx - (int) toks.size() - 1;
    int hard = std::min((int) hardMax, room);
    int soft = std::min((int) softMax, hard);
    int minT = std::min((int) minTokens, hard);

    // ---- fresh state every call: keeps long loops stable ----
    llama_memory_clear(llama_get_memory(g_ctx), true);
    const int nb = 512;
    for (size_t i = 0; i < toks.size(); i += nb) {
        if (g_stop) {
            return 0;
        }
        int len = (int) std::min((size_t) nb, toks.size() - i);
        llama_batch b = llama_batch_get_one(toks.data() + i, len);
        if (llama_decode(g_ctx, b) != 0) {
            return -5;
        }
    }

    // ---- sampler tuned for small models ----
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(96, 1.15f, 0.0f, 0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    jclass sinkCls = env->GetObjectClass(sink);
    jmethodID onToken = env->GetMethodID(sinkCls, "onToken", "([B)V");
    int nv = llama_vocab_n_tokens(vocab);
    std::vector<llama_token_data> cand(nv);
    int rc = 0;

    for (int i = 0; i < hard && !g_stop; i++) {
        float *logits = llama_get_logits_ith(g_ctx, -1);
        if (!logits) {
            rc = -6;
            break;
        }
        for (int t = 0; t < nv; t++) {
            cand[t].id = t;
            cand[t].logit = logits[t];
            cand[t].p = 0.0f;
        }
        if (i < minT) {
            for (size_t k = 0; k < g_eog.size(); k++) {
                cand[g_eog[k]].logit = -INFINITY;
            }
        }
        llama_token_data_array arr = {cand.data(), (size_t) nv, -1, false};
        llama_sampler_apply(smpl, &arr);
        llama_token id = arr.data[arr.selected].id;
        llama_sampler_accept(smpl, id);

        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        char piece[256];
        int len = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, true);
        if (len > 0) {
            jbyteArray bytes = env->NewByteArray(len);
            env->SetByteArrayRegion(bytes, 0, len, (const jbyte *) piece);
            env->CallVoidMethod(sink, onToken, bytes);
            env->DeleteLocalRef(bytes);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                break;
            }
        }
        if (i + 1 >= soft && len > 0 && ends_sentence(piece, len)) {
            break;
        }
        if (g_stop) {
            break;
        }
        llama_batch b = llama_batch_get_one(&id, 1);
        if (llama_decode(g_ctx, b) != 0) {
            rc = -6;
            break;
        }
    }

    llama_sampler_free(smpl);
    return rc;
}

static void n_stop(JNIEnv *, jclass) {
    g_stop = true;
}

static void n_release(JNIEnv *, jclass) {
    g_stop = true;
    release_all();
}

static const JNINativeMethod METHODS[] = {
        {"nativeLoad", "(Ljava/lang/String;II)Z", (void *) n_load},
        {"nativeGenerate",
         "([Ljava/lang/String;[Ljava/lang/String;IIIFL" JNI_CLASS "$TokenSink;)I",
         (void *) n_generate},
        {"nativeStop", "()V", (void *) n_stop},
        {"nativeRelease", "()V", (void *) n_release},
};

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
    JNIEnv *env = nullptr;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    jclass cls = env->FindClass(JNI_CLASS);
    if (!cls) {
        return JNI_ERR;
    }
    if (env->RegisterNatives(cls, METHODS, sizeof(METHODS) / sizeof(METHODS[0])) != JNI_OK) {
        return JNI_ERR;
    }
    return JNI_VERSION_1_6;
}

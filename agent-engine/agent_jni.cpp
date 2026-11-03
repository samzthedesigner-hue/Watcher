#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

#include "llama.h"

#ifndef JNI_CLASS
#define JNI_CLASS "com/aegis/agent/LlamaBridge"
#endif

#define TAG "AgentLLM"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct Slot {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    int n_ctx = 4096;
    int threads = 4;
    std::vector<llama_token> cache;
};

static Slot g_slots[2];
static std::atomic<bool> g_stop(false);
static bool g_backend_ready = false;

static void log_cb(ggml_log_level level, const char *text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        LOGE("%s", text);
    }
}

static void free_ctx(Slot &s) {
    if (s.ctx) {
        llama_free(s.ctx);
        s.ctx = nullptr;
    }
    s.cache.clear();
}

static void free_all(Slot &s) {
    free_ctx(s);
    if (s.model) {
        llama_model_free(s.model);
        s.model = nullptr;
    }
}

static bool ensure_ctx(Slot &s) {
    if (s.ctx) {
        return true;
    }
    if (!s.model) {
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) s.n_ctx;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = s.threads;
    cp.n_threads_batch = s.threads;
    s.ctx = llama_init_from_model(s.model, cp);
    if (!s.ctx) {
        LOGE("context init failed");
        return false;
    }
    s.n_ctx = (int) llama_n_ctx(s.ctx);
    s.cache.clear();
    return true;
}

static std::string jstr(JNIEnv *env, jstring js) {
    const char *c = env->GetStringUTFChars(js, nullptr);
    std::string out(c ? c : "");
    env->ReleaseStringUTFChars(js, c);
    return out;
}

static jboolean n_load(JNIEnv *env, jclass, jint slot, jstring jpath, jint nCtx, jint nThreads) {
    if (slot < 0 || slot > 1) {
        return JNI_FALSE;
    }
    Slot &s = g_slots[slot];
    free_all(s);
    if (!g_backend_ready) {
        llama_log_set(log_cb, nullptr);
        llama_backend_init();
        g_backend_ready = true;
    }
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    s.model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!s.model) {
        LOGE("model load failed (slot %d)", (int) slot);
        return JNI_FALSE;
    }
    s.n_ctx = nCtx;
    s.threads = nThreads;
    return JNI_TRUE;
}

static void n_free_ctx(JNIEnv *, jclass, jint slot) {
    if (slot >= 0 && slot <= 1) {
        free_ctx(g_slots[slot]);
    }
}

static void n_release(JNIEnv *, jclass, jint slot) {
    if (slot >= 0 && slot <= 1) {
        g_stop = true;
        free_all(g_slots[slot]);
    }
}

static void n_stop(JNIEnv *, jclass) {
    g_stop = true;
}

/*
 * Returns 0 on success, negative on error:
 * -1 no model, -2 template, -3 tokenize, -4 prompt too long, -5 prompt decode, -6 decode, -7 no context
 */
static jint n_generate(JNIEnv *env, jclass, jint slot, jobjectArray jroles, jobjectArray jcontents,
                       jint maxTokens, jfloat temperature, jobject sink) {
    if (slot < 0 || slot > 1) {
        return -1;
    }
    Slot &s = g_slots[slot];
    if (!s.model) {
        return -1;
    }
    if (!ensure_ctx(s)) {
        return -7;
    }
    g_stop = false;

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

    const char *tmpl = llama_model_chat_template(s.model, nullptr);
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

    const llama_vocab *vocab = llama_model_get_vocab(s.model);
    int nt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (nt <= 0) {
        return -3;
    }
    std::vector<llama_token> toks(nt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(),
                       (int32_t) toks.size(), true, true) < 0) {
        return -3;
    }
    if ((int) toks.size() + 32 >= s.n_ctx) {
        return -4;
    }
    int maxGen = std::min((int) maxTokens, s.n_ctx - (int) toks.size() - 1);
    if (maxGen <= 0) {
        return -4;
    }

    // Reuse the KV cache for the part of the prompt that matches the previous call.
    llama_memory_t mem = llama_get_memory(s.ctx);
    size_t keep = 0;
    size_t maxKeep = toks.size() - 1;
    while (keep < s.cache.size() && keep < maxKeep && s.cache[keep] == toks[keep]) {
        keep++;
    }
    if (keep == 0 || !llama_memory_seq_rm(mem, 0, (llama_pos) keep, -1)) {
        llama_memory_clear(mem, true);
        keep = 0;
    }
    s.cache.assign(toks.begin(), toks.begin() + keep);

    const int nb = 512;
    for (size_t i = keep; i < toks.size(); i += nb) {
        if (g_stop) {
            s.cache.clear();
            llama_memory_clear(mem, true);
            return 0;
        }
        int len = (int) std::min((size_t) nb, toks.size() - i);
        llama_batch b = llama_batch_get_one(toks.data() + i, len);
        if (llama_decode(s.ctx, b) != 0) {
            s.cache.clear();
            llama_memory_clear(mem, true);
            return -5;
        }
        s.cache.insert(s.cache.end(), toks.begin() + i, toks.begin() + i + len);
    }

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 128, 1.1f, 0.0f, 0.0f));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    jclass sinkCls = env->GetObjectClass(sink);
    jmethodID onToken = env->GetMethodID(sinkCls, "onToken", "([B)V");
    int rc = 0;

    for (int i = 0; i < maxGen && !g_stop; i++) {
        llama_token id = llama_sampler_sample(smpl, s.ctx, -1);
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
        if (g_stop) {
            break;
        }
        llama_batch b = llama_batch_get_one(&id, 1);
        if (llama_decode(s.ctx, b) != 0) {
            rc = -6;
            s.cache.clear();
            llama_memory_clear(mem, true);
            break;
        }
        s.cache.push_back(id);
    }

    llama_sampler_free(smpl);
    return rc;
}

static const JNINativeMethod METHODS[] = {
        {"nativeLoad", "(ILjava/lang/String;II)Z", (void *) n_load},
        {"nativeFreeContext", "(I)V", (void *) n_free_ctx},
        {"nativeGenerate",
         "(I[Ljava/lang/String;[Ljava/lang/String;IFL" JNI_CLASS "$TokenSink;)I",
         (void *) n_generate},
        {"nativeStop", "()V", (void *) n_stop},
        {"nativeRelease", "(I)V", (void *) n_release},
};

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
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

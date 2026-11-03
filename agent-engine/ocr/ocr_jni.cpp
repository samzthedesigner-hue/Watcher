#include <jni.h>
#include <string>
#include <tesseract/baseapi.h>

#ifndef OCR_CLASS
#define OCR_CLASS "com/aegis/agent/Ocr"
#endif

static tesseract::TessBaseAPI *g_api = nullptr;

static void drop_api() {
    if (g_api) {
        g_api->End();
        delete g_api;
        g_api = nullptr;
    }
}

static jboolean o_init(JNIEnv *env, jclass, jstring jdir, jstring jlang) {
    drop_api();
    const char *dir = env->GetStringUTFChars(jdir, nullptr);
    const char *lang = env->GetStringUTFChars(jlang, nullptr);
    g_api = new tesseract::TessBaseAPI();
    int rc = g_api->Init(dir, lang, tesseract::OEM_LSTM_ONLY);
    env->ReleaseStringUTFChars(jdir, dir);
    env->ReleaseStringUTFChars(jlang, lang);
    if (rc != 0) {
        delete g_api;
        g_api = nullptr;
        return JNI_FALSE;
    }
    g_api->SetPageSegMode(tesseract::PSM_AUTO);
    return JNI_TRUE;
}

static jbyteArray o_recognize(JNIEnv *env, jclass, jbyteArray jdata, jint w, jint h) {
    std::string text;
    if (g_api) {
        jbyte *data = env->GetByteArrayElements(jdata, nullptr);
        g_api->SetImage((const unsigned char *) data, w, h, 1, w);
        g_api->SetSourceResolution(200);
        char *out = g_api->GetUTF8Text();
        if (out) {
            text = out;
            delete[] out;
        }
        g_api->Clear();
        env->ReleaseByteArrayElements(jdata, data, JNI_ABORT);
    }
    jbyteArray res = env->NewByteArray((jsize) text.size());
    if (!text.empty()) {
        env->SetByteArrayRegion(res, 0, (jsize) text.size(), (const jbyte *) text.data());
    }
    return res;
}

static void o_release(JNIEnv *, jclass) {
    drop_api();
}

static const JNINativeMethod METHODS[] = {
        {"nativeInit", "(Ljava/lang/String;Ljava/lang/String;)Z", (void *) o_init},
        {"nativeRecognize", "([BII)[B", (void *) o_recognize},
        {"nativeRelease", "()V", (void *) o_release},
};

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
    JNIEnv *env = nullptr;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    jclass cls = env->FindClass(OCR_CLASS);
    if (!cls) {
        return JNI_ERR;
    }
    if (env->RegisterNatives(cls, METHODS, sizeof(METHODS) / sizeof(METHODS[0])) != JNI_OK) {
        return JNI_ERR;
    }
    return JNI_VERSION_1_6;
}

/**
 * whisper_wasm.cpp
 *
 * Custom Emscripten bindings for whisper.cpp.
 * Exposes a minimal API for batch transcription from Float32 PCM data.
 *
 * Exported functions (via --bind):
 *   whisper_init(modelPath: string): number     — returns context handle (>0) or 0 on error
 *   whisper_free(handle: number): void
 *   whisper_transcribe(handle: number, pcmData: Float32Array, lang: string, nThreads: number): string
 *
 * The transcription is synchronous within the WASM worker thread.
 * The web worker wrapping this module runs the call in a dedicated thread via
 * pthreads, so the main browser thread is never blocked.
 */

#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <string>
#include <vector>
#include <algorithm>

// Up to 4 concurrent contexts (one is typical)
static std::vector<whisper_context*> g_contexts(4, nullptr);

// Returns 1-based handle (0 = error)
static size_t ctx_init(const std::string& path_model) {
    for (size_t i = 0; i < g_contexts.size(); ++i) {
        if (g_contexts[i] == nullptr) {
            whisper_context_params cparams = whisper_context_default_params();
            cparams.use_gpu = false;  // CPU-only in WASM
            g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), cparams);
            if (g_contexts[i] != nullptr) {
                return i + 1;
            }
            return 0;
        }
    }
    return 0;
}

static void ctx_free(size_t handle) {
    if (handle == 0 || handle > g_contexts.size()) return;
    size_t idx = handle - 1;
    if (g_contexts[idx] != nullptr) {
        whisper_free(g_contexts[idx]);
        g_contexts[idx] = nullptr;
    }
}

// Synchronous transcription.
// pcm_val: a JS Float32Array passed from the worker
// Returns the full transcript as a single string.
static std::string ctx_transcribe(size_t handle, const emscripten::val& pcm_val,
                                   const std::string& lang, int n_threads) {
    if (handle == 0 || handle > g_contexts.size()) return "";
    size_t idx = handle - 1;
    whisper_context* ctx = g_contexts[idx];
    if (ctx == nullptr) return "";

    // Copy Float32Array from JS → C++ vector
    const int n = pcm_val["length"].as<int>();
    std::vector<float> pcmf32(n);

    emscripten::val heap   = emscripten::val::module_property("HEAPU8");
    emscripten::val memory = heap["buffer"];
    emscripten::val view   = pcm_val["constructor"].new_(memory,
                                 reinterpret_cast<uintptr_t>(pcmf32.data()), n);
    view.call<void>("set", pcm_val);

    // Configure inference params
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime   = false;
    params.print_progress   = false;
    params.print_timestamps = false;
    params.print_special    = false;
    params.translate        = false;
    params.single_segment   = false;
    params.no_context       = true;
    params.n_threads        = std::max(1, std::min(n_threads, 16));
    params.offset_ms        = 0;
    params.duration_ms      = 0;  // full audio
    params.suppress_blank   = true;
    params.suppress_nst     = true;

    bool is_multilingual = whisper_is_multilingual(ctx);
    if (is_multilingual) {
        params.language = lang.c_str();
    } else {
        params.language = "en";
    }

    // Run inference (synchronous in this thread)
    if (whisper_full(ctx, params, pcmf32.data(), (int)pcmf32.size()) != 0) {
        return "[whisper_full failed]";
    }

    // Collect all segments into a single string
    std::string result;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        if (text) {
            result += text;
        }
    }

    return result;
}

EMSCRIPTEN_BINDINGS(whisper) {
    emscripten::function("whisper_init",       emscripten::optional_override(
        [](const std::string& path) { return ctx_init(path); }));

    emscripten::function("whisper_free",       emscripten::optional_override(
        [](size_t handle) { ctx_free(handle); }));

    emscripten::function("whisper_transcribe", emscripten::optional_override(
        [](size_t handle, const emscripten::val& pcm,
           const std::string& lang, int n_threads) -> std::string {
            return ctx_transcribe(handle, pcm, lang, n_threads);
        }));
}

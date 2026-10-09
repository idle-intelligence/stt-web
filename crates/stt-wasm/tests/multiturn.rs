//! Back-to-back turns in one session: two distinct utterances fed through
//! the same SttStream with `reset_keep_buffers()` between them (mirrors
//! what `web/bindings.rs::reset()` does between recordings), checking that
//! the second turn isn't corrupted by the first turn's KV cache / delay
//! pipeline / cycle-guard state.
//!
//! Needs model weights that are not committed to the repo (same convention
//! as crates/stt-wasm/tests/e2e_transcript.rs); skips with a message when
//! they are absent, so this does not run in CI.
//!
//! Run: cargo test -p stt-wasm --release --test multiturn -- --nocapture

use burn::backend::wgpu::WgpuDevice;

use stt_wasm::gguf::Q4ModelLoader;
use stt_wasm::mimi_encoder::MimiEncoder;
use stt_wasm::stream::SttStream;
use stt_wasm::tokenizer::SpmDecoder;
use stt_wasm::SttConfig;

fn read_wav_f32(path: &std::path::Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let spec = reader.spec();
    match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / (1i64 << (bits - 1)) as f32)
                .collect()
        }
    }
}

async fn transcribe_turn(
    samples: &[f32],
    mimi: &mut MimiEncoder,
    stream: &mut SttStream,
    model: &stt_wasm::model::SttModel,
    config: &SttConfig,
    tokenizer: &SpmDecoder,
) -> String {
    let chunk_size = 1920;
    let num_codebooks = config.num_codebooks;
    let mut all_tokens: Vec<u32> = Vec::new();

    for chunk_start in (0..samples.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(samples.len());
        let chunk = &samples[chunk_start..chunk_end];
        let tokens = mimi.feed_audio(chunk);
        for frame_start in (0..tokens.len()).step_by(num_codebooks) {
            if frame_start + num_codebooks > tokens.len() {
                break;
            }
            let frame = &tokens[frame_start..frame_start + num_codebooks];
            if let Some(token) = stream.feed_frame(frame, model).await {
                all_tokens.push(token);
            }
        }
    }

    let flush_tokens = stream.flush(model).await;
    all_tokens.extend(&flush_tokens);
    tokenizer.decode(&all_tokens)
}

#[test]
fn back_to_back_turns_same_session() {
    pollster::block_on(async {
        let fixtures_dir = std::path::Path::new("../../tests/fixtures");
        let mimi_weights_path = "../../models/mimi-encoder-f16.safetensors";
        let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4_0.gguf");
        let tokenizer_path = std::path::Path::new("../../models/tokenizer.model");

        if !gguf_path.exists() || !std::path::Path::new(mimi_weights_path).exists() {
            println!("Skipping: model files not found (not committed to the repo)");
            return;
        }

        let tokenizer = SpmDecoder::load(tokenizer_path.to_str().unwrap()).await.unwrap();
        let device = WgpuDevice::default();
        let config = SttConfig::default();
        let file_data = std::fs::read(gguf_path).unwrap();
        let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
        let parts = loader.load_deferred(&device, &config).unwrap();
        drop(loader);
        let model = parts.finalize(&device).unwrap();

        // Three distinct utterances, back to back, one SttStream, reset
        // between turns (not a fresh stream/cache each time).
        let turns: &[(&str, &str)] = &[
            ("demo-1-vera.wav", "Hello, can you tell me a joke about yourself?"),
            ("demo-2-vera.wav", "That's pretty funny. Can you explain why?"),
            ("demo-3-vera.wav", "Now tell me one about cats."),
        ];

        let mut stream = SttStream::new(config.clone(), model.create_cache());
        let mimi_data = std::fs::read(mimi_weights_path).unwrap();

        // WER threshold, not exact match: this test checks turns don't
        // corrupt each other's state (KV cache, delay pipeline, cycle
        // guard), not Q4 quantization noise, which is already covered
        // (and not gated hard) by regression.rs.
        const WER_BOUND: f64 = 0.3;

        for (i, (file, expected)) in turns.iter().enumerate() {
            let mut mimi = MimiEncoder::from_bytes(&mimi_data).unwrap();
            let samples = read_wav_f32(&fixtures_dir.join(file));
            let got = transcribe_turn(&samples, &mut mimi, &mut stream, &model, &config, &tokenizer).await;
            let w = wer(expected, &got);
            println!("turn {i} ({file}): wer={:.1}% expected={expected:?} got={got:?}", w * 100.0);

            assert!(
                w <= WER_BOUND,
                "turn {i} ({file}): {:.1}% WER exceeds {:.0}% bound; expected {expected:?}, got {got:?}",
                w * 100.0, WER_BOUND * 100.0,
            );

            // Reset for the next turn, same as web/bindings.rs::reset() between recordings.
            stream.reset_keep_buffers();
        }

        println!("\n=== back-to-back turns: all {} turns within WER bound ===", turns.len());
    });
}

/// Lowercase a word and strip surrounding punctuation for WER comparison.
fn normalize_word(w: &str) -> String {
    w.to_lowercase()
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_string()
}

/// Normalized word-error rate: Levenshtein distance over words / reference word count.
fn wer(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<String> = reference.split_whitespace().map(normalize_word).collect();
    let h: Vec<String> = hypothesis.split_whitespace().map(normalize_word).collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let (n, m) = (r.len(), h.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if r[i - 1] == h[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }
    dp[n][m] as f64 / n as f64
}

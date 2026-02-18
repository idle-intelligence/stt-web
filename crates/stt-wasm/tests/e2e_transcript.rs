//! End-to-end test: reference Mimi tokens → STT model → text transcript.
//!
//! Feeds ALL 568 reference frames through the streaming decoder,
//! decodes the output tokens to text using sentencepiece, and
//! compares the result to the expected transcript.
//!
//! Run: cargo test -p stt-wasm --features wgpu --test e2e_transcript -- --nocapture

use burn::backend::wgpu::WgpuDevice;

use stt_wasm::gguf::Q4ModelLoader;
use stt_wasm::stream::SttStream;
use stt_wasm::tokenizer::SpmDecoder;
use stt_wasm::SttConfig;

fn device() -> WgpuDevice {
    WgpuDevice::default()
}

#[derive(serde::Deserialize)]
struct MimiTokens {
    tokens: Vec<Vec<u32>>,
}

#[derive(serde::Deserialize)]
struct TextTokens {
    tokens: Vec<u32>,
}

#[test]
fn test_e2e_full_transcript() {
    pollster::block_on(async {
        // Paths
        let mimi_path = std::path::Path::new("../../tests/reference/mimi_tokens.json");
        let text_path = std::path::Path::new("../../tests/reference/text_tokens.json");
        let transcript_path = std::path::Path::new("../../tests/reference/transcript.txt");
        let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
        let tokenizer_path = std::path::Path::new("../../models/tokenizer.model");

        if !gguf_path.exists() {
            println!("Skipping: GGUF not found at {}", gguf_path.display());
            return;
        }

        // Load reference data
        let mimi_data: MimiTokens =
            serde_json::from_str(&std::fs::read_to_string(mimi_path).unwrap()).unwrap();
        let text_data: TextTokens =
            serde_json::from_str(&std::fs::read_to_string(text_path).unwrap()).unwrap();
        let expected_transcript = std::fs::read_to_string(transcript_path).unwrap().trim().to_string();

        let num_frames = mimi_data.tokens.len();
        println!("Reference: {} Mimi frames, {} text tokens", num_frames, text_data.tokens.len());
        println!("Expected: {:?}...", &expected_transcript[..80.min(expected_transcript.len())]);

        // Load tokenizer vocabulary
        let tokenizer = SpmDecoder::load(tokenizer_path.to_str().unwrap()).await.unwrap();
        println!("Loaded tokenizer: {} vocab entries", tokenizer.vocab_len());

        // Load model
        let device = device();
        let config = SttConfig::default();

        println!("Loading model...");
        let file_data = std::fs::read(gguf_path).unwrap();
        let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
        let parts = loader.load_deferred(&device, &config).unwrap();
        drop(loader);
        let model = parts.finalize(&device).unwrap();
        println!("Model loaded.");

        // Create streaming state
        let mut stream = SttStream::new(config.clone(), model.create_cache());

        // Feed ALL frames
        let mut predicted_tokens: Vec<u32> = Vec::new();
        let mut matches = 0;
        let mut mismatches = 0;

        for i in 0..num_frames {
            let frame = &mimi_data.tokens[i];
            let token = stream.feed_frame(frame, &model).await;

            if let Some(t) = token {
                predicted_tokens.push(t);
                if t == text_data.tokens[i] {
                    matches += 1;
                } else {
                    mismatches += 1;
                }
            }

            // Progress
            if (i + 1) % 100 == 0 {
                eprintln!("  [{}/{}]", i + 1, num_frames);
            }
        }

        // Flush remaining tokens (for delayed text)
        let flush_tokens = stream.flush(&model).await;
        println!("Flush produced {} additional tokens: {:?}", flush_tokens.len(), flush_tokens);
        predicted_tokens.extend(&flush_tokens);

        println!("\n=== TOKEN STATS ===");
        println!("Total frames: {}", num_frames);
        println!("Warmup frames (no output): {}", config.text_delay);
        println!("Token matches: {}", matches);
        println!("Token mismatches: {}", mismatches);
        println!("Match rate: {:.1}%", 100.0 * matches as f64 / (matches + mismatches) as f64);
        println!("Total predicted tokens: {}", predicted_tokens.len());

        // Decode predicted tokens to text
        let predicted_text = tokenizer.decode(&predicted_tokens);

        // Also decode reference tokens for comparison
        let ref_text = tokenizer.decode(&text_data.tokens);

        println!("\n=== PREDICTED TRANSCRIPT ===");
        println!("{}", predicted_text);

        println!("\n=== REFERENCE (from tokens) ===");
        println!("{}", ref_text);

        println!("\n=== EXPECTED (from file) ===");
        println!("{}", expected_transcript);

        // Word-level comparison
        let predicted_words: Vec<&str> = predicted_text.split_whitespace().collect();
        let expected_words: Vec<&str> = expected_transcript.split_whitespace().collect();
        let ref_words: Vec<&str> = ref_text.split_whitespace().collect();

        println!("\n=== WORD COMPARISON ===");
        println!("Predicted: {} words", predicted_words.len());
        println!("Reference (tokens): {} words", ref_words.len());
        println!("Expected (file): {} words", expected_words.len());

        // Simple word match against expected transcript
        let min_len = predicted_words.len().min(expected_words.len());
        let mut word_matches = 0;
        let mut first_mismatches = Vec::new();

        for i in 0..min_len {
            if predicted_words[i].to_lowercase() == expected_words[i].to_lowercase() {
                word_matches += 1;
            } else if first_mismatches.len() < 20 {
                first_mismatches.push((i, predicted_words[i], expected_words[i]));
            }
        }

        let total_words = expected_words.len().max(predicted_words.len());
        let wer = 1.0 - (word_matches as f64 / total_words as f64);

        println!("Word matches: {}/{}", word_matches, min_len);
        println!("Approx WER: {:.1}%", wer * 100.0);

        if !first_mismatches.is_empty() {
            println!("\nFirst mismatched words:");
            for (i, pred, exp) in &first_mismatches {
                println!("  Word {}: predicted={:?} expected={:?}", i, pred, exp);
            }
        }

        // Assertions
        let non_special = predicted_tokens.iter().filter(|&&t| t != 0 && t != 3).count();
        assert!(
            non_special > 50,
            "Should produce substantial text (>50 non-special tokens), got {}", non_special
        );

        let predicted_lower = predicted_text.to_lowercase();
        assert!(
            predicted_lower.contains("forest"),
            "Transcript should mention 'forest': {:?}", predicted_text
        );

        println!("\n=== E2E TEST PASSED ===");
    });
}

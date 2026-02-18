//! Streaming test: feed reference Mimi tokens through the full streaming pipeline
//! and compare output with reference text tokens.
//!
//! This simulates exactly what happens in the browser:
//! SttStream.feed_frame() called once per Mimi frame.

use burn::backend::wgpu::WgpuDevice;

use stt_wasm::gguf::Q4ModelLoader;
use stt_wasm::stream::SttStream;
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
fn test_streaming_with_reference_tokens() {
    pollster::block_on(async {
    // Load reference data
    let mimi_path = std::path::Path::new("../../tests/reference/mimi_tokens.json");
    let text_path = std::path::Path::new("../../tests/reference/text_tokens.json");
    let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");

    if !gguf_path.exists() {
        println!("Skipping: GGUF not found at {}", gguf_path.display());
        return;
    }

    let mimi_data: MimiTokens =
        serde_json::from_str(&std::fs::read_to_string(mimi_path).unwrap()).unwrap();
    let text_data: TextTokens =
        serde_json::from_str(&std::fs::read_to_string(text_path).unwrap()).unwrap();

    println!(
        "Reference: {} Mimi frames, {} text tokens",
        mimi_data.tokens.len(),
        text_data.tokens.len()
    );

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

    // Create streaming state with pre-allocated KV cache
    let mut stream = SttStream::new(config.clone(), model.create_cache());

    // Feed first 50 frames and collect predicted tokens
    let num_frames = 50.min(mimi_data.tokens.len());
    let mut predicted_tokens: Vec<Option<u32>> = Vec::new();

    for i in 0..num_frames {
        let frame = &mimi_data.tokens[i];
        let token = stream.feed_frame(frame, &model).await;
        predicted_tokens.push(token);

        let ref_token = text_data.tokens[i];
        let pred_str = match token {
            Some(t) => format!("{t}"),
            None => "None (warmup)".to_string(),
        };
        let match_str = match token {
            Some(t) if t == ref_token => "MATCH",
            Some(_) => "MISMATCH",
            None => "",
        };
        println!(
            "Frame {:3}: predicted={:>15} reference={:>5} {}",
            i, pred_str, ref_token, match_str
        );
    }

    // Summary
    let mut matches = 0;
    let mut mismatches = 0;
    let mut non_padding_predicted = 0;

    for (i, pred) in predicted_tokens.iter().enumerate() {
        if let Some(token) = pred {
            if *token != 3 {
                non_padding_predicted += 1;
            }
            if *token == text_data.tokens[i] {
                matches += 1;
            } else {
                mismatches += 1;
            }
        }
    }

    println!("\n=== SUMMARY (first {num_frames} frames) ===");
    println!("Warmup frames (None): {}", config.text_delay);
    println!("Matches: {matches}");
    println!("Mismatches: {mismatches}");
    println!("Non-padding tokens predicted: {non_padding_predicted}");
    println!(
        "Reference non-padding tokens: {}",
        text_data.tokens[..num_frames]
            .iter()
            .filter(|&&t| t != 3)
            .count()
    );

    // The model should produce at least SOME non-padding tokens
    assert!(
        non_padding_predicted > 0,
        "Model produced ONLY padding tokens — streaming is broken"
    );
    }); // end pollster::block_on
}

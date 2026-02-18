//! Integration test: full model forward pass with real GGUF weights.
//!
//! Loads the actual model, runs multiple forward steps with different tokens,
//! and checks if the output logits vary.

use burn::backend::wgpu::WgpuDevice;

use stt_wasm::gguf::Q4ModelLoader;
use stt_wasm::SttConfig;

fn device() -> WgpuDevice {
    WgpuDevice::default()
}

#[test]
fn test_forward_produces_varying_logits() {
    let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
    if !gguf_path.exists() {
        println!("Skipping: GGUF not found at {}", gguf_path.display());
        return;
    }

    let device = device();
    let config = SttConfig::default();

    // Load model
    println!("Loading model from GGUF...");
    let file_data = std::fs::read(gguf_path).unwrap();
    let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
    let parts = loader.load_deferred(&device, &config).unwrap();
    drop(loader);
    let model = parts.finalize(&device).unwrap();
    println!("Model loaded: {} layers", config.num_layers);

    // Create cache
    let mut cache = model.create_cache();

    // Run several forward steps with different audio tokens
    let test_frames: Vec<Vec<u32>> = vec![
        vec![326, 955, 1016, 546, 1200, 400, 800, 1500, 100, 200, 300, 400, 500, 600, 700, 800,
             900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 50, 150, 250, 350],
        vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
             1700, 1800, 1900, 2000, 50, 150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150],
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    ];

    let text_token = config.text_padding_id;
    let mut logits_list = Vec::new();

    for (i, frame) in test_frames.iter().enumerate() {
        let logits = model.forward(frame, text_token, &mut cache);
        let logits_data: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap();

        // Top-5
        let mut indexed: Vec<(usize, f32)> = logits_data.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();
        let argmax = top5[0].0;

        let min = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("Frame {i}: argmax={argmax} top5={top5:?} min={min:.4} max={max:.4}");
        logits_list.push(logits_data);
    }

    // Check if logits differ between frames
    for i in 0..logits_list.len() {
        for j in (i + 1)..logits_list.len() {
            let diff: f32 = logits_list[i]
                .iter()
                .zip(logits_list[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let max_diff: f32 = logits_list[i]
                .iter()
                .zip(logits_list[j].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("Frame {i} vs {j}: sum_diff={diff:.4} max_diff={max_diff:.4}");

            if diff < 1.0 {
                println!("WARNING: Frames {i} and {j} have nearly identical logits!");
            }
        }
    }
}

#[test]
fn test_forward_no_cache_varying() {
    // Test without KV cache: run forward on different frames with fresh caches
    let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
    if !gguf_path.exists() {
        println!("Skipping: GGUF not found at {}", gguf_path.display());
        return;
    }

    let device = device();
    let config = SttConfig::default();

    println!("Loading model from GGUF...");
    let file_data = std::fs::read(gguf_path).unwrap();
    let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
    let parts = loader.load_deferred(&device, &config).unwrap();
    drop(loader);
    let model = parts.finalize(&device).unwrap();

    // Two different token sets, FRESH cache each time
    let frame1: Vec<u32> = vec![326, 955, 1016, 546, 1200, 400, 800, 1500, 100, 200, 300, 400, 500, 600, 700, 800,
         900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 50, 150, 250, 350];
    let frame2: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

    let mut cache1 = model.create_cache();
    let mut cache2 = model.create_cache();

    let logits1: Vec<f32> = model
        .forward(&frame1, config.text_padding_id, &mut cache1)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let logits2: Vec<f32> = model
        .forward(&frame2, config.text_padding_id, &mut cache2)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // Top-5 for each
    let top5 = |logits: &[f32]| -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().take(5).collect()
    };

    println!("Frame 1 top5: {:?}", top5(&logits1));
    println!("Frame 2 top5: {:?}", top5(&logits2));

    let diff: f32 = logits1
        .iter()
        .zip(logits2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let max_diff: f32 = logits1
        .iter()
        .zip(logits2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Diff: sum={diff:.4} max={max_diff:.6}");

    assert!(
        diff > 1.0,
        "FROZEN LOGITS: different inputs with fresh caches produced same output! diff={diff}"
    );
}

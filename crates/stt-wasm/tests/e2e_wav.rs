//! True end-to-end test: WAV file → Mimi encoder → STT transformer → text.
//!
//! This is the full pipeline with no pre-computed tokens.
//!
//! Run: cargo test -p stt-wasm --features wgpu --test e2e_wav -- --nocapture

use burn::backend::wgpu::WgpuDevice;

use stt_wasm::gguf::Q4ModelLoader;
use stt_wasm::stream::SttStream;
use stt_wasm::SttConfig;

fn device() -> WgpuDevice {
    WgpuDevice::default()
}

/// Minimal SentencePiece .model parser (protobuf).
fn load_sentencepiece_vocab(path: &std::path::Path) -> Vec<String> {
    let data = std::fs::read(path).expect("Failed to read tokenizer model");
    let mut pieces = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let (tag, tag_len) = read_varint(&data, pos);
        pos += tag_len;
        let field_num = tag >> 3;
        let wire_type = tag & 0x7;
        if field_num == 1 && wire_type == 2 {
            let (len, len_len) = read_varint(&data, pos);
            pos += len_len;
            let piece = parse_sentence_piece(&data[pos..pos + len as usize]);
            pieces.push(piece);
            pos += len as usize;
        } else {
            pos = skip_field(&data, pos, wire_type as u8);
        }
    }
    pieces
}

fn parse_sentence_piece(buf: &[u8]) -> String {
    let mut pos = 0;
    let mut piece = String::new();
    while pos < buf.len() {
        let (tag, tag_len) = read_varint(buf, pos);
        pos += tag_len;
        let field_num = tag >> 3;
        let wire_type = tag & 0x7;
        if field_num == 1 && wire_type == 2 {
            let (len, len_len) = read_varint(buf, pos);
            pos += len_len;
            piece = String::from_utf8_lossy(&buf[pos..pos + len as usize]).to_string();
            pos += len as usize;
        } else {
            pos = skip_field(buf, pos, wire_type as u8);
        }
    }
    piece
}

fn read_varint(buf: &[u8], start: usize) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut pos = start;
    while pos < buf.len() {
        let byte = buf[pos];
        pos += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            break;
        }
    }
    (value, pos - start)
}

fn skip_field(buf: &[u8], pos: usize, wire_type: u8) -> usize {
    match wire_type {
        0 => {
            let mut p = pos;
            while p < buf.len() && buf[p] & 0x80 != 0 { p += 1; }
            p + 1
        }
        1 => pos + 8,
        2 => {
            let (len, len_len) = read_varint(buf, pos);
            pos + len_len + len as usize
        }
        5 => pos + 4,
        _ => pos + 1,
    }
}

fn decode_tokens(vocab: &[String], token_ids: &[u32]) -> String {
    let mut pieces = Vec::new();
    let mut byte_buffer: Vec<u8> = Vec::new();
    let flush_bytes = |byte_buffer: &mut Vec<u8>, pieces: &mut Vec<String>| {
        if !byte_buffer.is_empty() {
            if let Ok(s) = String::from_utf8(byte_buffer.clone()) {
                pieces.push(s);
            }
            byte_buffer.clear();
        }
    };
    for &id in token_ids {
        if id == 0 || id == 3 { continue; }
        if let Some(piece) = vocab.get(id as usize) {
            if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16) {
                    byte_buffer.push(byte_val);
                    continue;
                }
            }
            flush_bytes(&mut byte_buffer, &mut pieces);
            pieces.push(piece.clone());
        }
    }
    flush_bytes(&mut byte_buffer, &mut pieces);
    let text = pieces.join("");
    text.replace('\u{2581}', " ").trim().to_string()
}

#[test]
fn test_e2e_wav_to_text() {
    pollster::block_on(async {
        let wav_path = std::path::Path::new("../../web/test-loona.wav");
        let mimi_weights_path = "../../models/mimi.safetensors";
        let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
        let tokenizer_path = std::path::Path::new("../../models/tokenizer.model");

        if !gguf_path.exists() {
            println!("Skipping: GGUF not found");
            return;
        }
        if !std::path::Path::new(mimi_weights_path).exists() {
            println!("Skipping: Mimi weights not found");
            return;
        }

        // === Step 1: Read WAV ===
        println!("=== Step 1: Reading WAV ===");
        let reader = hound::WavReader::open(wav_path).expect("Failed to open WAV");
        let spec = reader.spec();
        println!("WAV: {}Hz, {}ch, {} samples",
            spec.sample_rate, spec.channels, reader.len());

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1 << (bits - 1)) as f32;
                reader.into_samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect()
            }
        };
        let duration_s = samples.len() as f64 / spec.sample_rate as f64;
        println!("Audio: {:.2}s, {} samples at {}Hz", duration_s, samples.len(), spec.sample_rate);

        // === Step 2: Mimi encoder ===
        println!("\n=== Step 2: Mimi encoding (batch mode) ===");
        let mut mimi = mimi_wasm::MimiCodec::new(mimi_weights_path).await
            .expect("Failed to create Mimi codec");

        // Encode all audio at once (batch mode — avoids frame-boundary artifacts)
        let mimi_start = std::time::Instant::now();
        let flat_tokens = mimi.encode_all(&samples);
        let mimi_elapsed = mimi_start.elapsed().as_secs_f64();
        let num_codebooks = 32;
        let num_frames = flat_tokens.len() / num_codebooks;
        let mimi_rtf = mimi_elapsed / duration_s;
        println!("Mimi produced {} frames ({} tokens)", num_frames, flat_tokens.len());
        println!("Mimi RTF: {:.3}x ({:.3}s processing / {:.2}s audio)", mimi_rtf, mimi_elapsed, duration_s);

        // Reshape flat tokens to frames of 32
        let mut mimi_frames: Vec<Vec<u32>> = Vec::new();
        for f in 0..num_frames {
            let start = f * num_codebooks;
            let frame: Vec<u32> = flat_tokens[start..start + num_codebooks].to_vec();
            mimi_frames.push(frame);
        }

        // Print first few frames
        for (i, frame) in mimi_frames.iter().take(3).enumerate() {
            println!("  Frame {}: {:?}...", i, &frame[..8]);
        }

        // === Step 3: STT transformer ===
        println!("\n=== Step 3: STT inference ===");
        let vocab = load_sentencepiece_vocab(tokenizer_path);
        println!("Tokenizer: {} vocab entries", vocab.len());

        let device = device();
        let config = SttConfig::default();

        println!("Loading STT model...");
        let file_data = std::fs::read(gguf_path).unwrap();
        let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
        let parts = loader.load_deferred(&device, &config).unwrap();
        drop(loader);
        let model = parts.finalize(&device).unwrap();
        println!("STT model loaded.");

        let mut stream = SttStream::new(config.clone(), config.num_layers);
        let mut all_tokens: Vec<u32> = Vec::new();

        let stt_start = std::time::Instant::now();
        for (i, frame) in mimi_frames.iter().enumerate() {
            let token = stream.feed_frame(frame, &model).await;
            if let Some(t) = token {
                all_tokens.push(t);
            }
            if (i + 1) % 10 == 0 || i + 1 == num_frames {
                eprintln!("  [{}/{}]", i + 1, num_frames);
            }
        }

        // Flush
        let flush_tokens = stream.flush(&model).await;
        all_tokens.extend(&flush_tokens);
        let stt_elapsed = stt_start.elapsed().as_secs_f64();
        let stt_rtf = stt_elapsed / duration_s;
        let total_rtf = (mimi_elapsed + stt_elapsed) / duration_s;
        println!("STT RTF:   {:.3}x ({:.3}s processing / {:.2}s audio)", stt_rtf, stt_elapsed, duration_s);
        println!("Total RTF: {:.3}x (Mimi: {:.3}x, STT: {:.3}x)", total_rtf, mimi_rtf, stt_rtf);

        // === Step 4: Decode to text ===
        println!("\n=== Step 4: Decode tokens ===");
        let non_special: Vec<u32> = all_tokens.iter()
            .copied()
            .filter(|&t| t != 0 && t != 3)
            .collect();
        println!("Total tokens: {}", all_tokens.len());
        println!("Non-special tokens: {}", non_special.len());
        println!("Token IDs: {:?}", non_special);

        let transcript = decode_tokens(&vocab, &all_tokens);

        println!("\n========================================");
        println!("TRANSCRIPT: {}", transcript);
        println!("========================================");

        // Basic check: we should get SOME text out
        assert!(
            !transcript.is_empty(),
            "Transcript should not be empty"
        );
        println!("\n=== E2E WAV TEST PASSED ===");
    });
}

#[test]
fn test_e2e_wav_bria() {
    pollster::block_on(async {
        let wav_path = std::path::Path::new("../../web/test-bria.wav");
        let mimi_weights_path = "../../models/mimi.safetensors";
        let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
        let tokenizer_path = std::path::Path::new("../../models/tokenizer.model");

        if !gguf_path.exists() || !std::path::Path::new(mimi_weights_path).exists() {
            println!("Skipping: model files not found");
            return;
        }

        // === Read WAV ===
        println!("=== Reading WAV: test-bria.wav ===");
        let reader = hound::WavReader::open(wav_path).expect("Failed to open WAV");
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1 << (bits - 1)) as f32;
                reader.into_samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect()
            }
        };
        let duration_s = samples.len() as f64 / spec.sample_rate as f64;
        println!("Audio: {:.2}s, {} samples", duration_s, samples.len());

        // === Mimi encoding (batch mode) ===
        println!("Mimi encoding (batch)...");
        let mut mimi = mimi_wasm::MimiCodec::new(mimi_weights_path).await
            .expect("Failed to create Mimi codec");

        let mimi_start = std::time::Instant::now();
        let flat_tokens = mimi.encode_all(&samples);
        let mimi_elapsed = mimi_start.elapsed().as_secs_f64();
        let num_codebooks = 32;
        let num_frames = flat_tokens.len() / num_codebooks;
        let mimi_rtf = mimi_elapsed / duration_s;
        println!("Mimi: {} frames from {:.2}s audio", num_frames, duration_s);
        println!("Mimi RTF: {:.3}x ({:.3}s processing / {:.2}s audio)", mimi_rtf, mimi_elapsed, duration_s);

        let mut mimi_frames: Vec<Vec<u32>> = Vec::new();
        for f in 0..num_frames {
            let start = f * num_codebooks;
            mimi_frames.push(flat_tokens[start..start + num_codebooks].to_vec());
        }

        // === STT inference ===
        println!("Loading STT model...");
        let vocab = load_sentencepiece_vocab(tokenizer_path);
        let device = device();
        let config = SttConfig::default();

        let file_data = std::fs::read(gguf_path).unwrap();
        let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
        let parts = loader.load_deferred(&device, &config).unwrap();
        drop(loader);
        let model = parts.finalize(&device).unwrap();

        let mut stream = SttStream::new(config.clone(), config.num_layers);
        let mut all_tokens: Vec<u32> = Vec::new();

        println!("Running STT on {} frames...", num_frames);
        let stt_start = std::time::Instant::now();
        for (i, frame) in mimi_frames.iter().enumerate() {
            let token = stream.feed_frame(frame, &model).await;
            if let Some(t) = token {
                all_tokens.push(t);
            }
            if (i + 1) % 50 == 0 || i + 1 == num_frames {
                eprintln!("  [{}/{}]", i + 1, num_frames);
            }
        }

        let flush_tokens = stream.flush(&model).await;
        all_tokens.extend(&flush_tokens);
        let stt_elapsed = stt_start.elapsed().as_secs_f64();
        let stt_rtf = stt_elapsed / duration_s;
        let total_rtf = (mimi_elapsed + stt_elapsed) / duration_s;
        println!("STT RTF:   {:.3}x ({:.3}s processing / {:.2}s audio)", stt_rtf, stt_elapsed, duration_s);
        println!("Total RTF: {:.3}x (Mimi: {:.3}x, STT: {:.3}x)", total_rtf, mimi_rtf, stt_rtf);

        // === Decode ===
        let transcript = decode_tokens(&vocab, &all_tokens);

        println!("\n========================================");
        println!("TRANSCRIPT ({:.1}s audio):", duration_s);
        println!("{}", transcript);
        println!("========================================");

        assert!(!transcript.is_empty(), "Transcript should not be empty");
        println!("\n=== E2E WAV BRIA TEST PASSED ===");
    });
}

/// Streaming E2E test: simulates the browser pipeline.
///
/// Uses streaming Mimi (feed_audio with small chunks) + streaming STT,
/// exactly as the browser's SttEngine.feedAudio() does.
#[test]
fn test_e2e_streaming() {
    pollster::block_on(async {
        let wav_path = std::path::Path::new("../../web/test-loona.wav");
        let mimi_weights_path = "../../models/mimi.safetensors";
        let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
        let tokenizer_path = std::path::Path::new("../../models/tokenizer.model");

        if !gguf_path.exists() || !std::path::Path::new(mimi_weights_path).exists() {
            println!("Skipping: model files not found");
            return;
        }

        // === Read WAV ===
        println!("=== Streaming E2E: Reading WAV ===");
        let reader = hound::WavReader::open(wav_path).expect("Failed to open WAV");
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1 << (bits - 1)) as f32;
                reader.into_samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect()
            }
        };
        let duration_s = samples.len() as f64 / spec.sample_rate as f64;
        println!("Audio: {:.2}s, {} samples at {}Hz", duration_s, samples.len(), spec.sample_rate);

        // === Load Mimi (streaming mode) ===
        println!("\n=== Loading Mimi (streaming) ===");
        let mut mimi = mimi_wasm::MimiCodec::new(mimi_weights_path).await
            .expect("Failed to create Mimi codec");

        // === Load STT model ===
        println!("Loading STT model...");
        let vocab = load_sentencepiece_vocab(tokenizer_path);
        let device = device();
        let config = SttConfig::default();

        let file_data = std::fs::read(gguf_path).unwrap();
        let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
        let parts = loader.load_deferred(&device, &config).unwrap();
        drop(loader);
        let model = parts.finalize(&device).unwrap();
        let mut stream = SttStream::new(config.clone(), config.num_layers);

        // === Stream audio in browser-sized chunks ===
        // Browser AudioWorklet sends 1920 samples (80ms at 24kHz)
        // File upload uses 2400 samples (100ms at 24kHz)
        let chunk_size = 2400;
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut chunk_count = 0;
        let num_codebooks = config.num_codebooks;

        println!("\n=== Streaming audio in {}‐sample chunks ===", chunk_size);
        let start = std::time::Instant::now();

        for chunk_start in (0..samples.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(samples.len());
            let chunk = &samples[chunk_start..chunk_end];
            chunk_count += 1;

            // Streaming Mimi encode (exactly what browser does)
            let tokens = mimi.feed_audio(chunk);
            let mimi_frames = tokens.len() / num_codebooks;

            // Streaming STT forward for each Mimi frame
            for frame_start in (0..tokens.len()).step_by(num_codebooks) {
                if frame_start + num_codebooks > tokens.len() {
                    break;
                }
                let frame = &tokens[frame_start..frame_start + num_codebooks];
                if let Some(token) = stream.feed_frame(frame, &model).await {
                    all_tokens.push(token);
                }
            }

            if chunk_count <= 3 || mimi_frames > 0 {
                println!("  Chunk {}: {} samples → {} Mimi frames, {} text tokens total",
                    chunk_count, chunk.len(), mimi_frames, all_tokens.len());
            }
        }

        // === Flush remaining ===
        println!("\n=== Flushing delayed tokens ===");
        let flush_tokens = stream.flush(&model).await;
        println!("Flush produced {} tokens", flush_tokens.len());
        all_tokens.extend(&flush_tokens);

        let elapsed = start.elapsed().as_secs_f64();
        let rtf = elapsed / duration_s;
        println!("Total RTF: {:.3}x ({:.3}s processing / {:.2}s audio)", rtf, elapsed, duration_s);

        // === Decode ===
        let non_special: Vec<u32> = all_tokens.iter()
            .copied()
            .filter(|&t| t != 0 && t != 3)
            .collect();
        println!("\nTotal tokens: {} (non-special: {})", all_tokens.len(), non_special.len());
        println!("All token IDs: {:?}", all_tokens);
        println!("Non-special IDs: {:?}", non_special);

        let transcript = decode_tokens(&vocab, &all_tokens);
        println!("\n========================================");
        println!("STREAMING TRANSCRIPT: {}", transcript);
        println!("========================================");

        assert!(!transcript.is_empty(), "Streaming transcript should not be empty");
        println!("\n=== STREAMING E2E TEST PASSED ===");
    });
}

/// Streaming E2E test with longer audio (test-bria.wav, ~30s).
///
/// Uses streaming Mimi (feed_audio with 2400-sample chunks) + streaming STT,
/// exactly as the browser's SttEngine.feedAudio() does for file upload.
#[test]
fn test_e2e_streaming_bria() {
    pollster::block_on(async {
        let wav_path = std::path::Path::new("../../web/test-bria.wav");
        let mimi_weights_path = "../../models/mimi.safetensors";
        let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
        let tokenizer_path = std::path::Path::new("../../models/tokenizer.model");

        if !gguf_path.exists() || !std::path::Path::new(mimi_weights_path).exists() {
            println!("Skipping: model files not found");
            return;
        }

        // === Read WAV ===
        println!("=== Streaming E2E (bria): Reading WAV ===");
        let reader = hound::WavReader::open(wav_path).expect("Failed to open WAV");
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                let max_val = (1 << (bits - 1)) as f32;
                reader.into_samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect()
            }
        };
        let duration_s = samples.len() as f64 / spec.sample_rate as f64;
        println!("Audio: {:.2}s, {} samples at {}Hz", duration_s, samples.len(), spec.sample_rate);

        // === Load Mimi (streaming mode) ===
        println!("\n=== Loading Mimi (streaming) ===");
        let mut mimi = mimi_wasm::MimiCodec::new(mimi_weights_path).await
            .expect("Failed to create Mimi codec");

        // === Load STT model ===
        println!("Loading STT model...");
        let vocab = load_sentencepiece_vocab(tokenizer_path);
        let device = device();
        let config = SttConfig::default();

        let file_data = std::fs::read(gguf_path).unwrap();
        let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
        let parts = loader.load_deferred(&device, &config).unwrap();
        drop(loader);
        let model = parts.finalize(&device).unwrap();
        let mut stream = SttStream::new(config.clone(), config.num_layers);

        // === Stream audio in browser-sized chunks ===
        let chunk_size = 2400; // 100ms at 24kHz (file upload mode)
        let mut all_tokens: Vec<u32> = Vec::new();
        let mut chunk_count = 0;
        let mut total_frames = 0;
        let num_codebooks = config.num_codebooks;

        println!("\n=== Streaming audio in {}‐sample chunks ===", chunk_size);
        let start = std::time::Instant::now();

        for chunk_start in (0..samples.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(samples.len());
            let chunk = &samples[chunk_start..chunk_end];
            chunk_count += 1;

            // Streaming Mimi encode
            let tokens = mimi.feed_audio(chunk);
            // Streaming STT forward for each Mimi frame
            for frame_start in (0..tokens.len()).step_by(num_codebooks) {
                if frame_start + num_codebooks > tokens.len() {
                    break;
                }
                let frame = &tokens[frame_start..frame_start + num_codebooks];
                total_frames += 1;
                if let Some(token) = stream.feed_frame(frame, &model).await {
                    all_tokens.push(token);
                }

                // Progress every 50 frames
                if total_frames % 50 == 0 {
                    let partial = decode_tokens(&vocab, &all_tokens);
                    println!("  [frame {}, chunk {}] {} text tokens so far: \"{}\"",
                        total_frames, chunk_count, all_tokens.len(),
                        if partial.len() > 80 { format!("...{}", &partial[partial.len()-80..]) } else { partial });
                }
            }
        }

        // === Flush remaining ===
        println!("\n=== Flushing delayed tokens ===");
        let flush_tokens = stream.flush(&model).await;
        println!("Flush produced {} tokens", flush_tokens.len());
        all_tokens.extend(&flush_tokens);

        let elapsed = start.elapsed().as_secs_f64();
        let rtf = elapsed / duration_s;
        println!("Total: {} frames from {} chunks, RTF: {:.3}x ({:.3}s / {:.2}s audio)",
            total_frames, chunk_count, rtf, elapsed, duration_s);

        // === Decode ===
        let non_special: Vec<u32> = all_tokens.iter()
            .copied()
            .filter(|&t| t != 0 && t != 3)
            .collect();
        println!("\nTotal tokens: {} (non-special: {})", all_tokens.len(), non_special.len());

        let transcript = decode_tokens(&vocab, &all_tokens);
        println!("\n========================================");
        println!("STREAMING TRANSCRIPT (bria, {:.1}s):", duration_s);
        println!("{}", transcript);
        println!("========================================");

        assert!(!transcript.is_empty(), "Streaming transcript should not be empty");
        println!("\n=== STREAMING E2E BRIA TEST PASSED ===");
    });
}

#[test]
fn test_profiling_all() {
    pollster::block_on(async {
        let mimi_weights_path = "../../models/mimi.safetensors";
        let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
        let tokenizer_path = std::path::Path::new("../../models/tokenizer.model");

        if !gguf_path.exists() || !std::path::Path::new(mimi_weights_path).exists() {
            println!("Skipping: model files not found");
            return;
        }

        // Load models once
        let vocab = load_sentencepiece_vocab(tokenizer_path);
        let device = WgpuDevice::default();
        let config = SttConfig::default();
        let file_data = std::fs::read(gguf_path).unwrap();
        let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
        let parts = loader.load_deferred(&device, &config).unwrap();
        drop(loader);
        let model = parts.finalize(&device).unwrap();

        let test_files = vec![
            "test-loona.wav",
            "test-bria-3s.wav",
            "test-bria-5s.wav",
            "test-bria-10s.wav",
            "test-bria-10s-noisy.wav",
            "test-bria.wav",
            "test-crepes-fr-10s.wav",
            "test-crepes-fr.wav",
            "test-silence-60s.wav",
            "test-bria-120s.wav",
        ];

        println!("\n{:=<100}", "");
        println!("STREAMING PROFILING BENCHMARK");
        println!("{:=<100}\n", "");

        let mut results = Vec::new();

        for file_name in &test_files {
            let wav_path = std::path::Path::new("../../web").join(file_name);
            if !wav_path.exists() {
                println!("Skipping {}: not found", file_name);
                continue;
            }

            // Read WAV
            let reader = hound::WavReader::open(&wav_path).expect("Failed to open WAV");
            let spec = reader.spec();
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => {
                    reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
                }
                hound::SampleFormat::Int => {
                    let bits = spec.bits_per_sample;
                    let max_val = (1 << (bits - 1)) as f32;
                    reader.into_samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect()
                }
            };
            let duration_s = samples.len() as f64 / spec.sample_rate as f64;

            // Fresh Mimi + STT stream
            let mut mimi = mimi_wasm::MimiCodec::new(mimi_weights_path).await
                .expect("Failed to create Mimi codec");
            let mut stream = SttStream::new(config.clone(), config.num_layers);

            let chunk_size = 2400;
            let num_codebooks = config.num_codebooks;
            let mut all_tokens: Vec<u32> = Vec::new();
            let mut mimi_times: Vec<f64> = Vec::new();
            let mut stt_times: Vec<f64> = Vec::new();
            let mut total_frames = 0usize;

            let wall_start = std::time::Instant::now();

            for chunk_start in (0..samples.len()).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(samples.len());
                let chunk = &samples[chunk_start..chunk_end];

                // Time Mimi encode
                let mimi_start = std::time::Instant::now();
                let tokens = mimi.feed_audio(chunk);
                let mimi_elapsed = mimi_start.elapsed().as_secs_f64() * 1000.0;

                let mimi_frames = tokens.len() / num_codebooks;
                if mimi_frames > 0 {
                    let per_frame_mimi = mimi_elapsed / mimi_frames as f64;
                    for _ in 0..mimi_frames {
                        mimi_times.push(per_frame_mimi);
                    }
                }

                // Time STT forward for each frame
                for frame_start in (0..tokens.len()).step_by(num_codebooks) {
                    if frame_start + num_codebooks > tokens.len() { break; }
                    let frame = &tokens[frame_start..frame_start + num_codebooks];
                    total_frames += 1;

                    let stt_start = std::time::Instant::now();
                    if let Some(token) = stream.feed_frame(frame, &model).await {
                        all_tokens.push(token);
                    }
                    let stt_elapsed = stt_start.elapsed().as_secs_f64() * 1000.0;
                    stt_times.push(stt_elapsed);
                }
            }

            // Flush
            let flush_start = std::time::Instant::now();
            let flush_tokens = stream.flush(&model).await;
            let flush_ms = flush_start.elapsed().as_secs_f64() * 1000.0;
            all_tokens.extend(&flush_tokens);

            let wall_elapsed = wall_start.elapsed().as_secs_f64();
            let rtf = wall_elapsed / duration_s;

            // Percentile helper
            fn percentile(sorted: &[f64], p: f64) -> f64 {
                if sorted.is_empty() { return 0.0; }
                let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
                sorted[idx.min(sorted.len() - 1)]
            }

            let mut mimi_sorted = mimi_times.clone();
            mimi_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut stt_sorted = stt_times.clone();
            stt_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mimi_avg = if mimi_sorted.is_empty() { 0.0 } else { mimi_sorted.iter().sum::<f64>() / mimi_sorted.len() as f64 };
            let stt_avg = if stt_sorted.is_empty() { 0.0 } else { stt_sorted.iter().sum::<f64>() / stt_sorted.len() as f64 };

            let transcript = decode_tokens(&vocab, &all_tokens);

            println!("--- {} ({:.1}s, {} frames) ---", file_name, duration_s, total_frames);
            println!("  Mimi:  avg={:.1}ms  p50={:.1}ms  p95={:.1}ms  p99={:.1}ms",
                mimi_avg, percentile(&mimi_sorted, 50.0), percentile(&mimi_sorted, 95.0), percentile(&mimi_sorted, 99.0));
            println!("  STT:   avg={:.1}ms  p50={:.1}ms  p95={:.1}ms  p99={:.1}ms",
                stt_avg, percentile(&stt_sorted, 50.0), percentile(&stt_sorted, 95.0), percentile(&stt_sorted, 99.0));
            println!("  Total: avg={:.1}ms/frame  RTF={:.3}x  flush={:.1}ms",
                mimi_avg + stt_avg, rtf, flush_ms);
            println!("  Text:  \"{}\"", if transcript.len() > 80 { format!("{}...", &transcript[..80]) } else { transcript.clone() });
            println!();

            results.push((file_name.to_string(), duration_s, total_frames, mimi_avg, stt_avg, rtf));
        }

        // Summary table
        println!("\n{:=<100}", "");
        println!("SUMMARY TABLE");
        println!("{:=<100}", "");
        println!("{:<25} {:>6} {:>6} {:>10} {:>10} {:>10} {:>8}",
            "File", "Dur", "Frames", "Mimi ms", "STT ms", "Total ms", "RTF");
        println!("{:-<85}", "");
        for (name, dur, frames, mimi, stt, rtf) in &results {
            println!("{:<25} {:>5.1}s {:>6} {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>7.3}x",
                name, dur, frames, mimi, stt, mimi + stt, rtf);
        }
        println!("{:-<85}", "");
        println!("\nBudget: 80ms/frame for real-time (12.5Hz)\n");
    });
}

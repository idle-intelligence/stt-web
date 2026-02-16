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
        let flat_tokens = mimi.encode_all(&samples);
        let num_codebooks = 32;
        let num_frames = flat_tokens.len() / num_codebooks;
        println!("Mimi produced {} frames ({} tokens)", num_frames, flat_tokens.len());

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

        let flat_tokens = mimi.encode_all(&samples);
        let num_codebooks = 32;
        let num_frames = flat_tokens.len() / num_codebooks;
        println!("Mimi: {} frames from {:.2}s audio", num_frames, duration_s);

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

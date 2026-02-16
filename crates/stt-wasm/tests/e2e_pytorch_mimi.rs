//! Test: PyTorch Mimi tokens (from bria.wav) → STT → text.
//! Verifies the STT model works with real Mimi tokens from PyTorch.

use burn::backend::wgpu::WgpuDevice;
use stt_wasm::gguf::Q4ModelLoader;
use stt_wasm::stream::SttStream;
use stt_wasm::SttConfig;

fn device() -> WgpuDevice { WgpuDevice::default() }

#[derive(serde::Deserialize)]
struct MimiTokens { tokens: Vec<Vec<u32>> }

fn load_sentencepiece_vocab(path: &std::path::Path) -> Vec<String> {
    let data = std::fs::read(path).unwrap();
    let mut pieces = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let (tag, tl) = rv(&data, pos); pos += tl;
        if (tag >> 3) == 1 && (tag & 7) == 2 {
            let (len, ll) = rv(&data, pos); pos += ll;
            pieces.push(parse_sp(&data[pos..pos + len as usize]));
            pos += len as usize;
        } else { pos = sf(&data, pos, (tag & 7) as u8); }
    }
    pieces
}
fn parse_sp(buf: &[u8]) -> String {
    let mut pos = 0; let mut piece = String::new();
    while pos < buf.len() {
        let (tag, tl) = rv(buf, pos); pos += tl;
        if (tag >> 3) == 1 && (tag & 7) == 2 {
            let (len, ll) = rv(buf, pos); pos += ll;
            piece = String::from_utf8_lossy(&buf[pos..pos + len as usize]).to_string();
            pos += len as usize;
        } else { pos = sf(buf, pos, (tag & 7) as u8); }
    }
    piece
}
fn rv(buf: &[u8], s: usize) -> (u64, usize) {
    let (mut v, mut sh, mut p) = (0u64, 0, s);
    while p < buf.len() { let b = buf[p]; p += 1; v |= ((b & 0x7f) as u64) << sh; sh += 7; if b & 0x80 == 0 { break; } }
    (v, p - s)
}
fn sf(buf: &[u8], pos: usize, wt: u8) -> usize {
    match wt { 0 => { let mut p = pos; while p < buf.len() && buf[p] & 0x80 != 0 { p += 1; } p + 1 }
        1 => pos + 8, 2 => { let (l, ll) = rv(buf, pos); pos + ll + l as usize } 5 => pos + 4, _ => pos + 1 }
}
fn decode_tokens(vocab: &[String], ids: &[u32]) -> String {
    let mut pieces = Vec::new(); let mut bb: Vec<u8> = Vec::new();
    for &id in ids {
        if id == 0 || id == 3 { continue; }
        if let Some(p) = vocab.get(id as usize) {
            if p.starts_with("<0x") && p.ends_with('>') && p.len() == 6 {
                if let Ok(b) = u8::from_str_radix(&p[3..5], 16) { bb.push(b); continue; }
            }
            if !bb.is_empty() { if let Ok(s) = String::from_utf8(bb.clone()) { pieces.push(s); } bb.clear(); }
            pieces.push(p.clone());
        }
    }
    if !bb.is_empty() { if let Ok(s) = String::from_utf8(bb) { pieces.push(s); } }
    pieces.join("").replace('\u{2581}', " ").trim().to_string()
}

#[test]
fn test_pytorch_mimi_tokens_to_text() {
    pollster::block_on(async {
        let mimi_path = std::path::Path::new("/tmp/mimi_bria_ref.json");
        let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
        let tok_path = std::path::Path::new("../../models/tokenizer.model");

        if !gguf_path.exists() || !mimi_path.exists() {
            println!("Skipping: model or reference not found");
            return;
        }

        let mimi_data: MimiTokens = serde_json::from_str(
            &std::fs::read_to_string(mimi_path).unwrap()
        ).unwrap();
        let num_frames = mimi_data.tokens.len();
        println!("PyTorch Mimi: {} frames from bria.wav", num_frames);

        let vocab = load_sentencepiece_vocab(tok_path);
        let device = device();
        let config = SttConfig::default();

        println!("Loading STT model...");
        let file_data = std::fs::read(gguf_path).unwrap();
        let mut loader = Q4ModelLoader::from_shards(vec![file_data]).unwrap();
        let parts = loader.load_deferred(&device, &config).unwrap();
        drop(loader);
        let model = parts.finalize(&device).unwrap();

        let mut stream = SttStream::new(config.clone(), config.num_layers);
        let mut all_tokens: Vec<u32> = Vec::new();

        for (i, frame) in mimi_data.tokens.iter().enumerate() {
            let token = stream.feed_frame(frame, &model).await;
            if let Some(t) = token { all_tokens.push(t); }
            if (i + 1) % 50 == 0 || i + 1 == num_frames {
                eprintln!("  [{}/{}]", i + 1, num_frames);
            }
        }

        let flush = stream.flush(&model).await;
        all_tokens.extend(&flush);

        let transcript = decode_tokens(&vocab, &all_tokens);
        let non_special: Vec<u32> = all_tokens.iter().copied().filter(|&t| t != 0 && t != 3).collect();

        println!("\n=== RESULT ===");
        println!("Total tokens: {}, non-special: {}", all_tokens.len(), non_special.len());
        println!("\nTRANSCRIPT:");
        println!("{}", transcript);

        assert!(!transcript.is_empty(), "Should produce text from PyTorch Mimi tokens");
    });
}

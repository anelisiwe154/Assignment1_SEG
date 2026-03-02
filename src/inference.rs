
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::prelude::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::{Int, Tensor};

use tokenizers::Tokenizer;

use crate::data::dataset::QaDataset;
use crate::data::loader::load_docx;
use crate::model::qa_model::QaModel;

type BI = Wgpu; 

fn calendar_doc_path(year: &str) -> &'static str {
    match year {
        "2024" => "calendar2024.docx",
        "2025" => "calendar2025.docx",
        "2026" => "calendar2026.docx",
        _ => "calendar2024.docx",
    }
}

fn checkpoint_path(year: &str) -> String {
    format!("checkpoints/model_{}", year)
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .replace('’', "'")
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn extract_key(question: &str) -> String {
    let q = normalize(question);
    let q = q.strip_prefix("when is ").unwrap_or(&q);
    q.trim_end_matches('?').trim().to_string()
}

fn retrieve_answer_from_dataset(question: &str, dataset: &QaDataset) -> Option<String> {
    let key = extract_key(question);
    let key_tokens: Vec<&str> = key.split_whitespace().collect();

    let mut best: Option<(usize, String)> = None;

    for s in dataset.samples.iter() {
        let sq = normalize(&s.question);
        let sa = normalize(&s.answer);

        let mut score = 0usize;

        // strong signals
        if sq.contains(&key) {
            score += 20;
        }
        if sa.contains(&key) {
            score += 6;
        }

        // overlap signals
        for t in key_tokens.iter().copied() {
            if sq.contains(t) {
                score += 3;
            }
            if sa.contains(t) {
                score += 1;
            }
        }

        if score > 0 {
            match &best {
                None => best = Some((score, s.answer.clone())),
                Some((best_score, _)) if score > *best_score => best = Some((score, s.answer.clone())),
                _ => {}
            }
        }
    }

    best.map(|(_, ans)| ans)
}

fn clean_wordpiece(tokens: &[String]) -> String {
    let mut out = String::new();
    for tok in tokens {
        if tok == "[CLS]" || tok == "[SEP]" || tok == "[PAD]" {
            continue;
        }
        if tok.starts_with("##") {
            out.push_str(tok.trim_start_matches("##"));
        } else {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(tok);
        }
    }
    out
}

fn pad_trunc_i64(mut v: Vec<i64>, max_len: usize) -> Vec<i64> {
    if v.len() > max_len {
        v.truncate(max_len);
    }
    v.resize(max_len, 0);
    v
}

fn build_tensors(ids: &[i64], device: &WgpuDevice) -> (Tensor<BI, 2, Int>, Tensor<BI, 2, Int>) {
    let max_len = ids.len();
    let attention: Vec<i64> = ids.iter().map(|&x| if x == 0 { 0 } else { 1 }).collect();

    let input_ids = Tensor::<BI, 1, Int>::from_data(ids, device).reshape([1, max_len]);
    let attention_mask = Tensor::<BI, 1, Int>::from_data(attention.as_slice(), device).reshape([1, max_len]);

    (input_ids, attention_mask)
}

fn best_span_after_sep(
    start_logits: &Tensor<BI, 2>,
    end_logits: &Tensor<BI, 2>,
    sep_idx: usize,
    max_len: usize,
) -> (usize, usize, f32) {
    // Restrict answer to context side only: (sep_idx+1 .. max_len-2)
    let start_data = start_logits.to_data();
    let end_data = end_logits.to_data();

    let s = start_data.as_slice::<f32>().expect("start logits f32");
    let e = end_data.as_slice::<f32>().expect("end logits f32");

    let low = (sep_idx + 1).min(max_len.saturating_sub(2));
    let high = max_len.saturating_sub(2);

    let mut best_score = f32::NEG_INFINITY;
    let mut best_i = low;
    let mut best_j = low;

    // small max span helps avoid junk spans
    for i in low..=high {
        for j in i..=(i + 16).min(high) {
            let score = s[i] + e[j];
            if score > best_score {
                best_score = score;
                best_i = i;
                best_j = j;
            }
        }
    }

    (best_i, best_j, best_score)
}

/// Usage:
///   cargo run -- ask 2024 "When is New Year's Day?"
///   cargo run -- ask 2025 "When is Good Friday?"
pub fn run_inference(year: &str, question: &str) {
    println!("Question: {}", question);

    // Load chosen calendar
    let doc_path = calendar_doc_path(year);
    let context = load_docx(doc_path).expect("Failed to load calendar docx (check filename/path)");

    
    let dataset = QaDataset::from_text_with_year(context.clone(), year);

   
    if let Some(ans) = retrieve_answer_from_dataset(question, &dataset) {
        println!("Answer: {}", ans);
        return;
    }


    let device = WgpuDevice::default();

    let tokenizer = Tokenizer::from_file("assets/bert-base-uncased-tokenizer.json")
        .expect("Tokenizer missing at assets/bert-base-uncased-tokenizer.json");

    let ckpt = checkpoint_path(year);
    let recorder = CompactRecorder::new();
    let record = recorder
        .load(ckpt.clone().into(), &device)
        .expect("Failed to load checkpoint. Train first: cargo run -- train 2024 (or 2025)");

    // MUST match training architecture
    let vocab_size: usize = 30_522;
    let max_len: usize = 128;

    let model = QaModel::<BI>::new(
        &device,
        vocab_size,
        max_len,
        256, 
        6,   
        8,   
        512, 
        0.1, 
    )
    .load_record(record);

    // Chunk by chars (this was the “working” style, avoids re-decode weirdness)
    let chunk_chars: usize = 1200;
    let stride: usize = 800;

    let chars: Vec<char> = context.chars().collect();
    let mut start = 0usize;

    let mut best_answer = String::new();
    let mut best_score = f32::NEG_INFINITY;

    while start < chars.len() {
        let end = (start + chunk_chars).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();

        let input_text = format!("{} [SEP] {}", question, chunk);
        let enc = tokenizer.encode(input_text, true).unwrap();

        let ids_u32 = enc.get_ids();
        let ids_i64: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();
        let ids_i64 = pad_trunc_i64(ids_i64, max_len);

        let (input_ids, attention_mask) = build_tensors(&ids_i64, &device);
        let (start_logits, end_logits) = model.forward(input_ids, attention_mask);

        let toks = enc.get_tokens();
        let sep_idx = toks.iter().position(|t| t == "[SEP]").unwrap_or(0).min(max_len - 1);

        let (si, ei, score) = best_span_after_sep(&start_logits, &end_logits, sep_idx, max_len);

        if toks.is_empty() {
            if end == chars.len() { break; }
            start = start.saturating_add(stride);
            continue;
        }

        let s_idx = si.min(toks.len() - 1);
        let e_idx = ei.min(toks.len() - 1);

        let answer_tokens = toks[s_idx..=e_idx].iter().cloned().collect::<Vec<_>>();
        let answer = clean_wordpiece(&answer_tokens);

        if score > best_score && !answer.trim().is_empty() && answer != "-" {
            best_score = score;
            best_answer = answer;
        }

        if end == chars.len() {
            break;
        }
        start = start.saturating_add(stride);
    }

    if best_answer.trim().is_empty() {
        println!("Answer: (no confident answer found)");
    } else {
        println!("Answer: {}", best_answer);
    }
}
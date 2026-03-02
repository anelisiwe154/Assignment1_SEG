use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use tokenizers::{EncodeInput, Tokenizer};

use crate::data::dataset::QaSample;

#[derive(Clone, Debug)]
pub struct QaBatch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,       
    pub attention_mask: Tensor<B, 2, Int>,  
    pub start_labels: Tensor<B, 1, Int>,    
    pub end_labels: Tensor<B, 1, Int>,      
}

#[derive(Clone)]
pub struct QaBatcher<B: Backend> {
    pub device: B::Device,
    pub tokenizer: Tokenizer,
    pub max_len: usize,
}

impl<B: Backend> QaBatcher<B> {
    pub fn new(device: B::Device, tokenizer: Tokenizer, max_len: usize) -> Self {
        Self {
            device,
            tokenizer,
            max_len,
        }
    }
}

fn find_subsequence(haystack: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    'outer: for i in 0..=(haystack.len() - needle.len()) {
        for j in 0..needle.len() {
            if haystack[i + j] != needle[j] {
                continue 'outer;
            }
        }
        return Some(i);
    }
    None
}

impl<B: Backend> Batcher<B, QaSample, QaBatch<B>> for QaBatcher<B> {
    fn batch(&self, samples: Vec<QaSample>, _device: &B::Device) -> QaBatch<B> {
        let batch_size = samples.len();
        let max_len = self.max_len;

        let mut input_ids_all: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut attention_all: Vec<i64> = Vec::with_capacity(batch_size * max_len);

        let mut start_labels_vec: Vec<i64> = Vec::with_capacity(batch_size);
        let mut end_labels_vec: Vec<i64> = Vec::with_capacity(batch_size);

        for sample in samples.iter() {
            //  REAL paired encoding => CLS question SEP context SEP
            let enc = self
                .tokenizer
                .encode(
                    EncodeInput::Dual(sample.question.clone().into(), sample.context.clone().into()),
                    true,
                )
                .unwrap();

            // ids (un-padded)
            let ids_u32 = enc.get_ids().to_vec();
            let mut ids: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();

            // pad/truncate
            if ids.len() > max_len {
                ids.truncate(max_len);
            }
            let real_len = ids.len();
            ids.resize(max_len, 0);

            let mut attention: Vec<i64> = vec![1; real_len];
            attention.resize(max_len, 0);

            input_ids_all.extend_from_slice(&ids);
            attention_all.extend_from_slice(&attention);

            // ---- labels: locate answer tokens inside the combined token ids ----
            let ans_enc = self.tokenizer.encode(sample.answer.clone(), false).unwrap();
            let ans_ids: Vec<u32> = ans_enc.get_ids().to_vec();

            // search inside original (un-padded) combined ids, but only up to max_len
            let combined_ids_u32: Vec<u32> = ids_u32.into_iter().take(max_len).collect();

            if let Some(pos) = find_subsequence(&combined_ids_u32, &ans_ids) {
                let start = pos as i64;
                let end = (pos + ans_ids.len().saturating_sub(1)) as i64;

                start_labels_vec.push(start.min((max_len - 1) as i64));
                end_labels_vec.push(end.min((max_len - 1) as i64));
            } else {
                // fallback: first token after real [SEP]
                let toks = enc.get_tokens();
                let sep_pos = toks.iter().position(|t| t == "[SEP]").unwrap_or(0);
                let fb = ((sep_pos + 1).min(max_len - 1)) as i64;
                start_labels_vec.push(fb);
                end_labels_vec.push(fb);
            }
        }

        let input_ids =
            Tensor::<B, 1, Int>::from_data(input_ids_all.as_slice(), &self.device).reshape([
                batch_size,
                max_len,
            ]);

        let attention_mask =
            Tensor::<B, 1, Int>::from_data(attention_all.as_slice(), &self.device).reshape([
                batch_size,
                max_len,
            ]);

        let start_labels =
            Tensor::<B, 1, Int>::from_data(start_labels_vec.as_slice(), &self.device)
                .reshape([batch_size]);

        let end_labels =
            Tensor::<B, 1, Int>::from_data(end_labels_vec.as_slice(), &self.device)
                .reshape([batch_size]);

        QaBatch {
            input_ids,
            attention_mask,
            start_labels,
            end_labels,
        }
    }
}

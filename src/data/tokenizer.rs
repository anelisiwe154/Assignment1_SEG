use std::path::PathBuf;
use tokenizers::Tokenizer;

pub fn create_tokenizer() -> Tokenizer {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("assets");
    path.push("bert-base-uncased-tokenizer.json");

    Tokenizer::from_file(path).expect("Failed to load tokenizer JSON")
}
#[allow(dead_code)]
pub fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<usize> {
    let encoding = tokenizer.encode(text, true).unwrap();
    encoding.get_ids().iter().map(|&id| id as usize).collect()
}







#![recursion_limit = "256"]

mod inference;
mod data;
mod model;
mod training;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;

use data::batcher::QaBatcher;
use data::dataset::QaDataset;
use data::loader::load_docx;
use data::tokenizer::create_tokenizer;

use model::qa_model::QaModel;
use training::trainer::{train, TrainConfig};

type B = Autodiff<Wgpu>;

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

fn read_line(prompt: &str) -> String {
    use std::io::{self, Write};
    print!("{}", prompt);
    let _ = io::stdout().flush();
    let mut s = String::new();
    io::stdin().read_line(&mut s).expect("Failed to read input");
    s.trim().to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

   

    if args.len() >= 3 && args[1] == "ask" {
        let year = &args[2];
        let question = if args.len() >= 4 {
            args[3..].join(" ")
        } else {
            read_line("Enter your question: ")
        };
        inference::run_inference(year, &question);
        return;
    }

    if args.len() >= 3 && args[1] == "train" {
        let year = &args[2];
        train_for_year(year);
        return;
    }

    // Interactive mode
    println!("SEG Q&A ML Pipeline Initialized!");

    let mode = read_line("Choose mode (train/ask): ").to_lowercase();
    let year = read_line("Which calendar year? (2024/2025/2026carg): ");

    if mode == "train" {
        train_for_year(&year);
    } else {
        let question = read_line("Enter your question: ");
        inference::run_inference(&year, &question);
    }
}

fn train_for_year(year: &str) {
    println!("Training for year: {}", year);

    // 1) Load the right calendar doc
    let doc_path = calendar_doc_path(year);
    let text = load_docx(doc_path).expect("Failed to load document (check .docx path)");

    // 2) Preview
    println!("--- DOC PREVIEW ---");
    println!("{}", text.chars().take(250).collect::<String>());
    println!("-------------------");

    // 3) Build dataset
    let dataset = QaDataset::from_text_with_year(text, year);
    println!("Dataset size: {}", dataset.len());

    // 4) Backend + device
    let device = WgpuDevice::default();

    // 5) Tokenizer + batcher
    let tokenizer = create_tokenizer();
    let batcher = QaBatcher::<B>::new(device.clone(), tokenizer, 128);

    // 6) DataLoader (Arc<dyn DataLoader<...>>)
    let dataloader = DataLoaderBuilder::<B, _, _>::new(batcher)
        .batch_size(8)
        .shuffle(42)
        .num_workers(0)
        .build(dataset);

    // 7) Model
    let vocab_size: usize = 30_522;
    let model = QaModel::<B>::new(
        &device,
        vocab_size,
        128, 
        256, 
        6,   
        8,   
        512, 
        0.1, 
    );

    // 8) Train + save checkpoint for this year
    let config = TrainConfig::new();
    std::fs::create_dir_all("checkpoints").ok();
    let ckpt = checkpoint_path(year);

    let _trained = train(&device, model, dataloader, config, &ckpt);

    println!(" Training completed. Saved checkpoint -> {}", ckpt);
}
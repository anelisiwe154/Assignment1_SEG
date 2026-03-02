use burn::config::Config;
use burn::data::dataloader::DataLoader;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::ToElement;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;

use crate::data::batcher::QaBatch;
use crate::model::qa_model::QaModel;

#[derive(Config, Debug)]
pub struct TrainConfig {
    #[config(default = 3)]
    pub epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 2e-4)]
    pub lr: f64,
}

pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    mut model: QaModel<B>,
    dataloader: std::sync::Arc<dyn DataLoader<B, QaBatch<B>>>,
    config: TrainConfig,
    checkpoint_path: &str,
) -> QaModel<B> {
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut optim = AdamConfig::new().init();

    for epoch in 1..=config.epochs {
        let mut total_loss = 0.0f64;
        let mut steps = 0usize;

        // iterator created per epoch
        for batch in dataloader.iter() {
            let (start_logits, end_logits) =
                model.forward(batch.input_ids, batch.attention_mask);

            let loss_start = loss_fn.forward(start_logits, batch.start_labels);
            let loss_end = loss_fn.forward(end_logits, batch.end_labels);
            let loss = loss_start + loss_end;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.lr, model, grads);

            total_loss += loss.into_scalar().to_f64();
            steps += 1;
        }

        if steps > 0 {
            println!(
                "Epoch {}/{} - avg loss: {:.6}",
                epoch,
                config.epochs,
                total_loss / steps as f64
            );
        } else {
            println!("Epoch {}/{} - no batches", epoch, config.epochs);
        }

        let recorder = CompactRecorder::new();
        recorder
            .record(model.clone().into_record(), checkpoint_path.into())
            .expect("Failed to save checkpoint");
        println!("Saved checkpoint -> {}", checkpoint_path);
    }

    model
}
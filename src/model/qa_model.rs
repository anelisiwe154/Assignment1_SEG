
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::model::transformer::TransformerEncoderBlock;

#[derive(Module, Debug)]
pub struct QaModel<B: Backend> {
    tok_embed: Embedding<B>,
    pos_embed: Embedding<B>,
    ln: LayerNorm<B>,
    dropout: Dropout,
    layers: Vec<TransformerEncoderBlock<B>>,
    start_head: Linear<B>,
    end_head: Linear<B>,
}

impl<B: Backend> QaModel<B> {
    pub fn new(
        device: &B::Device,
        vocab_size: usize,
        max_len: usize,
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: f64,
    ) -> Self {
        let tok_embed = EmbeddingConfig::new(vocab_size, d_model).init(device);
        let pos_embed = EmbeddingConfig::new(max_len, d_model).init(device);
        let ln = LayerNormConfig::new(d_model).init(device);
        let dropout_layer = DropoutConfig::new(dropout).init();

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TransformerEncoderBlock::new(device, d_model, num_heads, d_ff, dropout));
        }

        let start_head = LinearConfig::new(d_model, 1).init(device);
        let end_head = LinearConfig::new(d_model, 1).init(device);

        Self {
            tok_embed,
            pos_embed,
            ln,
            dropout: dropout_layer,
            layers,
            start_head,
            end_head,
        }
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, seq] = input_ids.dims();

        // positions [batch, seq]
        let pos = Tensor::<B, 1, Int>::arange(0..seq as i64, &input_ids.device())
            .reshape([1, seq])
            .repeat(&[batch, 1]);

        let mut x = self.tok_embed.forward(input_ids) + self.pos_embed.forward(pos);
        x = self.ln.forward(x);
        x = self.dropout.forward(x);

        for layer in self.layers.iter() {
            x = layer.forward(x, attention_mask.clone());
        }

        let start_logits = self.start_head.forward(x.clone()).reshape([batch, seq]);
        let end_logits = self.end_head.forward(x).reshape([batch, seq]);

        (start_logits, end_logits)
    }
}



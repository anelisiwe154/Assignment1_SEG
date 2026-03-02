
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::{activation, backend::Backend, Int, Tensor};

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(device: &B::Device, d_model: usize, d_ff: usize, dropout: f64) -> Self {
        let lin1 = LinearConfig::new(d_model, d_ff).init(device);
        let lin2 = LinearConfig::new(d_ff, d_model).init(device);
        let dropout = DropoutConfig::new(dropout).init();
        Self { lin1, lin2, dropout }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.lin1.forward(x);
        let h = h.tanh(); // simple nonlinearity
        let h = self.dropout.forward(h);
        self.lin2.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    w_q: Linear<B>,
    w_k: Linear<B>,
    w_v: Linear<B>,
    w_o: Linear<B>,
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(device: &B::Device, d_model: usize, num_heads: usize, dropout: f64) -> Self {
        assert!(d_model % num_heads == 0, "d_model must be divisible by num_heads");
        let head_dim = d_model / num_heads;

        Self {
            w_q: LinearConfig::new(d_model, d_model).init(device),
            w_k: LinearConfig::new(d_model, d_model).init(device),
            w_v: LinearConfig::new(d_model, d_model).init(device),
            w_o: LinearConfig::new(d_model, d_model).init(device),
            num_heads,
            head_dim,
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    // attention_mask is Int now (so .float() works)
    pub fn forward(&self, x: Tensor<B, 3>, attention_mask: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq, d_model] = x.dims();

        let q = self.w_q.forward(x.clone());
        let k = self.w_k.forward(x.clone());
        let v = self.w_v.forward(x);

        // [batch, heads, seq, head_dim]
        let q = q.reshape([batch, seq, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let k = k.reshape([batch, seq, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let v = v.reshape([batch, seq, self.num_heads, self.head_dim]).swap_dims(1, 2);

        // scores [batch, heads, seq, seq]
        let kt = k.swap_dims(2, 3);
        let mut scores = q.matmul(kt) / (self.head_dim as f32).sqrt();

        // mask [batch, 1, 1, seq] float
        let mask = attention_mask.reshape([batch, 1, 1, seq]).float();

        // push pads down
        scores = scores + (mask - 1.0) * 1e9;

        let attn = activation::softmax(scores, 3);
        let attn = self.dropout.forward(attn);

        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2).reshape([batch, seq, d_model]);

        self.w_o.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct TransformerEncoderBlock<B: Backend> {
    attn: SelfAttention<B>,
    ff: FeedForward<B>,
    ln1: LayerNorm<B>,
    ln2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerEncoderBlock<B> {
    pub fn new(device: &B::Device, d_model: usize, num_heads: usize, d_ff: usize, dropout: f64) -> Self {
        Self {
            attn: SelfAttention::new(device, d_model, num_heads, dropout),
            ff: FeedForward::new(device, d_model, d_ff, dropout),
            ln1: LayerNormConfig::new(d_model).init(device),
            ln2: LayerNormConfig::new(d_model).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    // attention_mask is Int now
    pub fn forward(&self, x: Tensor<B, 3>, attention_mask: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let h = self.attn.forward(x.clone(), attention_mask.clone());
        let x = self.ln1.forward(x + self.dropout.forward(h));

        let h2 = self.ff.forward(x.clone());
        self.ln2.forward(x + self.dropout.forward(h2))
    }
}



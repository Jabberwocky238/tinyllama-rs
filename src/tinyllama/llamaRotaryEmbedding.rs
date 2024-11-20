use tch::nn;
use tch::nn::Init;
use tch::nn::LinearConfig;
use tch::nn::ModuleT;
use tch::nn::VarStore;
use tch::vision::resnet::resnet18;
use tch::Device;
use tch::IndexOp;
use tch::Kind;
use tch::Result;
use tch::Tensor;

#[derive(Debug)]
pub struct LlamaRotaryEmbedding {
    dim: i64,
    max_position_embeddings: i64,
    base: f64,
    scaling_factor: f64,

    inv_freq: Tensor,
    _cos_cached: Tensor,
    _sin_cached: Tensor,
}

unsafe impl Send for LlamaRotaryEmbedding {}

#[macro_export]
macro_rules! llamaRotaryEmbeddingM {
    { dim: $dim:expr, $($param:ident: $value:expr),* $(,)? } => {
        llamaRotaryEmbeddingM! { $dim, $($param: $value),* }
    };
    { $dim:expr, $($param:ident: $value:expr),* $(,)? } => {
        // 使用 tt 循环处理可选参数
        {
            // 构建参数列表
            use std::collections::HashMap;
            let mut params: HashMap<&str, Box<dyn std::any::Any>> = HashMap::new();

            params.insert("max_position_embeddings", Box::new(2048));
            params.insert("base", Box::new(10000.0));
            params.insert("scaling_factor", Box::new(1.0));
            
            $(
                params.insert(stringify!($param), Box::new($value));
            )*

            // 展开参数并调用函数
            llamaRotaryEmbedding(
                $dim,
                *params["max_position_embeddings"].downcast_ref::<i64>().unwrap(),
                *params["base"].downcast_ref::<f64>().unwrap(),
                *params["scaling_factor"].downcast_ref::<f64>().unwrap(),
            )
        }
    };
}

pub fn llamaRotaryEmbedding(
    dim: i64,
    max_position_embeddings: i64, // 2048
    base: f64, // 10000.0
    scaling_factor: f64, // 1.0
) -> Result<LlamaRotaryEmbedding> {
    // inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
    // 1: arange = torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
    let arange = Tensor::arange_start_step(0, dim, 2, (Kind::Float, Device::Cpu));
    // 2: inv_freq = 1.0 / (self.base ** (arange / self.dim))
    let arange_div_dim = arange.f_div_scalar(dim as f64)?;
    let base_pow = Tensor::from(base).pow(&arange_div_dim);
    let inv_freq = Tensor::from(1.0).f_div(&base_pow)?;

    // t = torch.arange(max_position_embeddings, device=device, dtype=torch.int64).type_as(self.inv_freq)
    // t = t / self.scaling_factor
    // freqs = torch.outer(t, self.inv_freq)
    let t = Tensor::arange(max_position_embeddings, (Kind::Int64, Device::Cpu));
    let t = t.f_div_scalar(scaling_factor)?;
    let freqs = t.unsqueeze(1).matmul(&inv_freq);

    // Different from paper, but it uses a different permutation in order to obtain the same calculation
    // emb = torch.cat((freqs, freqs), dim=-1)
    let emb = Tensor::cat(&[&freqs, &freqs], -1);
    // self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
    // self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)
    let _cos_cached = emb.cos();
    let _sin_cached = emb.sin();

    let llamaRotaryEmbedding = LlamaRotaryEmbedding {
        scaling_factor,
        dim,
        max_position_embeddings,
        base,
        inv_freq,
        _cos_cached,
        _sin_cached,
    };
    Ok(llamaRotaryEmbedding)
}

impl LlamaRotaryEmbedding {
    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> (Tensor, Tensor) {
        // x: [bs, num_attention_heads, seq_len, head_size]
        // inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        let inv_freq_expanded = self
            .inv_freq
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(&[position_ids.size()[0], -1, 1], false);
        // position_ids_expanded = position_ids[:, None, :].float()
        let position_ids_expanded = position_ids.unsqueeze(1).to_kind(Kind::Float);
        // freqs = (inv_freq_expanded.float() @ position_ids_expanded.float())
        let freqs = inv_freq_expanded.matmul(&position_ids_expanded);
        // emb = torch.cat((freqs, freqs), dim=-1)
        let emb = Tensor::cat(&[&freqs, &freqs], -1);
        // cos = emb.cos()
        let cos = emb.cos();
        // sin = emb.sin()
        let sin = emb.sin();
        (cos.to_kind(x.kind()), sin.to_kind(x.kind()))
    }
}


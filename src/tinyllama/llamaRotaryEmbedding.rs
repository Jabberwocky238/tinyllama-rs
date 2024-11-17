use tch::nn::ModuleT;
use tch::nn;
use tch::nn::Init;
use tch::nn::LinearConfig;
use tch::nn::VarStore;
use tch::vision::resnet::resnet18;
use tch::Device;
use tch::IndexOp;
use tch::Kind;
use tch::Tensor;
use tch::Result;

#[derive(Debug)]
struct LlamaRotaryEmbedding {
    /// scaling_factor
    sf: f64, 
    dim: i64,
    /// max_position_embeddings 
    mpe: i64, 
    base: i64,

    inv_freq: Tensor,
    _cos_cached: Tensor,
    _sin_cached: Tensor,
}

unsafe impl Send for LlamaRotaryEmbedding {}

fn llamaRotaryEmbedding(dim: i64, max_position_embeddings: i64, base: i64, scaling_factor: f64) -> Result<LlamaRotaryEmbedding> {
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
        sf: scaling_factor,
        dim,
        mpe: max_position_embeddings,
        base,
        inv_freq,
        _cos_cached,
        _sin_cached,
    };
    Ok(llamaRotaryEmbedding)
}

impl ModuleT for LlamaRotaryEmbedding {
    fn forward_t(&self, xs: &tch::Tensor, train: bool) -> tch::Tensor {
        
    }
}

//     @torch.no_grad()
//     def forward(self, x, position_ids):
//         # x: [bs, num_attention_heads, seq_len, head_size]
//         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
//         position_ids_expanded = position_ids[:, None, :].float()
//         # Force float32 since bfloat16 loses precision on long contexts
//         # See https://github.com/huggingface/transformers/pull/29285
//         device_type = x.device.type
//         device_type = device_type if isinstance(device_type, str) else "cpu"
//         with torch.autocast(device_type=device_type, enabled=False):
//             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
//             emb = torch.cat((freqs, freqs), dim=-1)
//             cos = emb.cos()
//             sin = emb.sin()
//         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

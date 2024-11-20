use super::llamaConfig::LlamaConfig;
use super::llamaRotaryEmbedding::llamaRotaryEmbedding;
use super::llamaRotaryEmbedding::LlamaRotaryEmbedding;
use tch::nn;
use tch::nn::Module;
use tch::Device;
use tch::IndexOp;
use tch::Result;
use tch::Tensor;

use super::utils::linear;
use crate::llamaRotaryEmbeddingM;

#[derive(Debug)]
struct LlamaSdpaAttentionConfig {
    attention_dropout: f64,
    hidden_size: i64,
    num_heads: i64,
    head_dim: i64,
    num_key_value_heads: i64,
    num_key_value_groups: i64,
    max_position_embeddings: i64,
    rope_theta: f64,
    is_causal: bool,

    layer_idx: Option<i32>,
}

impl LlamaSdpaAttentionConfig {
    pub fn new(c: &LlamaConfig) -> Result<Self> {
        let cc = LlamaSdpaAttentionConfig {
            attention_dropout: c.attention_dropout,
            hidden_size: c.hidden_size,
            num_heads: c.num_attention_heads,
            head_dim: c.hidden_size / c.num_attention_heads,
            num_key_value_heads: c.num_key_value_heads,
            num_key_value_groups: c.num_attention_heads / c.num_key_value_heads,
            max_position_embeddings: c.max_position_embeddings,
            rope_theta: c.rope_theta,
            is_causal: true,

            layer_idx: None,
        };
        if cc.hidden_size % cc.num_heads != cc.hidden_size {
            let err = format!("hidden_size must be divisible by num_heads (got `hidden_size`: {} and `num_heads`: {}).", c.hidden_size, c.num_attention_heads);
            return Err(tch::TchError::Shape(err));
        }
        Ok(cc)
    }
}

#[derive(Debug)]
struct LlamaSdpaAttention {
    c: LlamaSdpaAttentionConfig,

    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    rotary_emb: LlamaRotaryEmbedding,
}

unsafe impl Send for LlamaSdpaAttention {}

fn llamaSdpaAttention(vs: &nn::Path, layer_idx: Option<i32>, c: &LlamaConfig) -> Result<LlamaSdpaAttention> {
    let mut c = LlamaSdpaAttentionConfig::new(c)?;

    c.layer_idx = layer_idx;
    if c.layer_idx.is_none() {
        println!(
            "Instantiating LlamaSdpaAttention without passing a `layer_idx` is not recommended and will lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` when creating this class."
        );
    }
        

    let q_proj = linear(vs / "q_proj", c.hidden_size, c.num_heads * c.head_dim);
    let k_proj = linear(
        vs / "k_proj",
        c.hidden_size,
        c.num_key_value_heads * c.head_dim,
    );
    let v_proj = linear(
        vs / "v_proj",
        c.hidden_size,
        c.num_key_value_heads * c.head_dim,
    );
    let o_proj = linear(vs / "o_proj", c.hidden_size, c.hidden_size);
    let rotary_emb = llamaRotaryEmbeddingM! {
        c.head_dim,
        max_position_embeddings: c.max_position_embeddings,
        base: c.rope_theta
    }?;

    Ok(LlamaSdpaAttention {
        c,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        rotary_emb,
    })
}
// class LlamaSdpaAttention(nn.Module):
//     def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
// self.config = config

// self.attention_dropout = config.attention_dropout
// self.hidden_size = config.hidden_size
// self.num_heads = config.num_attention_heads
// self.head_dim = self.hidden_size // self.num_heads
// self.num_key_value_heads = config.num_key_value_heads
// self.num_key_value_groups = self.num_heads // self.num_key_value_heads
// self.max_position_embeddings = config.max_position_embeddings
// self.rope_theta = config.rope_theta
// self.is_causal = True

// if (self.head_dim * self.num_heads) != self.hidden_size:
//     raise ValueError(
//         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
//         f" and `num_heads`: {self.num_heads})."
//     )

// self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
// self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
// self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
// self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
// self.rotary_emb = LlamaRotaryEmbedding(
//     self.head_dim,
//     max_position_embeddings=self.max_position_embeddings,
//     base=self.rope_theta,
// )

impl LlamaSdpaAttention {
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_value: Option<&Tensor>,
        output_attentions: bool,
        use_cache: bool,
        cache_position: Option<&Tensor>,
    ) -> (Tensor, Option<Tensor>, Option<Tensor>) {
        // bsz, q_len, _ = hidden_states.size() # batch size, query length, hidden size
        // query_states: torch.Tensor = self.q_proj(hidden_states)
        // key_states: torch.Tensor = self.k_proj(hidden_states)
        // value_states: torch.Tensor = self.v_proj(hidden_states)

        let bsz = hidden_states.size()[0];
        let q_len = hidden_states.size()[1];

        let query_states = self.q_proj.forward(&hidden_states);
        let key_states = self.k_proj.forward(&hidden_states);
        let value_states = self.v_proj.forward(&hidden_states);

        // query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        // key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        // value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        let query_states = query_states
            .view([bsz, q_len, self.c.num_heads, self.c.head_dim])
            .transpose(1, 2);
        let key_states = key_states
            .view([bsz, q_len, self.c.num_key_value_heads, self.c.head_dim])
            .transpose(1, 2);
        let value_states = value_states
            .view([bsz, q_len, self.c.num_key_value_heads, self.c.head_dim])
            .transpose(1, 2);

        // cos, sin = self.rotary_emb(value_states, position_ids)
        // query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        let (cos, sin) = self
            .rotary_emb
            .forward(&value_states, position_ids.unwrap());
        let (mut query_states, mut key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, &cos, &sin);

        // past_key_value = getattr(self, "past_key_value", past_key_value)

        let past_key_value = match past_key_value {
            Some(past_key_value) => past_key_value,
            None => None,
        };

        // if past_key_value is not None:# DynamicCache()
        //     # sin and cos are specific to RoPE models; position_ids needed for the static cache
        //     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        //     # print(key_states.shape, value_states.shape)
        //     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        //     # print(key_states.shape, value_states.shape)

        if let Some(past_key_value) = past_key_value {
            let cache_kwargs = (sin, cos, cache_position);
            let (key_states, value_states) =
                past_key_value.update(&key_states, &value_states, self.c.layer_idx, cache_kwargs);
        }

        // key_states = repeat_kv(key_states, self.num_key_value_groups)
        // value_states = repeat_kv(value_states, self.num_key_value_groups)

        let key_states = repeat_kv(key_states, self.c.num_key_value_groups);
        let value_states = repeat_kv(value_states, self.c.num_key_value_groups);

        // causal_mask = attention_mask
        // if attention_mask is not None and cache_position is not None:
        //     causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

        let mut causal_mask = attention_mask;
        if let (Some(attention_mask), Some(cache_position)) = (attention_mask, cache_position) {
            causal_mask = Some(&attention_mask.index_select(2, cache_position));
        } else {
            causal_mask = None;
        }

        // if query_states.device.type == "cuda" and causal_mask is not None:
        //     query_states = query_states.contiguous()
        //     key_states = key_states.contiguous()
        //     value_states = value_states.contiguous()

        if query_states.device().is_cuda() && causal_mask.is_some() {
            query_states = query_states.contiguous();
            key_states = key_states.contiguous();
            value_states = value_states.contiguous();
        }

        // attn_output = torch.nn.functional.scaled_dot_product_attention(
        //     query_states,
        //     key_states,
        //     value_states,
        //     attn_mask=causal_mask,
        //     dropout_p=self.attention_dropout if self.training else 0.0,
        // )

        let attn_output = Tensor::scaled_dot_product_attention(
            &query_states,
            &key_states,
            &value_states,
            causal_mask,
            self.c.attention_dropout,
            self.c.is_causal, // is_causal
            None,             // scale
            false,            // dropout_mask
        );

        // attn_output = attn_output.transpose(1, 2).contiguous()
        // attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        // attn_output = self.o_proj(attn_output)

        let attn_output = attn_output.transpose(1, 2).contiguous();
        let attn_output = attn_output.view([bsz, q_len, self.c.hidden_size]);
        let attn_output = self.o_proj.forward(&attn_output);

        // return attn_output, None, past_key_value

        (attn_output, None, past_key_value)
    }
}

fn repeat_kv(hidden_states: Tensor, n_rep: i64) -> Tensor {
    let (batch, num_key_value_heads, slen, head_dim) = hidden_states.size4().unwrap();
    if n_rep == 1 {
        return hidden_states;
    }
    hidden_states
        .view([batch, num_key_value_heads, 1, slen, head_dim])
        .expand(&[batch, num_key_value_heads, n_rep, slen, head_dim], false)
        .view([batch, num_key_value_heads * n_rep, slen, head_dim])
}

// def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
//     """
//     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
//     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
//     """
//     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
//     if n_rep == 1:
// return hidden_states
//     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
//     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

fn rotate_half(x: &Tensor) -> Tensor {
    let dim = x.size().last().unwrap() / 2; // 获取最后一个维度的大小并除以2
    let x1 = x.narrow(-1, 0, dim); // 获取前半部分
    let x2 = x.narrow(-1, dim, dim); // 获取后半部分
    let x2_neg = x2.neg(); // 取后半部分的负值
    tch::Tensor::cat(&[x1, x2_neg], -1) // 沿最后一个维度拼接
}

// def rotate_half(x):
//     """Rotates half the hidden dims of the input."""
//     x1 = x[..., : x.shape[-1] // 2]
//     x2 = x[..., x.shape[-1] // 2 :]
//     return torch.cat((-x2, x1), dim=-1)

fn apply_rotary_pos_emb(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> (Tensor, Tensor) {
    let cos = cos.unsqueeze(1);
    let sin = sin.unsqueeze(1);
    let q_embed = (q * &cos) + (rotate_half(q) * &sin);
    let k_embed = (k * &cos) + (rotate_half(k) * &sin);
    (q_embed, k_embed)
}

// # from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
// def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
//     """Applies Rotary Position Embedding to the query and key tensors.

//     Args:
// q (`torch.Tensor`): The query tensor.
// k (`torch.Tensor`): The key tensor.
// cos (`torch.Tensor`): The cosine part of the rotary embedding.
// sin (`torch.Tensor`): The sine part of the rotary embedding.
// position_ids (`torch.Tensor`, *optional*):
//     Deprecated and unused.
// unsqueeze_dim (`int`, *optional*, defaults to 1):
//     The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
//     sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
//     that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
//     k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
//     cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
//     the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
//     Returns:
// `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
//     """
//     cos = cos.unsqueeze(unsqueeze_dim)
//     sin = sin.unsqueeze(unsqueeze_dim)
//     q_embed = (q * cos) + (rotate_half(q) * sin)
//     k_embed = (k * cos) + (rotate_half(k) * sin)
//     return q_embed, k_embed

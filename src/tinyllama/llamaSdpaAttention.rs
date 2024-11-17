
#[derive(Debug)]
struct LlamaSdpaAttention {
    
}

// class LlamaSdpaAttention(nn.Module):
//     def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
//         self.config = config
//         self.layer_idx = layer_idx
//         if layer_idx is None:
//             print(
//                 f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
//                 "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
//                 "when creating this class."
//             )

//         self.attention_dropout = config.attention_dropout
//         self.hidden_size = config.hidden_size
//         self.num_heads = config.num_attention_heads
//         self.head_dim = self.hidden_size // self.num_heads
//         self.num_key_value_heads = config.num_key_value_heads
//         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
//         self.max_position_embeddings = config.max_position_embeddings
//         self.rope_theta = config.rope_theta
//         self.is_causal = True

//         if (self.head_dim * self.num_heads) != self.hidden_size:
//             raise ValueError(
//                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
//                 f" and `num_heads`: {self.num_heads})."
//             )

//         self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
//         self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
//         self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
//         self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
//         self.rotary_emb = LlamaRotaryEmbedding(
//             self.head_dim,
//             max_position_embeddings=self.max_position_embeddings,
//             base=self.rope_theta,
//         )

//     # Adapted from LlamaAttention.forward
//     def forward(
//         self,
//         hidden_states: torch.Tensor,
//         attention_mask: Optional[torch.Tensor] = None,
//         position_ids: Optional[torch.LongTensor] = None,
//         past_key_value: Optional[Cache] = None,
//         output_attentions: bool = False,
//         use_cache: bool = False,
//         cache_position: Optional[torch.LongTensor] = None,
//     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
//         bsz, q_len, _ = hidden_states.size() # batch size, query length, hidden size

//         query_states: torch.Tensor = self.q_proj(hidden_states)
//         key_states: torch.Tensor = self.k_proj(hidden_states)
//         value_states: torch.Tensor = self.v_proj(hidden_states)

//         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
//         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
//         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

//         cos, sin = self.rotary_emb(value_states, position_ids)
//         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

//         past_key_value = getattr(self, "past_key_value", past_key_value)

//         if past_key_value is not None:# DynamicCache()
//             # sin and cos are specific to RoPE models; position_ids needed for the static cache
//             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
//             # print(key_states.shape, value_states.shape)
//             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
//             # print(key_states.shape, value_states.shape)

//         key_states = repeat_kv(key_states, self.num_key_value_groups)
//         value_states = repeat_kv(value_states, self.num_key_value_groups)

//         causal_mask = attention_mask
//         if attention_mask is not None and cache_position is not None:
//             causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]

//         # print(causal_mask.shape, key_states.shape, value_states.shape)
//         # torch.Size([1, 1, 1, 49]) torch.Size([1, 32, 49, 64]) torch.Size([1, 32, 49, 64])
//         # torch.Size([1, 1, 1, 48]) torch.Size([1, 32, 48, 64]) torch.Size([1, 32, 48, 64])
//         # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
//         # Reference: https://github.com/pytorch/pytorch/issues/112577.
//         if query_states.device.type == "cuda" and causal_mask is not None:
//             query_states = query_states.contiguous()
//             key_states = key_states.contiguous()
//             value_states = value_states.contiguous()

//         attn_output = torch.nn.functional.scaled_dot_product_attention(
//             query_states,
//             key_states,
//             value_states,
//             attn_mask=causal_mask,
//             dropout_p=self.attention_dropout if self.training else 0.0,
//         )

//         attn_output = attn_output.transpose(1, 2).contiguous()
//         attn_output = attn_output.view(bsz, q_len, self.hidden_size)

//         attn_output = self.o_proj(attn_output)

//         return attn_output, None, past_key_value



// def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
//     """
//     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
//     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
//     """
//     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
//     if n_rep == 1:
//         return hidden_states
//     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
//     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

// def rotate_half(x):
//     """Rotates half the hidden dims of the input."""
//     x1 = x[..., : x.shape[-1] // 2]
//     x2 = x[..., x.shape[-1] // 2 :]
//     return torch.cat((-x2, x1), dim=-1)

// # from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

// def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
//     """Applies Rotary Position Embedding to the query and key tensors.

//     Args:
//         q (`torch.Tensor`): The query tensor.
//         k (`torch.Tensor`): The key tensor.
//         cos (`torch.Tensor`): The cosine part of the rotary embedding.
//         sin (`torch.Tensor`): The sine part of the rotary embedding.
//         position_ids (`torch.Tensor`, *optional*):
//             Deprecated and unused.
//         unsqueeze_dim (`int`, *optional*, defaults to 1):
//             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
//             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
//             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
//             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
//             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
//             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
//     Returns:
//         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
//     """
//     cos = cos.unsqueeze(unsqueeze_dim)
//     sin = sin.unsqueeze(unsqueeze_dim)
//     q_embed = (q * cos) + (rotate_half(q) * sin)
//     k_embed = (k * cos) + (rotate_half(k) * sin)
//     return q_embed, k_embed

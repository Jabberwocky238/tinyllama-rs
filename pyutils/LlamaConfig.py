#   "torch_dtype": "bfloat16",
#   "transformers_version": "4.35.0",

# from .PretrainedConfig import PretrainedConfig
from transformers.modeling_utils import PretrainedConfig

class LlamaConfig(PretrainedConfig):
    model_type = "llama"
    
    keys_to_ignore_at_inference = ["past_key_values"]
    vocab_size=32000
    hidden_size=2048
    intermediate_size=5632
    num_hidden_layers=22
    num_attention_heads=32
    num_key_value_heads=4
    hidden_act="silu"
    max_position_embeddings=2048
    max_length=2084
    initializer_range=0.02
    rms_norm_eps=1e-5
    use_cache=True
    pad_token_id=0
    bos_token_id=1
    eos_token_id=2
    pretraining_tp=1
    tie_word_embeddings=False
    rope_theta=10000.0
    rope_scaling=None
    attention_bias=False
    attention_dropout=0.0
    
    def __init__(
        self,
        **kwargs,
    ):
        # self.pad_token_id=pad_token_id
        # self.bos_token_id=bos_token_id
        # self.eos_token_id=eos_token_id
        # self.tie_word_embeddings=tie_word_embeddings
        super().__init__(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            **kwargs,
        )

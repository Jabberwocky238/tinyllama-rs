use std::borrow::Borrow;
use tch::nn;
use tch::nn::Init;
use tch::nn::LinearConfig;
use tch::nn::ModuleT;
use tch::nn::VarStore;
use tch::vision::resnet::resnet18;
use tch::IndexOp;

#[derive(Debug, Clone, Copy)]
pub struct LlamaConfig {
    // pub keys_to_ignore_at_inference: Vec<&'static str>,
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    // pub hidden_act: &'static str,
    pub max_position_embeddings: i64,
    pub max_length: i64,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub pad_token_id: i64,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub pretraining_tp: i64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rope_scaling: Option<f64>,
    pub attention_bias: bool,
    pub attention_dropout: f64,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            max_position_embeddings: 2048,
            max_length: 2084,
            initializer_range: 0.02,
            rms_norm_eps: 1e-5,
            use_cache: true,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
            pretraining_tp: 1,
            tie_word_embeddings: false,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            attention_dropout: 0.0,
        }
    }
}

#[macro_export]
macro_rules! llamaConfigM {
    ($($key:ident: $value:expr,)*) => {
        LlamaConfig {
            $($key: $value,)*
            ..Default::default()
        }
    };
    () => {
        Default::default()
    };
}
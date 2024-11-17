use std::borrow::Borrow;

use tch::nn;
use tch::nn::Init;
use tch::nn::LinearConfig;
use tch::nn::ModuleT;
use tch::nn::VarStore;
use tch::vision::resnet::resnet18;
use tch::IndexOp;
use tch::Tensor;

#[derive(Debug)]
struct LlamaRMSNorm {
    pub eps: f64,
    pub weight: Tensor,
}

unsafe impl Send for LlamaRMSNorm {}

fn llamaRMSNorm(vs: &nn::Path, hidden_size: i64, eps: f64) -> LlamaRMSNorm {
    let w: Tensor = vs.var("layernorm", &[2048], Init::Const(1.0));
    LlamaRMSNorm { eps, weight: w }
}

impl ModuleT for LlamaRMSNorm {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let input_dtype = xs.kind();
        let hidden_states = xs.to_kind(tch::Kind::Float);
        let variance = hidden_states
            .pow(&Tensor::from(2))
            .mean_dim(&[-1i64][..], true, xs.kind());
        let hidden_states = hidden_states * variance.rsqrt() + self.eps;
        let result = &self.weight * hidden_states.to_kind(input_dtype);
        result
    }
}

#[test]
fn load_LlamaRMSNorm() -> Result<(), Box<dyn std::error::Error>> {
    let mut vs = VarStore::new(tch::Device::cuda_if_available());

    let llama_rms_norm = llamaRMSNorm(&vs.root(), 2048, 1e-12);
    let weight_file = "input_layernorm.safetensors";
    vs.load(weight_file)?;

    llama_rms_norm.weight.i(..5).print();

    Ok(())
}
use std::borrow::Borrow;
use tch::nn;
use tch::nn::Init;
use tch::nn::LinearConfig;
use tch::nn::ModuleT;
use tch::nn::VarStore;
use tch::vision::resnet::resnet18;
use tch::IndexOp;
use tch::Tensor;

#[derive(Debug, Clone, Copy)]
struct LlamaConfig {
    pub h_s: i64, // hidden_size
    pub i_s: i64, // intermediate_size
    pub pretraining_tp: i64,
}

#[derive(Debug)]
struct LlamaMLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
    pub config: LlamaConfig,
}

fn llamaMLP(vs: &nn::Path, c: LlamaConfig) -> LlamaMLP {
    let linear_conf = LinearConfig {
        bias: false,
        ..Default::default()
    };
    let gate_proj = nn::linear(vs / "gate_proj", c.h_s, c.i_s, linear_conf);
    let up_proj = nn::linear(vs / "up_proj", c.h_s, c.i_s, linear_conf);
    let down_proj = nn::linear(vs / "down_proj", c.i_s, c.h_s, linear_conf);
    LlamaMLP {
        gate_proj,
        up_proj,
        down_proj,
        config: c,
    }
}

unsafe impl Send for LlamaMLP {}
impl ModuleT for LlamaMLP {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let kind = xs.kind();

        if self.config.pretraining_tp > 1 {
            let slice = self.config.i_s / self.config.pretraining_tp;
            let gate_proj_slices = self.gate_proj.ws.split(slice, 0);
            let up_proj_slices = self.up_proj.ws.split(slice, 0);
            let down_proj_slices = self.down_proj.ws.split(slice, 1);

            // gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            let gate_proj = gate_proj_slices
                .iter()
                .enumerate()
                .map(|(i, linear)| xs.linear::<Tensor>(linear, None))
                .collect::<Vec<_>>();
            let gate_proj = Tensor::cat(&gate_proj, -1);

            // up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            let up_proj = up_proj_slices
                .iter()
                .enumerate()
                .map(|(i, linear)| xs.linear::<Tensor>(linear, None))
                .collect::<Vec<_>>();
            let up_proj = Tensor::cat(&up_proj, -1);

            let intermediate_states = (gate_proj.silu() * up_proj).split(slice, 2);

            // down_proj = [
            //     F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            // ]
            let down_proj = intermediate_states
                .iter()
                .enumerate()
                .map(|(i, intermediate_state)| {
                    intermediate_state.linear::<Tensor>(&down_proj_slices[i], None)
                })
                .collect::<Vec<_>>();
            let down_proj = Tensor::cat(&down_proj, 0);
            down_proj.sum(kind)
        } else {
            // down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            // self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            let gate_proj = xs.linear::<Tensor>(&self.gate_proj.ws, None).silu();
            let up_proj = xs.linear::<Tensor>(&self.up_proj.ws, None);
            let down_proj = self.down_proj.forward_t(&(gate_proj * up_proj), train);
            down_proj
        }
    }
}

#[test]
fn load_LlamaMLP() -> Result<(), Box<dyn std::error::Error>> {
    let mut vs = VarStore::new(tch::Device::cuda_if_available());

    let llama_config = LlamaConfig {
        h_s: 2048,
        i_s: 5632,
        pretraining_tp: 1,
    };
    let llama_mlp_path = &vs.root() / "model.layers.0.mlp";
    let llama_mlp = llamaMLP(&llama_mlp_path, llama_config);
    let weight_file = "model.safetensors";
    vs.load(weight_file)?;

    llama_mlp.gate_proj.ws.i((0, ..5)).print();
    llama_mlp.up_proj.ws.i((0, ..5)).print();
    llama_mlp.down_proj.ws.i((0, ..5)).print();

    Ok(())
}
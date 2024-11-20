use tch::nn;

pub fn linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    nn::linear(vs, in_dim, out_dim, Default::default())
}
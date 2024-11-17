use anyhow::Result;

use tch::Device;

use tch::nn::VarStore;

// fn classifier(p: nn::Path, nclasses: i64) -> impl ModuleT {
//     nn::seq_t()
//         .add_fn_t(|xs, train| xs.dropout(0.5, train))
//         .add(nn::linear(&p / "1", 256 * 6 * 6, 4096, Default::default()))
//         .add_fn(|xs| xs.relu())
//         .add_fn_t(|xs, train| xs.dropout(0.5, train))
//         .add(nn::linear(&p / "4", 4096, 4096, Default::default()))
//         .add_fn(|xs| xs.relu())
//         .add(nn::linear(&p / "6", 4096, nclasses, Default::default()))
// }

pub fn run() -> Result<()> {
    // Create the model and load the pre-trained weights
    let mut vs = VarStore::new(Device::cuda_if_available());
    // let _net = alexnet(&vs.root(), 1000);
    
    Ok(())
}

fn main() -> Result<()> {
    run()?;

    Ok(())
}

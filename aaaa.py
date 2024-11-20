import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaTokenizer, TextGenerationPipeline
from safetensors import safe_open

from utils.LlamaConfig import LlamaConfig

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

from model.LlamaForCausalLM import LlamaForCausalLM

config = LlamaConfig()
model = LlamaForCausalLM(config)
def load(model: nn.Module):
    safetensors = "C:/Users/Administrator/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors"
    result = {}
    with safe_open(safetensors, framework="pt", device='cpu') as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    model.load_state_dict(result, strict=True)
    return model
model = load(model).cuda()


from transformers.models.llama.modeling_llama import LlamaForCausalLM
# model = LlamaForCausalLM.from_pretrained(model_name)

# print(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device="cuda"
)

prompt = "How to get in a good university?"
formatted_prompt = (
    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
)


sequences = pipeline(
    formatted_prompt,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    num_return_sequences=1,
    repetition_penalty=1.1,
    max_new_tokens=256,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
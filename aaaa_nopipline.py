# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
import torch.nn as nn
from safetensors import safe_open

# from transformers import LlamaForCausalLM
from pymodel.LlamaForCausalLM import LlamaForCausalLM
from pyutils.LlamaConfig import LlamaConfig

# from transformers import LlamaTokenizerFast
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from pytokenizer.LlamaTokenizerFast import LlamaTokenizerFast
from pytokenizer.PreTrainedTokenizerBase import PreTrainedTokenizerBase

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_path = "C:/Users/Administrator/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"

config = LlamaConfig()
model = LlamaForCausalLM(config)
def load(model: nn.Module) -> LlamaForCausalLM:
    safetensors = f"{model_path}/model.safetensors"
    result = {}
    with safe_open(safetensors, framework="pt", device='cpu') as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    model.load_state_dict(result, strict=True)
    return model
model = load(model).cuda()

# tokenizer: PreTrainedTokenizerBase = LlamaTokenizerFast.from_pretrained(model_name, local_files_only=True)
tokenizer: PreTrainedTokenizerBase = LlamaTokenizerFast.from_pretrained(model_name, cache_dir=model_path, local_files_only=True)

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

prompt = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
output1 = model.generate(prompt, max_length=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
output = tokenizer.decode(output1[0], skip_special_tokens=True)
print(output)

# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...

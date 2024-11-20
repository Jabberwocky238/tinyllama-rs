import torch
from safetensors import safe_open
from safetensors.torch import load_file
from safetensors.torch import save_file 
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, TextGenerationPipeline, BitsAndBytesConfig
import transformers 
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = LlamaForCausalLM.from_pretrained(model_name,
                                        #  quantization_config=BitsAndBytesConfig(load_in_8bit=True), 
                                         device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="cuda",
)

prompt = "你是傻逼?"
formatted_prompt = (
    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
)

sequences = pipeline(
    formatted_prompt,
    do_sample=True,
    top_k=50,
    top_p = 0.9,
    num_return_sequences=1,
    repetition_penalty=1.1,
    max_new_tokens=1024,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
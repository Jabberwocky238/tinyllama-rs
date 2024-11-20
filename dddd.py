from tokenizer.LlamaTokenizerFast import LlamaTokenizerFast
from tokenizer.PreTrainedTokenizerBase import PreTrainedTokenizerBase

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_path = "C:/Users/Administrator/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"

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
output = tokenizer.decode(prompt, skip_special_tokens=True)
print(output)

# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...

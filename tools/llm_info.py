from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torchinfo import summary

# Adjust to your model
model_name = "meta-llama/Llama-3.2-1B"  
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

for name, param in model.named_parameters():
    print(f"Layer: {name} | Shape: {param.shape}")

# Create a dummy input tensor (tokenized text)
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "Hello, how are you?"
input_tokens = tokenizer(input_text, return_tensors="pt")["input_ids"]  # Ensure it's LongTensor

summary(model, input_data=input_tokens, depth=3, col_names=["input_size", "output_size", "num_params"])

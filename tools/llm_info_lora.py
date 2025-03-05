from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torchinfo import summary
from peft import LoraConfig, get_peft_model

# Adjust to your model
model_name = "meta-llama/Llama-3.2-1B"  
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

config = LoraConfig(
    r=16, # alternative: 32
    lora_alpha=32, # alternative: 64
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
        #"gate_proj", "up_proj", "down_proj",    # MLP layers
        #"lm_head",                              # Output layer
    ],
    bias="none",
    lora_dropout=0.05,  # Dropout applied to LoRA layers
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config).to("cuda")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Shape: {param.shape}")

from transformers import AutoModel

model_name = "meta-llama/Llama-3.2-1B"  
model = AutoModel.from_pretrained(model_name)

min_vals = {}
max_vals = {}

for name, param in model.named_parameters():
    if param.requires_grad:  # Only check trainable weights
        min_vals[name] = param.data.min().item()
        max_vals[name] = param.data.max().item()

# Print min/max values for each layer
for name in min_vals:
    print(f"{name}: min={min_vals[name]:.6f}, max={max_vals[name]:.6f}")

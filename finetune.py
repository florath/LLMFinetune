# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
#
# This used the cleaned dataset as of 202403-04

DATASET_DIR="/mixtral-infer/devel/data/coq-facts-props-proofs-v2"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import concatenate_datasets
from datasets import Dataset
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers
from datetime import datetime
import pandas as pd
import os

# ??? Good idea???
#import warnings warnings.filterwarnings("ignore")

def formatting_func(example):
    if example["proposition"] is not None:
        text = f"### Context: Filename: {example['filename']} Imports: {example['imports']}\n{example['proposition']}\n{example['proof']}"
    else:
        text = f"### Context: Filename: {example['filename']} Imports: {example['imports']}\n{example['fact']}"
    return text

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# +af+ ??? OOM
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

max_length = 256

def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def get_all_data():
    df_facts_raw = pd.read_parquet(os.path.join(DATASET_DIR, "facts.parquet"))
    df_props_proofs_raw = pd.read_parquet(os.path.join(DATASET_DIR, "props-proofs.parquet"))

    print("RAW")
    print(df_facts_raw)
    print(df_props_proofs_raw)

    return Dataset.from_pandas(df_facts_raw), Dataset.from_pandas(df_props_proofs_raw)

dataset_facts, dataset_props_proofs = get_all_data()

# Split the props dataset to train and eval
dataset_props_proofs_split = dataset_props_proofs.train_test_split(test_size=0.05)
dataset_props_proofs_train = dataset_props_proofs_split['train']
dataset_props_proofs_eval = dataset_props_proofs_split['test']
# Merge and shuffle the train dataset
train_dataset_sorted = concatenate_datasets([dataset_props_proofs_train, dataset_facts])
train_dataset = train_dataset_sorted.shuffle()

print("TRAIN")
print(train_dataset)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_eval_dataset = dataset_props_proofs_eval.map(generate_and_tokenize_prompt)

# XXX Added the <s>. Is this fine or better without?
eval_prompt = ""

# Init an eval tokenizer that doesn't add padding or eos token
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True))

# Optional if it does not fit into the GPU mem
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print("================================================================================")
print("===== MODEL")
print(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print("================================================================================")
print_trainable_parameters(model)

print("================================================================================")
print("===== MODEL (adapted)")
print(model)

project = "coq-facts-props-proofs-2"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

# +af+ ??? OOM
torch.cuda.empty_cache()

# About 270000 rows in dataset
# Use batch size of 64
# One epcoch is: 270000 / 64 = 4219
# 10 epochs are 42190
# Very high - will probably stop when I see that the loss is incrasing.
max_steps = 45000 
# Save every 400 (about 1/10 epoch)
save_steps = 400
# Eval every 200 (20 times per epoch)
eval_steps = 200
# This is where the loss is logged
logging_steps = 200

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        # per_device_train_batch_size=2,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=max_steps,
        #learning_rate=2.5e-5, # Want a small lr for finetuning
        learning_rate=5e-5, # About five epochs learing: trying a bit higher rate
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=logging_steps,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=save_steps,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=eval_steps,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        # report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",          # Name of the W&B run (optional)

        # Deprecation warning
        #use_reentrant=True
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("always")

# Load dataset
ds = load_dataset("json", data_files={"train": "/data/train.jsonl", "validation": "/data/valid.jsonl"})

# Load tokenizer and model
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Set pad token before LoRA
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Resize embeddings BEFORE LoRA
model.resize_token_embeddings(len(tokenizer))

# LoRA config and wrapping
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_ds = ds.map(tokenize_function, batched=True)

# 🔍 Token ID verification
def find_max_token_id(dataset, split):
    max_id = -1
    for item in dataset[split]:
        max_in_sample = max(item["input_ids"])
        if max_in_sample > max_id:
            max_id = max_in_sample
    return max_id

max_token_id = find_max_token_id(tokenized_ds, "train")
embed_size = model.base_model.transformer.wte.weight.size(0)
print(f"✅ Max token ID in training set: {max_token_id}")
print(f"✅ Embedding size: {embed_size}")
assert max_token_id < embed_size, f"🚨 Token ID {max_token_id} exceeds embedding size {embed_size}"

# Training setup
training_args = TrainingArguments(
    output_dir="/output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    logging_dir="/var/log/katib",
    logging_steps=10,
    report_to="none",
    fp16=torch.cuda.is_available()
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train!
trainer.train()

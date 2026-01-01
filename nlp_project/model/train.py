import json
import os
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)

# -----------------------------
# 0. Reproducibility
# -----------------------------
set_seed(42)

# -----------------------------
# 1. Load search-style corpus from text file
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "queries.txt")

with open(data_path, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

dataset = Dataset.from_dict({"text": texts})

# -----------------------------
# 2. Tokenizer (FAST – REQUIRED)
# -----------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding=False
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

# -----------------------------
# 3. Model
# -----------------------------
model = GPT2LMHeadModel.from_pretrained("gpt2")

# -----------------------------
# 4. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="checkpoint",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=5,
    save_total_limit=2,
    logging_steps=1,
    report_to="none"
)

# -----------------------------
# 5. Data collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

# -----------------------------
# 7. Save metrics
# -----------------------------
metrics_path = os.path.join(BASE_DIR, "metrics.json")

metrics = {
    "epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size,
    "learning_rate": training_args.learning_rate,
    "log_history": trainer.state.log_history
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

# -----------------------------
# 8. Save FINAL model + tokenizer
# -----------------------------
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "final_model")

model.save_pretrained(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)

print("✅ Training complete. Model saved to model/final_model/")

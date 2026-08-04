# pip install transformers datasets accelerate bitsandbytes peft fastchat

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

# ========== Step 1: Load a Text-to-Text Model ==========
model_name = "t5-large"  # Can be "t5-base", "facebook/bart-large", or LLaMA
tokenizer = AutoTokenizer.from_pretrained(model_name,                                         
                                          # load_in_8bit=True,  # Enables 8-bit optimization for efficient training
                                          # torch_dtype=torch.float16,  # Use mixed precision)
                                          device_map="auto"
)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

# ========== Step 2: Load a Text2Text Dataset ==========
with open("text_no_color.json", "r") as file:
    dataset = json.load(file) # load_dataset("xsum")  # Text summarization dataset
train_list, test_list = dataset["train"], dataset["test"]

train_dataset_dicts = [
    {"inputs": inp, "outputs": out}
    for inp, out in zip(train_list["input"], train_list["output"])
]

test_dataset_dicts = [
    {"inputs": inp, "outputs": out}
    for inp, out in zip(test_list["input"], test_list["output"])
]

train_data = Dataset.from_list(train_dataset_dicts)
test_data = Dataset.from_list(test_dataset_dicts)

# Function to preprocess data
def preprocess_function(examples):
    instruction = ""
    if instruction == "":
        inputs = examples["inputs"]
    else:
        inputs = [" ".join([instruction, exam]) for exam in examples["inputs"]]  # Input text
    targets = examples["outputs"]  # Summary (expected output)
    model_inputs = tokenizer(inputs, max_length=256, truncation=False, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=False, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
train_data = train_data.map(preprocess_function, batched=True)
val_data = test_data.map(preprocess_function, batched=True)

# ========== Step 3: Apply LoRA Fine-Tuning (RoLA) ==========
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]  # Fine-tune attention layers
)

model = get_peft_model(model, lora_config)
print("LoRA applied to:", ["q", "v", "k", "o", "wi_0", "wi_1", "wo"])

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Total Parameters: {total_params:,}")
print(f"Trainable LoRA Parameters: {trainable_params:,}")
print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%")

# ========== Step 4: Define Training Parameters ==========
training_args = TrainingArguments(
    output_dir="./checkpoints/lora/t5-large_no_color",
    per_device_train_batch_size=28,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    num_train_epochs=4,
    logging_dir="./logs",
#    logging_steps=10, 
#    fp16=True,  # Enable mixed precision
    push_to_hub=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data
)

# ========== Step 5: Train the Model ==========
trainer.train()

# ========== Step 6: Evaluate the Model ==========
def generate_summary(text):
    """Generate text-to-text output using RoLA fine-tuned model."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
    output_ids = model.generate(**inputs, max_length=150)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example
text_input = "The latest research in AI is transforming multiple industries including healthcare and finance."
print("\n[Generated Summary]:", generate_summary(text_input))

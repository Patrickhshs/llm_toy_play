import os
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
# token = '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'
# login(token=token)
# Load the tokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

# LoRA fine-tune
model = get_peft_model(model, lora_config)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized_output = tokenizer(examples["text"], truncation=True, max_length=128,padding='max_length')
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
training_args = TrainingArguments(
    output_dir="/gpfsnyu/scratch/js12556/llama-2/results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="/gpfsnyu/scratch/js12556/llama-2/logs",
    logging_steps=10,
    fp16=True, 
    dataloader_drop_last=True,  # Drop the last incomplete batch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator, 
)

trainer.train()
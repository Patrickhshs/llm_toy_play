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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the first LLaMA model and tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# model_name_1 = "meta-llama/Llama-2-7b-chat-hf"
model_name_1 = "meta-llama/Llama-2-7b-hf"
tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
model_1 = AutoModelForCausalLM.from_pretrained(model_name_1)

# Load the second LLaMA model and tokenizer
# model_name_1 = "meta-llama/Llama-2-7b-chat-hf"
model_name_2 = "meta-llama/Llama-2-7b-hf"
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
model_2 = AutoModelForCausalLM.from_pretrained(model_name_2)

def simulate_conversation(initial_prompt, num_turns):
    prompt = initial_prompt
    for turn in range(num_turns):
        if turn % 2 == 0:
            # Model 1 generates a response
            inputs = tokenizer_1(prompt, return_tensors="pt")
            outputs = model_1.generate(**inputs, max_length=256, pad_token_id=tokenizer_1.eos_token_id)
            response = tokenizer_1.decode(outputs[0], skip_special_tokens=True)
            print(f"Model 1: {response}\n")
        else:
            # Model 2 generates a response
            inputs = tokenizer_2(response, return_tensors="pt")
            outputs = model_2.generate(**inputs, max_length=256, pad_token_id=tokenizer_2.eos_token_id)
            response = tokenizer_2.decode(outputs[0], skip_special_tokens=True)
            print(f"Model 2: {response}\n")
        prompt = response

# Example usage
initial_prompt = "Hello, how are you?"
num_turns = 5
simulate_conversation(initial_prompt, num_turns)
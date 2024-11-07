import torch
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from module import load_prompts

# Load Python and Java prompts
PYTHON_PROMPTS, JAVA_PROMPTS = load_prompts()
print("Prompts loaded successfully.")

# Choose a random prompt from the JSONLines Reader by converting it to a list of items first
# Convert JSONLines Reader to a list and select a random prompt
prompts_list = list(PYTHON_PROMPTS)
random_prompt = random.choice(prompts_list)
random_id = random_prompt["question_id"]
prompt_text = random_prompt["prompt"]

print(f"Randomly selected prompt ID:\n{prompt_text}")

# Global variables
MAX_LENGTH = 1024  # Max window length
NUM_SAMPLES = 10   # Number of samples to generate
TEMPERATURE = 0.8

# Model setup
checkpoint = "Salesforce/codegen2-1B_P"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_cache_dir = "./models"

# Ensure the cache directory exists
os.makedirs(model_cache_dir, exist_ok=True)

# Tokenizer setup with pad_token_id set
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id
print("Tokenizer loaded successfully.")


model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    cache_dir=model_cache_dir
).to(device)
print("Model loaded successfully.")

# Generate samples for the test prompt
# Prepare inputs with attention mask
inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],  # Add attention_mask for reliability
    max_length=MAX_LENGTH,
    do_sample=True,
    temperature=TEMPERATURE,
    num_return_sequences=NUM_SAMPLES
)

# Decode and print each generated sample
print(f"Generated samples for prompt: {prompt_text}\n")
for i, output in enumerate(outputs, 1):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Sample {i}:\n{generated_text}\n{'-'*40}\n")

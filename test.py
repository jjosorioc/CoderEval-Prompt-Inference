import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from module import load_prompts

# Load Python and Java prompts
PYTHON_PROMPTS, JAVA_PROMPTS = load_prompts()
print("Prompts loaded successfully.")

# Choose one prompt to test
test_prompt = PYTHON_PROMPTS[0]  # Replace with a specific index or prompt as needed
prompt_text = test_prompt["prompt"]

# Global variables
MAX_LENGTH = 1024  # Max window length
NUM_SAMPLES = 10   # Number of samples to generate
TEMPERATURE = 0.8

# Model setup
checkpoint = "Deci/DeciCoder-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_cache_dir = "./models"

# Ensure the cache directory exists
os.makedirs(model_cache_dir, exist_ok=True)

# Load or cache the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
print("Tokenizer loaded successfully.")

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    cache_dir=model_cache_dir
).to(device)
print("Model loaded successfully.")

# Generate samples for the test prompt
inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
outputs = model.generate(
    inputs["input_ids"],
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

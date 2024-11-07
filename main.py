import torch
import jsonlines
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from module import load_prompts

# Load Python and Java prompts
PYTHON_PROMPTS, JAVA_PROMPTS = load_prompts()

# Global variables
MAX_LENGTH = 1024  # Max window length
NUM_SAMPLES = 10   # Number of samples to generate
TEMPERATURE = 0.8

# Model setup
checkpoint = "Deci/DeciCoder-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_cache_dir = "./modules"

# Ensure the cache directory exists
os.makedirs(model_cache_dir, exist_ok=True)

# Load or cache the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    cache_dir=model_cache_dir
).to(device)

# Function to generate and store samples for each prompt
def generate_and_save_samples(prompts, output_path) -> None:
    """
    Generates multiple code samples for each prompt using a language model 
    and saves the results in a JSONL file.

    Args:
        prompts (list): A list of prompt dictionaries, where each dictionary 
                        contains a "question_id" (unique identifier) and "prompt" 
                        (the text input for code generation).
        output_path (str): The file path where the generated samples will be saved 
                           in JSONL format.

    Returns:
        None: This function does not return a value. It writes the generated 
              outputs to the specified JSONL file.

    Each generated sample is stored with the following structure in the output file:
        {
            "_id": <question_id>,
            "generate_results": [<sample1>, <sample2>, ..., <sample10>]
        }

    Notes:
        - The function generates 10 samples per prompt using nucleus sampling with a
          temperature setting for controlled randomness.
        - Each generated sample is decoded and added as a string to the "generate_results" list.
    """

    # Ensure the output directory exists by extracting the directory from output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with jsonlines.open(output_path, mode='w') as writer:
        for prompt in prompts:
            question_id = prompt.get("question_id")
            prompt_text = prompt.get("prompt")
            
            # Tokenize the prompt text
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            # Generate samples
            outputs = model.generate(
                inputs["input_ids"],
                max_length=MAX_LENGTH,
                do_sample=True,
                temperature=TEMPERATURE,
                num_return_sequences=NUM_SAMPLES
            )

            # Decode each output and store as a list of strings
            generate_results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            # Write to JSONL format
            writer.write({"_id": question_id, "generate_results": generate_results})

# Generate and save samples for Python and Java prompts
generate_and_save_samples(PYTHON_PROMPTS, './generated_outputs/python_outputs.jsonl')
generate_and_save_samples(JAVA_PROMPTS, './generated_outputs/java_outputs.jsonl')

print("Generation complete and saved to JSONL files.")

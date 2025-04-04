import torch
import jsonlines
import os
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from module import load_prompts
from module import extract_python_code

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load Python and Java prompts
PYTHON_PROMPTS, JAVA_PROMPTS = load_prompts()
logging.info("Prompts loaded successfully.")

# Global variables
MAX_LENGTH = 1500  # Max window length
NUM_SAMPLES = 10   # Number of samples to generate
TEMPERATURE = 1.0

# Model setup
checkpoint = "Salesforce/codegen2-1B_P"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_cache_dir = "./models"

# Ensure the cache directory exists
os.makedirs(model_cache_dir, exist_ok=True)

# Load or cache the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)

logging.info("Tokenizer loaded successfully.")

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    torch_dtype= "auto", 
    trust_remote_code=True,
    cache_dir=model_cache_dir
).to(device)
logging.info("Model loaded successfully.")

# Validate endofcode token
try:
    endofcode_id = tokenizer.encode("<|endofcode|>", add_special_tokens=False)[0]
    logging.info(f"<|endofcode|> token ID: {endofcode_id}")
    logging.info(f"The eos_token symbol or word is: {tokenizer.decode([endofcode_id])}")
except IndexError:
    logging.error("<|endofcode|> token is not recognized. Falling back to eos_token_id.")
    endofcode_id = tokenizer.eos_token_id
    logging.error(f"eos_token_id: {endofcode_id}")
    eos_token = tokenizer.decode([endofcode_id])
    logging.error(f"The eos_token symbol or word is: {eos_token}")
    



# Function to generate and save samples for each prompt
def generate_and_save_samples(prompts, original_output_path: str, processed_output_path: str) -> None:
    """
    Generates multiple code samples for each prompt using a language model 
    and saves both the original and processed (extracted) code results in separate JSONL files.

    Args:
        prompts (list): A list of prompt dictionaries, where each dictionary 
                        contains a "question_id" (unique identifier) and "prompt" 
                        (the text input for code generation).
        original_output_path (str): The file path where the original generated samples will be saved.
        processed_output_path (str): The file path where the processed (extracted) samples will be saved.

    Returns:
        None: This function does not return a value. It writes the outputs to the specified JSONL files.
    """
    # Ensure the output directories exist
    original_output_dir = os.path.dirname(original_output_path)
    processed_output_dir = os.path.dirname(processed_output_path)
    if original_output_dir:
        os.makedirs(original_output_dir, exist_ok=True)
    if processed_output_dir:
        os.makedirs(processed_output_dir, exist_ok=True)

    with jsonlines.open(original_output_path, mode='w') as original_writer, \
         jsonlines.open(processed_output_path, mode='w') as processed_writer:
        for prompt in prompts:
            question_id = prompt.get("question_id")
            prompt_text = prompt.get("prompt")
            
            # Tokenize the prompt text
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            # Generate samples
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=MAX_LENGTH,
                do_sample=True,
                temperature=TEMPERATURE,
                num_return_sequences=NUM_SAMPLES,
                eos_token_id = endofcode_id
            )

            # Decode each output and save both original and processed versions
            original_results = []
            processed_results = []
            input_length = inputs["input_ids"].shape[1]  # Length of the prompt tokens

            for output in outputs:
                # Decode the generated content while excluding the prompt
                generated_tokens = output[input_length:]
                decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                # Strip potential residual prompt overlap in case of partial matches
                if decoded.startswith(prompt_text.strip()):
                    decoded = decoded[len(prompt_text.strip()):].strip()
                
                original_results.append(decoded)
                processed_code = extract_python_code(decoded)
                processed_results.append(processed_code)


            # Write results to JSONL files
            original_writer.write({"_id": question_id, "generate_results": original_results})
            processed_writer.write({"_id": question_id, "generate_results": processed_results})

# Generate and save samples for Python prompts
checkpoint_safe_name = checkpoint.replace("/", "_")
original_output_path = f'./generated_outputs/{checkpoint_safe_name}/python_original_outputs.jsonl'
processed_output_path = f'./generated_outputs/{checkpoint_safe_name}/python_processed_outputs.jsonl'

generate_and_save_samples(PYTHON_PROMPTS, original_output_path, processed_output_path)
logging.info("Python generation complete.")

# Uncomment and modify the following lines if you want to generate samples for Java prompts
# original_java_output_path = f'./generated_outputs/{checkpoint_safe_name}/java_original_outputs.jsonl'
# processed_java_output_path = f'./generated_outputs/{checkpoint_safe_name}/java_processed_outputs.jsonl'
# generate_and_save_samples(JAVA_PROMPTS, original_java_output_path, processed_java_output_path)
# print("Java generation complete.")

logging.info("Generation complete and saved to JSONL files.")

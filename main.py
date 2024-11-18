import torch
import jsonlines
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from module import load_prompts

# Load Python and Java prompts
PYTHON_PROMPTS, JAVA_PROMPTS = load_prompts()
print("Prompts loaded successfully.")

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
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id
print("Tokenizer loaded successfully.")

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    cache_dir=model_cache_dir
).to(device)
print("Model loaded successfully.")

# Function to generate and store samples for each prompt
import os
import re
import jsonlines

def generate_and_save_samples(prompts, output_path) -> None:
    """
    Generates multiple code samples for each prompt using a language model 
    and saves the extracted Python code results in a JSONL file.

    Args:
        prompts (list): A list of prompt dictionaries, where each dictionary 
                        contains a "question_id" (unique identifier) and "prompt" 
                        (the text input for code generation).
        output_path (str): The file path where the generated samples will be saved 
                           in JSONL format.

    Returns:
        None: This function does not return a value. It writes the extracted code 
              outputs to the specified JSONL file.

    Each extracted code sample is stored with the following structure in the output file:
        {
            "_id": <question_id>,
            "generate_results": [<code_sample1>, <code_sample2>, ..., <code_sample10>]
        }

    Notes:
        - The function generates 10 samples per prompt using nucleus sampling with a
          temperature setting for controlled randomness.
        - Each generated sample is decoded and processed to extract only the Python code.
    """

    # Ensure the output directory exists by extracting the directory from output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define a function to extract the code from the generated text
    def extract_code(s):
        """
        Extracts the Python code from the generated text.
        The code starts after '<|python|>' and ends at a sequence of '#' characters or at the end of the string.

        Args:
            s (str): The input string containing code and other text.

        Returns:
            str: The extracted Python code.
        """
        # Find the position of '<|python|>'
        start_marker = '<|python|>'
        start_idx = s.find(start_marker)
        if start_idx == -1:
            return ''  # Marker not found

        # Start of code
        code_start = start_idx + len(start_marker)

        # Define regex pattern to find a sequence of '#' characters (e.g., 50 or more)
        end_pattern = r'#{20,}'  # Adjust the number as needed
        # Search for the end pattern after code_start
        match = re.search(end_pattern, s[code_start:])
        if match:
            # End of code is where the pattern starts
            code_end = code_start + match.start()
        else:
            # If no end pattern found, code goes till the end
            code_end = len(s)

        # Extract code
        code = s[code_start:code_end].strip()
        return code

    with jsonlines.open(output_path, mode='w') as writer:
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
                num_return_sequences=NUM_SAMPLES
            )

            # Decode each output and extract the code
            generate_results = []
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                code = extract_code(decoded)
                generate_results.append(code)

            # Write to JSONL format
            writer.write({"_id": question_id, "generate_results": generate_results})


# Generate and save samples for Python and Java prompts
generate_and_save_samples(PYTHON_PROMPTS, './generated_outputs/python_outputs.jsonl')
print("Python generation complete.")

generate_and_save_samples(JAVA_PROMPTS, './generated_outputs/java_outputs.jsonl')
print("Java generation complete.")

print("Generation complete and saved to JSONL files.")

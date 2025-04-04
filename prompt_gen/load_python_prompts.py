import jsonlines

# Define the paths to the JSONL files
PYTHON_JSONL_PATH = './prompt_gen/CoderEval-Input4Models/CEPythonRaw.jsonl'
JAVA_JSONL_PATH = './prompt_gen/CoderEval-Input4Models/CEJavaRaw.jsonl'

# Create a dictionary to store prompts using question_id and language as keys
prompts = {"python": {}, "java": {}}

# Function to generate prompts from a JSONL file
def generate_prompts(file_path, language):
    with jsonlines.open(file_path) as data:
        for obj in data:
            # Constructing the prompt string using all values from each entry
            prompt_str = (
                f"Task: Generate the function implementation based on the provided signature, "
                f"description, and input code.\n\n"
                f"Function Signature:\n{obj['signature']}\n\n"
                f"Function Description:\n{obj['docstring']}\n\n"
                f"Function Input Code:\n{obj['input']}\n\n"
                f"Only return the relevant code implementation. Nothing else should be included."
            )
            
            # Store the prompt using question_id as the key under the specified language
            prompts[language][obj["question_id"]] = prompt_str

# Generate prompts for both Python and Java
generate_prompts(PYTHON_JSONL_PATH, "python")
generate_prompts(JAVA_JSONL_PATH, "java")

# Define output paths for the generated prompts
output_path_python = './prompt_gen/prompts/python_generated_prompts.jsonl'
output_path_java = './prompt_gen/prompts/java_generated_prompts.jsonl'

# Write prompts to JSONL files for each language
for language, output_path in [("python", output_path_python), ("java", output_path_java)]:
    with jsonlines.open(output_path, mode='w') as writer:
        for question_id, prompt in prompts[language].items():
            writer.write({"question_id": question_id, "prompt": prompt})
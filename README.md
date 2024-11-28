# CoderEval-Prompt-Inference

This project generates code samples based on provided function signatures, descriptions, and input code using a language model. It supports both Python and Java prompts and leverages the `transformers` library for generating code samples.

---

## Project Structure

```
.
├── .gitignore
├── main.py
├── module.py
├── prompt_gen/
│   ├── CoderEval-Input4Models/
│   │   ├── CEJavaHumanLabel.jsonl
│   │   ├── CEJavaRaw.jsonl
│   │   ├── CEPythonHumanLabel.jsonl
│   │   ├── CEPythonRaw.jsonl
│   ├── load_python_prompts.py
│   ├── prompts/
│   │   ├── java_generated_prompts.jsonl
│   │   ├── python_generated_prompts.jsonl
├── README.md
├── requirements.txt
├── results/
└── test.py
```

---

## Files and Directories

- **`main.py`**: The main script to load prompts, set up the model, and generate code samples.
- **`module.py`**: Contains utility functions for loading prompts and extracting Python code.
- **`prompt_gen/`**: Directory containing input JSONL files and scripts for generating prompts.
  - **`CoderEval-Input4Models/`**: Contains raw and human-labeled JSONL files for Python and Java.
  - **`load_python_prompts.py`**: Script to generate prompts from raw JSONL files.
  - **`prompts/`**: Directory containing pre-generated prompts for Python and Java.
- **`requirements.txt`**: Lists the dependencies required for the project.
- **`results/`**: Directory to store the generated results.
- **`test.py`**: Script for testing purposes.

---

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd CoderEval-Prompt-Inference
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary model files cached or downloaded.

---

## Usage

1. **Generate prompts from raw JSONL files**:
   ```bash
   python prompt_gen/load_python_prompts.py
   ```

2. **Run the main script to generate code samples**:
   ```bash
   python main.py
   ```

3. The generated code samples will be saved in the `generated_outputs` directory.

---

## Functions

### `module.py`
- **`load_prompts() -> tuple`**  
  Loads and returns the Python and Java prompts from pre-generated JSONL files.

- **`extract_python_code(code_string: str) -> str`**  
  Extracts Python code from the given content, removing all comments and docstrings.

### `load_python_prompts.py`
- **`generate_prompts(file_path, language)`**  
  Generates prompts from a JSONL file and stores them in a dictionary.

---

## Model and Tokenizer

The models and tokenizers are loaded and cached in the `./models` directory.

---

## Logging

The project uses the `logging` module to log information and errors during execution. Logs are printed to the console.
```


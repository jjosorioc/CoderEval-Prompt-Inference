import jsonlines

def load_prompts() -> tuple:
    """
    Loads and returns the Python and Java prompts from pre-generated JSONL files.

    Returns:
        tuple: A tuple containing two `jsonlines.Reader` objects. The first object is for Python
        prompts and the second for Java prompts.
    """
    
    python_path = './prompt_gen/prompts/python_generated_prompts.jsonl'
    java_path = './prompt_gen/prompts/java_generated_prompts.jsonl'

    PYTHON_PROMPTS = jsonlines.open(python_path)
    JAVA_PROMPTS = jsonlines.open(java_path)

    return PYTHON_PROMPTS, JAVA_PROMPTS

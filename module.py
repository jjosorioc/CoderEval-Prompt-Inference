import jsonlines
import ast
import astor
import io
import tokenize
import logging

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


# Function to extract Python code using AST


def extract_python_code(code_string: str) -> str:
    """
    Extracts Python code from the given content, removing all comments and docstrings.

    Args:
        code_string (str): The input string containing Python code.

    Returns:
        str: The code without comments and docstrings.
    """
    try:
            # First, remove docstrings using the AST module
            parsed = ast.parse(code_string)

            for node in ast.walk(parsed):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    if (docstring := ast.get_docstring(node)) is not None:
                        # Remove the docstring by deleting the first node in the body if it's a docstring
                        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                            node.body = node.body[1:]

            # Convert the AST back to source code
            code_without_docstrings = ast.unparse(parsed) if hasattr(ast, 'unparse') else astor.to_source(parsed)

            # Now, remove comments using the tokenize module
            io_obj = io.StringIO(code_without_docstrings)
            output_tokens = []

            tokens = tokenize.generate_tokens(io_obj.readline)
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0

            for tok in tokens:
                token_type, token_string, start, end, line = tok
                if token_type == tokenize.COMMENT:
                    # Skip comments
                    continue
                else:
                    output_tokens.append((token_type, token_string))

            # Reconstruct the code from tokens
            code_without_comments = tokenize.untokenize(output_tokens)

            return code_without_comments
    except Exception as e:
        logging.error(f"Error extracting Python code: {e}")
        return code_string
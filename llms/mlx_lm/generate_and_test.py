import argparse
import re
import io
import unittest

from contextlib import contextmanager
from typing import Optional, Any

import mlx.core as mx

from .utils import generate, load

DEFAULT_MODEL_PATH = "mlx-community/stablelm-2-zephyr-1_6b-4bit"
DEFAULT_CODE_PROMPT = "Write a simple python function that takes a value as an input and returns that value."
DEFAULT_TEST_PROMPT = """Write very basic unit tests in Python using the unittest framework for the following function:\n
Guidelines for Writing Tests:
1. Import unittest at the start.
2. Create a class that inherits from unittest.TestCase.
3. You can use following assertion methods to verify that the function behaves as expected. 
   Some common assertions include:
   - assertEqual(a, b): Checks if a equals b.
   - assertTrue(x): Checks if x is True.
   - assertFalse(x): Checks if x is False.
   - assertRaises(Error, func, *args, **kwargs): Checks if calling func with args and kwargs raises an Error.
4. Remember to include if __name__ == '__main__': at the end to make the test executable.

\n{}\n

"""
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMP = 0.6
DEFAULT_SEED = 42


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_CODE_PROMPT, help="Message to be processed by the model"
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        default=True,
        help="Colorize output based on T[0] probability",
    )
    return parser


def colorprint(color, s):
    color_codes = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 39,
    }
    ccode = color_codes.get(color, 30)
    print(f"\033[1m\033[{ccode}m{s}\033[0m", end="", flush=True)


def colorprint_by_t0(s, t0):
    if t0 > 0.95:
        color = "white"
    elif t0 > 0.70:
        color = "green"
    elif t0 > 0.30:
        color = "yellow"
    else:
        color = "red"
    colorprint(color, s)

def extract_code_from_markdown(text: str, pattern) -> Optional[str]:
    """
    Extracts code from a Markdown fenced code block.

    Args:
        text (str): Text containing Markdown fenced code block.

    Returns:
        Optional[str]: Extracted code if found, otherwise None.
    """

    match = re.search(pattern, text, re.DOTALL)
    if match:
        extracted_code = match.group(1).strip()
        print("Extracted Code:", extracted_code)  # Debugging print
        return extracted_code
    return None


def run(script: str) -> str:
    """
    Executes the combined Python code and unit tests and returns the test results.

    Args:
        script (str): Combined Python code and unit tests.

    Returns:
        str: Results of the unit tests.
    """
    # Creating a new namespace for executing the script
    namespace = {}
    exec(script, namespace)

    # Create a loader and suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Debug: Print loaded test methods and classes
    for name, obj in namespace.items():
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
            print(f"Found Test Class: {name}")
            loaded_suite = loader.loadTestsFromTestCase(obj)
            print(f"Loaded Tests: {loaded_suite.countTestCases()}")
            suite.addTests(loaded_suite)

    try:
        if suite.countTestCases() > 0:
            result_stream = io.StringIO()
            runner = unittest.TextTestRunner(stream=result_stream)
            runner.run(suite)

            # Debug print
            print("Test Execution Output:", result_stream.getvalue())

            return result_stream.getvalue()
        else:
            return "No tests were found to run."
    except Exception as e:
        return f"Error during test execution: {e}"   

def python(code: str, ask_for_input: bool = True) -> Any:
    """
    Executes the provided Python code. Optionally asks for user confirmation.

    Args:
        code (str): Python code to be executed.
        ask_for_input (bool): Whether to ask for user input to proceed.

    Returns:
        Any: Result of executing the code, or `#FAIL#` if execution is not permitted.
    """
    if ask_for_input:
        colorprint('red', f'Proceed with execution? Press Y if you are sure \n```\n{code}\n```\n')  # Colorize the prompt
        go = input()
        if go.lower() != 'y': return '#FAIL#'
    return run(code)

@contextmanager
def load_model_context(model_name: str, tokenizer_config: dict):
    """
    Context manager for loading and unloading a model.

    Args:
        model_name (str): Name or path of the model to be loaded.
        tokenizer_config (dict): Configuration for the tokenizer.

    Yields:
        Tuple: Loaded model and tokenizer.
    """
    model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
    try:
        yield model, tokenizer
    finally:
        del model, tokenizer # Ensure the model is deleted to free up memory

def run_model_generate_code(model: Any, tokenizer: Any, prompt: str, temp: float, max_tokens: int, ignore_chat_template: bool, formatter: Any, pattern) -> str:
    # Apply chat template if available and not ignored
    if not ignore_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    generated_code = generate(model, tokenizer, prompt, temp, max_tokens, True, formatter)
    extracted_code = extract_code_from_markdown(generated_code, pattern)
    
    # If no code was extracted, then stop further processing
    if extracted_code is None:
        print("No code block found.")
        return None

    return extracted_code

def combine_code_and_tests(code: str, tests: str) -> str:
    return f"{code}\n\n{tests}"

def create_full_context(script, test_results):
    return (
        "Analyze the Python script and unit test results provided below. "
        "Identify errors, inconsistencies, or areas for improvement, and correct them. "
        "Provide a revised version of the script."
        "Script:\n```python\n" + script + "\n```\n\n"
        "Unit Test Results (analyze and fix):\n" + test_results + "\n\n"
        "Make sure the updated unit tests comprehensively cover all scenarios, including edge cases.\n"
        "Format the corrections as a single Python code block (```python```). "
    )


def apply_model_advice():
    return input("Apply the above advice? (y/n): ").lower() == 'y'

def decide_to_give_up():
    return input("Would you like to give up? (y/n): ").lower() == 'y'

def is_test_successful(test_result):
    return "FAILED" not in test_result and "ERROR" not in test_result

def main(args):
    mx.random.seed(args.seed)

    # Building tokenizer_config
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token


    with load_model_context(args.model, tokenizer_config) as (model, tokenizer):
        formatter = colorprint_by_t0 if args.colorize else None
        
        # Initial prompt for generating Python code
        current_prompt = args.prompt
        retry_generation = False
        ask_for_input = True
        # Regular expression pattern for fenced code block
        # Adjusted to account for variations in whitespaces and line breaks
        pattern = r"```python[\s\S]*?(.*?)```"
        
        while True:
            # If retry_generation is False, generate new Python code
            if not retry_generation:
                python_code = run_model_generate_code(model, tokenizer, current_prompt, args.temp, args.max_tokens, args.ignore_chat_template, formatter, pattern)
                if python_code is None:
                    break

                # Generate unit tests for the Python code
                unit_test_code = run_model_generate_code(model, tokenizer, DEFAULT_TEST_PROMPT.format(python_code), args.temp, args.max_tokens, args.ignore_chat_template, formatter, pattern)
                if unit_test_code is None:
                    break

                combined_script = combine_code_and_tests(python_code, unit_test_code)
            else:
                # Reset the flags
                retry_generation = False  
                ask_for_input = False 

            combined_result = python(combined_script, ask_for_input=ask_for_input)

            if combined_result != '#FAIL#' and is_test_successful(combined_result):
                print("All tests passed successfully!")
                break

            # Get advice from the model to fix errors
            full_context = create_full_context(script=combined_script, test_results=combined_result)
            model_advice = run_model_generate_code(model, tokenizer, full_context, args.temp, args.max_tokens, args.ignore_chat_template, formatter, pattern)

            if model_advice is None:
                print("Model could not fix the code.")

            if apply_model_advice():
                # Update the current prompt with the model's advice for the next iteration
                combined_script = model_advice
                retry_generation = True
                continue
            elif decide_to_give_up():
                print("Process terminated by user.")
                break
            else:
                print("Continuing the process without applying advice.")
                
if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)

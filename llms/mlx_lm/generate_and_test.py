# python -m mlx_lm.generate_and_test

import argparse
import re
import io
import unittest

from contextlib import contextmanager
from typing import Optional, Any

import mlx.core as mx

from .utils import generate, load

DEFAULT_MODEL_PATH = "mlx-community/stablelm-2-zephyr-1_6b-4bit"
DEFAULT_CODE_PROMPT = "Create a Python function named 'calculate_factorial' that takes a non-negative integer and returns its factorial."
DEFAULT_TEST_PROMPT = "Write basic unit tests for the following code, make no mistake:\n{}"
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMP = 0.6
DEFAULT_SEED = 0


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

def extract_code_from_markdown(text: str) -> Optional[str]:
    """
    Extracts code from a Markdown fenced code block.

    Args:
        text (str): Text containing Markdown fenced code block.

    Returns:
        Optional[str]: Extracted code if found, otherwise None.
    """
    # Regular expression pattern for fenced code block
    # Adjusted to account for variations in whitespaces and line breaks
    pattern = r"```python[\s\S]*?(.*?)```"
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

    # Run the tests if any are found
    if suite.countTestCases() > 0:
        # Define a custom stream for capturing test results
        result_stream = io.StringIO()

        # Create a test runner that uses the custom stream
        runner = unittest.TextTestRunner(stream=result_stream)

        # Run the tests
        runner.run(suite)

        # Return the captured output
        return result_stream.getvalue()
    else:
        return "No tests were found to run."    

def python(code: str) -> Any:
    """
    Executes the provided Python code after user confirmation. Returns `#FAIL#` if execution is not permitted.

    Args:
        code (str): Python code to be executed.

    Returns:
        Any: Result of executing the code, or `#FAIL#` if not executed.
    """
    # Extract code from Markdown (if present)
    extracted_code = extract_code_from_markdown(code)
    if extracted_code is not None:
        code = extracted_code
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
        del model  # Ensure the model is deleted to free up memory

def run_model_generate_code(model: Any, tokenizer: Any, prompt: str, temp: float, max_tokens: int, ignore_chat_template: bool, formatter: Any) -> str:
    # Apply chat template if available and not ignored
    if not ignore_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    code = generate(model, tokenizer, prompt, temp, max_tokens, True, formatter)
    return code

def get_model_advice_for_error(model: Any, tokenizer: Any, error_message: str, temp: float, max_tokens: int, ignore_chat_template: bool, formatter: Any) -> str:
    advice_prompt = f"Fix these error and give updated code: {error_message}"
    advice = run_model_generate_code(model, tokenizer, advice_prompt, temp, max_tokens, ignore_chat_template, formatter)
    return advice

def combine_code_and_tests(code: str, tests: str) -> str:
    """
    Combines generated Python code and unit tests into a single script.

    Args:
        code (str): Generated Python code.
        tests (str): Generated unit tests.

    Returns:
        str: Combined Python code and unit tests.
    """
    return f"{code}\n\n{tests}"

def main(args):
    mx.random.seed(args.seed)

    # Building tokenizer_config
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token


    with load_model_context(args.model, tokenizer_config) as (model, tokenizer):
        formatter = colorprint_by_t0 if args.colorize else None
        
        while True:
            # Generate Python code
            python_code = run_model_generate_code(model, tokenizer, DEFAULT_CODE_PROMPT, args.temp, args.max_tokens, args.ignore_chat_template, formatter)
            python_code = extract_code_from_markdown(python_code)

            if "Error" in python_code:
                print(python_code)
                break

            # Generate unit tests for the Python code
            unit_test_code = run_model_generate_code(model, tokenizer, DEFAULT_TEST_PROMPT.format(python_code), args.temp, args.max_tokens, args.ignore_chat_template, formatter)
            unit_test_code = extract_code_from_markdown(unit_test_code)

            if "Error" in unit_test_code:
                print(unit_test_code)
                break

            combined_script = combine_code_and_tests(python_code, unit_test_code)
            combined_result = python(combined_script)

            if "FAILED" not in combined_result and "ERROR" not in combined_result:
                print("All tests passed successfully!")
                break

            # Send both the combined script and the test results for better context
            full_context = f"Script:\n{combined_script}\n\nTest Results:\n{combined_result}"
            model_advice =  get_model_advice_for_error(model, tokenizer, full_context, args.temp, args.max_tokens, args.ignore_chat_template, formatter)
            model_advice = extract_code_from_markdown(model_advice)

            apply_advice = input("Apply the above advice? (y/n): ").lower()
            if apply_advice != 'y':
                print("Stopping as per user's decision.")
                break
            
            combined_script = model_advice  # Replace with logic to integrate advice

            # Re-run tests with the updated script
            combined_result = python(combined_script)
            print("Re-run Combined Execution Result:\n", combined_result)

            if "FAILED" not in combined_result and "ERROR" not in combined_result:
                print("All tests passed successfully!")
                break
            else:
                print("Test still failing or error occurred.")
                give_up = input("Would you like to give up? (y/n): ").lower()
                if give_up == 'y':
                    print("Process terminated by user.")
                    break
                else:
                    print("Continuing the process.")
                
if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)

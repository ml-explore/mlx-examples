"""
MLX-LM Benchmark Tool

This script benchmarks the performance of MLX-LM models by generating synthetic
prompt tokens and capturing performance metrics such as:
Model Load Time (s)
Prompt Tokens
Prompt TPS
Response Tokens
Response TPS
Execution Time (s)
Memory Usage (GB)
It supports multiple input values for model, prompt tokens, and generation tokens as what llama.cpp's llama-bench does.
"""

import argparse
import json
import logging
import random
import re
import io
import contextlib
import csv
import time
from typing import Any, Dict, List, Optional, Union

from .utils import load, generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CommaSeparatedIntegers(argparse.Action):
    """
    Custom argparse action to allow comma-separated integers.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        result = []
        for value in values:
            result.extend([int(x) for x in value.split(",") if x])
        setattr(namespace, self.dest, result)


class CommaSeparatedPaths(argparse.Action):
    """
    Custom argparse action to allow comma-separated Paths.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        result = []
        for value in values:
            result.extend([x for x in value.split(",") if x])
        setattr(namespace, self.dest, result)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the benchmark.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark MLX-LM model performance.")

    # Allow multiple models by repeating the -m flag.
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        action=CommaSeparatedPaths,
        required=True,
        help="Path to the MLX model to benchmark. Accepts multiple comma-separated paths.",
    )
    parser.add_argument(
        "-p",
        "--n-prompt",
        nargs="+",
        action=CommaSeparatedIntegers,
        default=[128],
        help="Input Sequence Length (ISL). Number of synthetic prompt tokens. Accepts multiple comma-separated values.",
    )
    parser.add_argument(
        "-n",
        "--n-gen",
        nargs="+",
        action=CommaSeparatedIntegers,
        default=[512],
        help="Outout Sequence Length (OSL). Number of tokens to generate. Accepts multiple comma-separated values.",
    )
    parser.add_argument(
        "-r", "--repetitions", type=int, default=5, help="Number of benchmark repetitions to average results over."
    )
    parser.add_argument(
        "-o",
        "--output-format",
        type=str,
        choices=["csv", "json", "jsonl", "md"],
        default="md",
        help="Output format for the benchmark results.",
    )
    parser.add_argument(
        "-f",
        "--output-filename",
        type=str,
        default="benchmark_results",
        help="Name of the output file, without extension.",
    )
    parser.add_argument(
        "--gen-args",
        type=str,
        nargs="*",
        default=[],
        help="Additional keyword arguments for generate() in key=value format (e.g., --gen-args kv_group_size=64",
    )

    args = parser.parse_args()

    # Convert --gen-args list into a dictionary
    gen_args = {}
    for arg in args.gen_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Optionally, try to convert value to int or float
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # leave it as a string if conversion fails
            gen_args[key] = value
        else:
            parser.error("Invalid format for --gen-args. Expected key=value.")
    args.gen_args = gen_args

    return args


def load_model_tokenizer(model_path: str):
    """
    Load the MLX-LM model and its associated tokenizer.
    Calculate the load time for the model.

    Args:
        model_path (str): Path or identifier of the model.

    Returns:
        Tuple[Any, Any, Any]: The loaded model, tokenizer and model load time.
    """
    start_time = time.time()
    model, tokenizer = load(model_path)
    model_load_time = time.time() - start_time
    return model, tokenizer, model_load_time


def generate_synthetic_tokens(tokenizer: Any, seq_length: int) -> List[int]:
    """
    Generate a synthetic sequence of tokens using the tokenizer's vocabulary.

    Args:
        tokenizer (Any): The tokenizer instance.
        seq_length (int): Desired total number of tokens.

    Returns:
        List[int]: List of token IDs forming the synthetic sequence.
    """
    vocab_size = tokenizer.vocab_size

    # Prepend BOS token if available; otherwise, start with an empty list.
    tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    tokens += [random.randint(0, vocab_size - 1) for _ in range(seq_length - len(tokens))]

    return tokens


def parse_metrics(log_output: str) -> Dict[str, Optional[float]]:
    """
    Parse performance metrics from the log output generated by the `generate()` function.

    Args:
        log_output (str): Captured stdout containing performance logs.

    Returns:
        Dict[str, Optional[float]]: Parsed metrics including:
            - prompt_tokens: Number of prompt tokens processed.
            - prompt_tps: Prompt tokens processed per second.
            - response_tokens: Number of tokens generated.
            - response_tps: Generation tokens processed per second.
            - ram_usage: Peak memory usage in GB.
            - exec_time: Estimated execution time (s).
    """
    metrics: Dict[str, Optional[float]] = {
        "prompt_tokens": None,
        "prompt_tps": None,
        "response_tokens": None,
        "response_tps": None,
        "ram_usage": None,
        "exec_time": None,
    }

    # Extract prompt tokens and tokens-per-second
    prompt_match = re.search(r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", log_output)
    if prompt_match:
        metrics["prompt_tokens"] = int(prompt_match.group(1))
        metrics["prompt_tps"] = float(prompt_match.group(2))

    # Extract generation tokens and tokens-per-second
    generation_match = re.search(r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", log_output)
    if generation_match:
        metrics["response_tokens"] = int(generation_match.group(1))
        metrics["response_tps"] = float(generation_match.group(2))

    # Extract peak memory usage (GB)
    mem_match = re.search(r"Peak memory:\s*([\d.]+)\s*GB", log_output)
    if mem_match:
        metrics["ram_usage"] = float(mem_match.group(1))

    # Calculate total execution time if metrics are available
    if (
        metrics["prompt_tokens"] is not None
        and metrics["prompt_tps"] is not None
        and metrics["response_tokens"] is not None
        and metrics["response_tps"] is not None
    ):
        metrics["exec_time"] = (metrics["prompt_tokens"] / metrics["prompt_tps"]) + (
            metrics["response_tokens"] / metrics["response_tps"]
        )

    return metrics


def benchmark_performance(
    model: Any, tokenizer: Any, seq_length: int, max_tokens: int, **generate_kwargs
) -> Dict[str, Optional[float]]:
    """
    Run a single benchmark iteration, capturing performance metrics.

    Args:
        model (Any): The loaded MLX-LM model.
        tokenizer (Any): The associated tokenizer.
        seq_length (int): Number of synthetic prompt tokens.
        max_tokens (int): Maximum number of tokens to generate.
        **generate_kwargs: Additional keyword arguments for the generate() function.

    Returns:
        Dict[str, Optional[float]]: Performance metrics from this iteration.
    """
    input_tokens = generate_synthetic_tokens(tokenizer, seq_length)
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        generate(model, tokenizer, input_tokens, max_tokens=max_tokens, verbose=True, **generate_kwargs)
    captured_output = output_buffer.getvalue()
    return parse_metrics(captured_output)


def save_results(output_file, results: Union[Dict[str, Any], List[Dict[str, Any]]], output_format: str) -> None:
    """
    Save the benchmark results in the specified output format.

    Args:
        results (Union[Dict[str, Any], List[Dict[str, Any]]]): Benchmark results.
        output_format (str): Format to save the results ("csv", "json", "jsonl", or "md").
    """
    # Ensure results is a list of dictionaries.
    if not isinstance(results, list):
        results = [results]

    if output_format == "json":
        with open(f"{output_file}.json", "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to {output_file}.json")
    elif output_format == "jsonl":
        with open(f"{output_file}.jsonl", "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
        logging.info(f"Results saved to {output_file}.jsonl")
    elif output_format == "csv":
        with open(f"{output_file}.csv", "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                for res in results:
                    writer.writerow(res)
        logging.info(f"Results saved to {output_file}.csv")
    elif output_format == "md":
        if results:
            with open(f"{output_file}.md", "w") as f:
                headers = list(results[0].keys())
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
                for res in results:
                    f.write("| " + " | ".join(str(res[h]) for h in headers) + " |\n")
        logging.info(f"Results saved to {output_file}.md")
    else:
        logging.warning(f"Unsupported output format: {output_format}")


def print_headers(keys):
    """
    Print the header row for markdown table output.

    Args:
        keys (List[str]): List of column names.
    """
    print("| " + " | ".join(keys) + " |")
    print("|" + "|".join(["---"] * len(keys)) + "|")


def print_result_row(result):
    """
    Print a single result row in markdown table format.

    Args:
        result (Dict[str, Any]): Benchmark result data for one test.
    """
    headers = list(result.keys())
    print("| " + " | ".join(str(result[h]) for h in headers) + " |")


def run_benchmarks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Execute the benchmark for each combination of model, prompt tokens, and generation tokens.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        List[Dict[str, Any]]: Aggregated benchmark results.
    """
    all_results = []
    # Print headers only once at the beginning
    result_headers = [
        "Model",
        "Model Load Time (s)",
        "Prompt Tokens",
        "Prompt TPS",
        "Response Tokens",
        "Response TPS",
        "Execution Time (s)",
        "Memory Usage (GB)",
    ]
    print_headers(result_headers)

    for model_path in args.model:
        model, tokenizer, model_load_time = load_model_tokenizer(model_path)
        for n_prompt in args.n_prompt:
            for n_gen in args.n_gen:
                # Warmup run
                _ = benchmark_performance(model, tokenizer, n_prompt, n_gen, **args.gen_args)
                # Benchmark iterations
                metrics_list = []
                for i in range(args.repetitions):
                    metrics = benchmark_performance(model, tokenizer, n_prompt, n_gen, **args.gen_args)
                    metrics_list.append(metrics)
                # Compute average metrics
                avg_metrics = {}
                keys = ["prompt_tokens", "prompt_tps", "response_tokens", "response_tps", "exec_time", "ram_usage"]
                for key in keys:
                    valid_values = [m[key] for m in metrics_list if m[key] is not None]
                    avg_metrics[key] = sum(valid_values) / len(valid_values) if valid_values else None
                result = {
                    "Model": model_path,
                    "Model Load Time (s)": round(model_load_time, 3),
                    "Prompt Tokens": int(avg_metrics["prompt_tokens"]),
                    "Prompt TPS": round(avg_metrics["prompt_tps"], 3),
                    "Response Tokens": int(avg_metrics["response_tokens"]),
                    "Response TPS": round(avg_metrics["response_tps"], 3),
                    "Execution Time (s)": round(avg_metrics["exec_time"], 3),
                    "Memory Usage (GB)": (
                        round(avg_metrics["ram_usage"], 2) if avg_metrics["ram_usage"] is not None else None
                    ),
                }
                # Print the result row immediately after each test completes
                print_result_row(result)
                all_results.append(result)

    # Still save the full results to file if requested
    if args.output_format:
        save_results(args.output_filename, all_results, args.output_format)

    return all_results


def main() -> None:
    """
    Main entry point for the benchmark script.
    """
    args = parse_args()
    run_benchmarks(args)


if __name__ == "__main__":
    main()

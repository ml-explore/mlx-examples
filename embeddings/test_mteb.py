import glob
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Union

from embeddings import EmbeddingModel

try:
    from mteb import MTEB
except ImportError:
    print(
        "MTEB not installed. Please install it with `pip install mteb` to run evaluations."
    )
    print(
        "You'll also need to install `beir` if you want to run retrieval evaluations."
    )
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print(
        "sentence_transformers not installed. Please install it with `pip install sentence_transformers` to run evaluations."
    )
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Task:
    mteb_task: str
    metric: list[str]  # list of keys to extract the metric


class Evaluator:
    def __init__(self, tasks: Union[list[Task], list[dict]]):
        super().__init__()
        if isinstance(tasks[0], dict):
            tasks = [Task(**task) for task in tasks]
        self.tasks = tasks

    def run(self, model: EmbeddingModel):
        results = {}
        for task in self.tasks:
            evaluation = MTEB(tasks=[task.mteb_task], task_langs=["en"])
            eval_splits = ["dev"] if task.mteb_task == "MSMARCO" else ["test"]
            with tempfile.TemporaryDirectory() as tmpdirname:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = evaluation.run(
                        model, output_folder=tmpdirname, eval_splits=eval_splits
                    )[task.mteb_task]
            for key in task.metric:
                result = result[key]
            results[task.mteb_task] = result
        return results


def main(precision_nbits: int = 8):
    tasks = [
        Task(mteb_task="Banking77Classification", metric=["test", "accuracy"]),
        Task(mteb_task="STS12", metric=["test", "cos_sim", "spearman"]),
    ]
    evaluator = Evaluator(tasks)

    # Run the evaluation
    mx_model = EmbeddingModel(
        "BAAI/bge-small-en-v1.5",
        pooling_strategy="cls",
        precision_nbits=precision_nbits,
        normalize=False,
    )
    results = evaluator.run(mx_model)
    print("=== Results for MLX model ===")
    print(results)

    st_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    results = evaluator.run(st_model)
    print("=== Results for SentenceTransformer model ===")
    print(results)


if __name__ == "__main__":
    import sys

    precision_nbits = 8
    if len(sys.argv) > 1:
        precision_nbits = int(sys.argv[1])
    main(precision_nbits=precision_nbits)

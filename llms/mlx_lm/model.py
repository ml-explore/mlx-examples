import argparse
from typing import List, Union

from huggingface_hub import scan_cache_dir
from transformers.commands.user import tabulate


def ask_for_confirmation(message: str) -> bool:
    y = ("y", "yes", "1")
    n = ("n", "no", "0")
    all_values = y + n + ("",)
    full_message = f"{message} (Y/n) "
    while True:
        answer = input(full_message).lower()
        if answer == "":
            return True
        if answer in y:
            return True
        if answer in n:
            return False
        print(f"Invalid input. Must be one of {all_values}")


def main():
    parser = argparse.ArgumentParser(description="MLX Model Cache.")
    parser.add_argument(
        "--scan-models",
        action="store_true",
        help="Scan Hugging Face cache for mlx models.",
    )
    parser.add_argument(
        "--delete-model",
        type=str,
        help="Delete the model from Hugging Face cache.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Get detailed information about a model.",
    )
    args = parser.parse_args()

    if args.scan_models:
        print("Scanning Hugging Face cache for MLX models.")
        hf_cache_info = scan_cache_dir()
        # Keep only models that contain "mlx" in the name
        print(
            tabulate(
                rows=[
                    [
                        repo.repo_id,
                        "{:>12}".format(repo.size_on_disk_str),
                        repo.nb_files,
                        repo.last_accessed_str,
                        repo.last_modified_str,
                        str(repo.repo_path),
                    ]
                    for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path) if "mlx" in repo.repo_id
                ],
                headers=[
                    "REPO ID",
                    "SIZE ON DISK",
                    "NB FILES",
                    "LAST_ACCESSED",
                    "LAST_MODIFIED",
                    "LOCAL PATH",
                ],
            )
        )

    if args.model:
        print(f"Getting detailed information about model: {args.model}")
        hf_cache_info = scan_cache_dir()
        model_info = next((repo for repo in hf_cache_info.repos if repo.repo_id == args.model), None)
        if model_info:
            print(
                tabulate(
                    rows=[
                        [
                            model_info.repo_id,
                            model_info.repo_type,
                            revision.commit_hash,
                            "{:>12}".format(revision.size_on_disk_str),
                            revision.nb_files,
                            revision.last_modified_str,
                            ", ".join(sorted(revision.refs)),
                            str(revision.snapshot_path),
                        ]
                        for revision in sorted(model_info.revisions, key=lambda revision: revision.commit_hash)
                    ],
                    headers=[
                        "REPO ID",
                        "REPO TYPE",
                        "REVISION",
                        "SIZE ON DISK",
                        "NB FILES",
                        "LAST_MODIFIED",
                        "REFS",
                        "LOCAL PATH",
                    ],
                )
            )
        else:
            print(f"No model found with REPO ID: {args.model}")

    if args.delete_model:
        print(f"Deleting model: {args.delete_model}")
        hf_cache_info = scan_cache_dir()
        model_info = next((repo for repo in hf_cache_info.repos if repo.repo_id == args.delete_model), None)
        if model_info:
            confirmed = ask_for_confirmation(f"{model_info.repo_id} Confirm deletion ?")
            if confirmed:
                for revision in sorted(model_info.revisions, key=lambda revision: revision.commit_hash):
                    strategy = hf_cache_info.delete_revisions(revision.commit_hash)
                    strategy.execute()
                    print("Model deleted.")
            else:
                print("Deletion is cancelled. Do nothing.")
        else:
            print(f"No model found with REPO ID: {args.model}")


if __name__ == "__main__":
    main()

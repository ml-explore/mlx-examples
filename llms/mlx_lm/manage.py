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
            return False
        if answer in y:
            return True
        if answer in n:
            return False
        print(f"Invalid input. Must be one of {all_values}")


def main():
    parser = argparse.ArgumentParser(description="MLX Model Cache.")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan Hugging Face cache for mlx models.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete models matching the given pattern.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Model repos contain the pattern.",
        default="mlx",
    )

    args = parser.parse_args()

    if args.scan:
        print(
            "Scanning Hugging Face cache for models with" f'pattern "{args.pattern}".'
        )
        hf_cache_info = scan_cache_dir()
        print(
            tabulate(
                rows=[
                    [
                        repo.repo_id,
                        repo.repo_type,
                        "{:>12}".format(repo.size_on_disk_str),
                        repo.nb_files,
                        repo.last_accessed_str,
                        repo.last_modified_str,
                        str(repo.repo_path),
                    ]
                    for repo in sorted(
                        hf_cache_info.repos, key=lambda repo: repo.repo_path
                    )
                    if args.pattern in repo.repo_id
                ],
                headers=[
                    "REPO ID",
                    "REPO TYPE",
                    "SIZE ON DISK",
                    "NB FILES",
                    "LAST_ACCESSED",
                    "LAST_MODIFIED",
                    "LOCAL PATH",
                ],
            )
        )

    if args.delete:
        print(f'Deleting models matching pattern "{args.pattern}"')
        hf_cache_info = scan_cache_dir()

        repos = [
            repo
            for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path)
            if args.pattern in repo.repo_id
        ]
        if repos:
            print(
                tabulate(
                    rows=[
                        [
                            repo.repo_id,
                            str(repo.repo_path),
                        ]
                        for repo in repos
                    ],
                    headers=[
                        "REPO ID",
                        "LOCAL PATH",
                    ],
                )
            )

            confirmed = ask_for_confirmation(f"Confirm deletion ?")
            if confirmed:
                for model_info in repos:
                    for revision in sorted(
                        model_info.revisions, key=lambda revision: revision.commit_hash
                    ):
                        strategy = hf_cache_info.delete_revisions(revision.commit_hash)
                        strategy.execute()
                print("Model(s) deleted.")
            else:
                print("Deletion is cancelled. Do nothing.")
        else:
            print(f"No models found.")


if __name__ == "__main__":
    main()

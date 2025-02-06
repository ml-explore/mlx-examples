import argparse
from typing import List, Union

from huggingface_hub import scan_cache_dir
from transformers.commands.user import tabulate


def ask_for_confirmation(message: str) -> bool:
    """Ask user for confirmation with Y/N prompt.
    Returns True for Y/yes, False for N/no/empty."""
    y = ("y", "yes", "1")
    n = ("n", "no", "0", "")
    full_message = f"{message} (y/n) "
    while True:
        answer = input(full_message).lower()
        if answer in y:
            return True
        if answer in n:
            return False
        print(f"Invalid input. Must be one of: yes/no/y/n or empty for no")


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
        print(f'Scanning Hugging Face cache for models with pattern "{args.pattern}".')
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
            print("\nFound the following models:")
            print(
                tabulate(
                    rows=[
                        [
                            repo.repo_id,
                            repo.size_on_disk_str,  # Added size information
                            str(repo.repo_path),
                        ]
                        for repo in repos
                    ],
                    headers=[
                        "REPO ID",
                        "SIZE",  # Added size header
                        "LOCAL PATH",
                    ],
                )
            )

            confirmed = ask_for_confirmation(
                "\nAre you sure you want to delete these models?"
            )
            if confirmed:
                for model_info in repos:
                    print(f"\nDeleting {model_info.repo_id}...")
                    for revision in sorted(
                        model_info.revisions, key=lambda revision: revision.commit_hash
                    ):
                        strategy = hf_cache_info.delete_revisions(revision.commit_hash)
                        strategy.execute()
                print("\nModel(s) deleted successfully.")
            else:
                print("\nDeletion cancelled - no changes made.")
        else:
            print(f'No models found matching pattern "{args.pattern}"')


if __name__ == "__main__":
    main()

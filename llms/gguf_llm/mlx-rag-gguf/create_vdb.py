# File: create_vdb.py - Creates a Vector DB from a PDF file

import argparse
from vdb import vdb_from_pdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Vector DB from a PDF file")
    # Input
    parser.add_argument(
        "--pdf",
        help="The path to the input PDF file",
        default="mlx_docs.pdf",
    )
    # Output
    parser.add_argument(
        "--vdb",
        type=str,
        default="vdb.npz",
        help="The path to store the vector DB",
    )
    args = parser.parse_args()
    m = vdb_from_pdf(args.pdf)
    m.savez(args.vdb)
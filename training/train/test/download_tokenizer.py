"""
Download tokenizer from HuggingFace and save locally.

Usage:
    python download_tokenizer.py
    python download_tokenizer.py --tokenizer nvidia/Nemotron-3-8B-Base-4k --output /path/to/save
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Download tokenizer from HuggingFace")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="nvidia/Minitron-4B-Base",
        help="HuggingFace tokenizer name to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/work/m24047/m24047brmn/tokenizers/minitron-4b",
        help="Local path to save the tokenizer",
    )
    args = parser.parse_args()

    print(f"Downloading tokenizer: {args.tokenizer}")
    print(f"Saving to: {args.output}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Download and save tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(args.output)

    print(f"Tokenizer saved to {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()

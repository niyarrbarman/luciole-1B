import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_indexed_dataset(input_path: str, tokenizer_name: str, output_prefix: str, max_tokens: int = 1_000_000):
    """Tokenize plain text and write Megatron-style indexed dataset (.bin/.idx).

    Only the first `max_tokens` (including EOS) are kept.
    """
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    except ImportError:
        # Some Megatron builds expose only MMapIndexedDatasetBuilder; fall back to it.
        from megatron.core.datasets.indexed_dataset import MMapIndexedDatasetBuilder as IndexedDatasetBuilder

    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)
    eos_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None)

    builder = IndexedDatasetBuilder(f"{output_prefix}.bin")
    token_count = 0
    lines_processed = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if hasattr(tokenizer, "text_to_ids"):
                ids = tokenizer.text_to_ids(line)
            else:
                ids = tokenizer.encode(line)
            if eos_id is not None:
                ids = ids + [eos_id]
            token_count += len(ids)
            if token_count > max_tokens:
                excess = token_count - max_tokens
                if excess > 0:
                    ids = ids[:-excess]
                token_count = max_tokens
            tensor = torch.tensor(ids, dtype=torch.int64)
            builder.add_item(tensor)
            if hasattr(builder, "end_document"):
                builder.end_document()
            lines_processed += 1
            if token_count >= max_tokens:
                break

    builder.finalize(f"{output_prefix}.idx")
    logger.info("Wrote dataset: prefix=%s, lines=%s, tokens=%s", output_prefix, lines_processed, token_count)


def main():
    parser = argparse.ArgumentParser(description="Preprocess mock text into indexed dataset (Megatron format)")
    parser.add_argument("--input_txt", required=True, help="Path to plain text input")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--output_prefix", required=True, help="Output prefix (no extension) for bin/idx")
    parser.add_argument("--max_tokens", type=int, default=1_000_000, help="Maximum tokens to keep")
    args = parser.parse_args()

    os.makedirs(Path(args.output_prefix).parent, exist_ok=True)
    build_indexed_dataset(args.input_txt, args.tokenizer, args.output_prefix, args.max_tokens)


if __name__ == "__main__":
    main()

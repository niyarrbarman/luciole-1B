import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize_indexed_dataset(input_prefix: str, tokenizer_name: str, num_tokens: int = 10000):
    """Read Megatron-style indexed dataset (.bin/.idx) and detokenize first N tokens.

    Prints the detokenized text to stdout.
    """
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
    try:
        from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
    except ImportError:
        from megatron.core.datasets.indexed_dataset import IndexedDataset as MMapIndexedDataset

    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)

    # Load the indexed dataset
    dataset = MMapIndexedDataset(input_prefix)
    
    logger.info("Dataset loaded: prefix=%s, num_documents=%d", input_prefix, len(dataset))

    # Collect tokens up to num_tokens
    all_tokens = []
    tokens_collected = 0
    docs_processed = 0

    for doc_idx in range(len(dataset)):
        doc_tokens = dataset[doc_idx]
        if isinstance(doc_tokens, torch.Tensor):
            doc_tokens = doc_tokens.tolist()
        else:
            doc_tokens = list(doc_tokens)
        
        remaining = num_tokens - tokens_collected
        if remaining <= 0:
            break
        
        if len(doc_tokens) > remaining:
            doc_tokens = doc_tokens[:remaining]
        
        all_tokens.extend(doc_tokens)
        tokens_collected += len(doc_tokens)
        docs_processed += 1
        
        if tokens_collected >= num_tokens:
            break

    logger.info("Collected %d tokens from %d documents", tokens_collected, docs_processed)

    # Detokenize
    if hasattr(tokenizer, "ids_to_text"):
        text = tokenizer.ids_to_text(all_tokens)
    elif hasattr(tokenizer, "decode"):
        text = tokenizer.decode(all_tokens)
    else:
        raise AttributeError("Tokenizer has neither 'ids_to_text' nor 'decode' method")

    print("=" * 80)
    print(f"DETOKENIZED TEXT (first {tokens_collected} tokens from {docs_processed} documents):")
    print("=" * 80)
    print(text)
    print("=" * 80)

    return text


def main():
    parser = argparse.ArgumentParser(description="Detokenize Megatron indexed dataset (.bin/.idx)")
    parser.add_argument("--input_prefix", required=True, help="Input prefix (no extension) for bin/idx files")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--num_tokens", type=int, default=10000, help="Number of tokens to detokenize and print")
    args = parser.parse_args()

    detokenize_indexed_dataset(args.input_prefix, args.tokenizer, args.num_tokens)


if __name__ == "__main__":
    main()

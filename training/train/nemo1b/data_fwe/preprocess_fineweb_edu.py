"""
Tokenize FineWeb-Edu dataset into Megatron-style indexed datasets.

Reads JSONL files from FineWeb-Edu and converts them into 
Megatron-style indexed datasets (.bin/.idx).

Usage:
    python preprocess_fineweb_edu.py \
        --tokenizer /work/m24047/m24047brmn/tokenizers/minitron-4b \
        --input_dir /work/shares/IA-Datasets/fineweb_edu_10bt_shuffled \
        --output_prefix /tmpdir/m24047brmn/nemo_1b/data_fwe/fineweb_edu_text_document \
        --max_tokens 1000000000
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def iter_jsonl_files(input_dir: str, pattern: str = "*.jsonl"):
    """Iterate over all JSONL files in a directory, yielding one example at a time."""
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    logger.info(f"Found {len(jsonl_files)} JSONL files: {[os.path.basename(f) for f in jsonl_files]}")
    
    for filepath in jsonl_files:
        logger.info(f"Processing file: {os.path.basename(filepath)}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def tokenize_dataset(
    tokenizer_path: str,
    input_dir: str,
    output_prefix: str,
    max_tokens: int = 1_000_000_000,
    file_pattern: str = "*.chunk.*.jsonl",
):
    """Tokenize FineWeb-Edu JSONL files into Megatron indexed dataset format.
    
    Args:
        tokenizer_path: Path to the tokenizer (local or HuggingFace name)
        input_dir: Directory containing JSONL files
        output_prefix: Output prefix for .bin/.idx files
        max_tokens: Maximum number of tokens to process (default: 1B)
        file_pattern: Glob pattern to match JSONL files
    """
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    except ImportError:
        from megatron.core.datasets.indexed_dataset import MMapIndexedDatasetBuilder as IndexedDatasetBuilder

    # Load tokenizer - support both NeMo/HuggingFace and raw SentencePiece
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    
    # Check if this is a SentencePiece model directory
    sp_model_path = os.path.join(tokenizer_path, "tokenizer.model")
    if os.path.isdir(tokenizer_path) and os.path.exists(sp_model_path):
        # Load SentencePiece directly
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_model_path)
        logger.info(f"Loaded SentencePiece tokenizer with vocab size: {sp.GetPieceSize()}")
        
        # Wrap in a simple class for compatibility
        class SPWrapper:
            def __init__(self, sp_processor):
                self.sp = sp_processor
                self.eos_id = sp_processor.eos_id()
                self.bos_id = sp_processor.bos_id()
            
            def text_to_ids(self, text):
                return self.sp.EncodeAsIds(text)
            
            def encode(self, text):
                return self.sp.EncodeAsIds(text)
        
        tokenizer = SPWrapper(sp)
        eos_id = tokenizer.eos_id
    else:
        # Try NeMo/HuggingFace tokenizer
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
        tokenizer = get_tokenizer(tokenizer_name=tokenizer_path, use_fast=True)
        eos_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None)
    
    logger.info(f"EOS token ID: {eos_id}")

    # Create output directory
    os.makedirs(Path(output_prefix).parent, exist_ok=True)

    # Iterate over JSONL files
    logger.info(f"Reading JSONL files from: {input_dir}")

    # Build indexed dataset
    builder = IndexedDatasetBuilder(f"{output_prefix}.bin")
    token_count = 0
    docs_processed = 0
    
    logger.info(f"Starting tokenization (target: {max_tokens:,} tokens)...")

    try:
        for example in iter_jsonl_files(input_dir, file_pattern):
            text = example.get("text", "")
            if not text.strip():
                continue

            # Tokenize
            if hasattr(tokenizer, "text_to_ids"):
                ids = tokenizer.text_to_ids(text)
            else:
                ids = tokenizer.encode(text)

            # Add EOS token
            if eos_id is not None:
                ids = ids + [eos_id]

            # Check if we've reached the token limit
            if token_count + len(ids) > max_tokens:
                # Truncate to fit exactly
                remaining = max_tokens - token_count
                if remaining > 0:
                    ids = ids[:remaining]
                else:
                    break

            # Add to indexed dataset
            tensor = torch.tensor(ids, dtype=torch.int64)
            builder.add_item(tensor)
            if hasattr(builder, "end_document"):
                builder.end_document()

            token_count += len(ids)
            docs_processed += 1

            # Progress logging
            if docs_processed % 10000 == 0:
                logger.info(f"Progress: {docs_processed:,} docs, {token_count:,} tokens ({100*token_count/max_tokens:.2f}%)")

            if token_count >= max_tokens:
                break

    except KeyboardInterrupt:
        logger.warning("Interrupted by user, finalizing dataset...")

    # Finalize
    builder.finalize(f"{output_prefix}.idx")
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"  Output prefix: {output_prefix}")
    logger.info(f"  Documents: {docs_processed:,}")
    logger.info(f"  Tokens: {token_count:,}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu JSONL files into Megatron indexed dataset")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/work/m24047/m24047brmn/tokenizers/minitron-4b",
        help="Path to tokenizer (local or HuggingFace name)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/work/shares/IA-Datasets/fineweb_edu_10bt_shuffled",
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="/tmpdir/m24047brmn/nemo_1b/data_fwe/fineweb_edu_text_document",
        help="Output prefix for .bin/.idx files",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1_000_000_000,
        help="Maximum tokens to process (default: 1B)",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.chunk.*.jsonl",
        help="Glob pattern to match JSONL files (default: *.chunk.*.jsonl)",
    )
    args = parser.parse_args()

    tokenize_dataset(
        tokenizer_path=args.tokenizer,
        input_dir=args.input_dir,
        output_prefix=args.output_prefix,
        max_tokens=args.max_tokens,
        file_pattern=args.file_pattern,
    )


if __name__ == "__main__":
    main()

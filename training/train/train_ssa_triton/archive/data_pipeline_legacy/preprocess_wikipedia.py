"""
Tokenize Wikipedia-EN dataset into Megatron-style indexed datasets.

Reads parquet files from Wikipedia-EN and converts them into 
Megatron-style indexed datasets (.bin/.idx).

Usage:
    python preprocess_wikipedia.py \
        --tokenizer /work/m24047/m24047brmn/tokenizers/luciole_50k \
        --input_dir /work/shares/IA-Datasets/Wikipedia-EN/wikipedia/data/20220301.en \
        --output_prefix /tmpdir/m24047brmn/nemo_1b/data_wiki/wikipedia_en_text_document \
        --max_tokens 10000000
"""

import argparse
import glob
import logging
import os
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def iter_huggingface_wikipedia(language: str = "en", date: str = "20220301"):
    """Stream Wikipedia from HuggingFace datasets."""
    from datasets import load_dataset
    
    config_name = f"{date}.{language}"
    logger.info(f"Streaming Wikipedia from HuggingFace: wikipedia/{config_name}")
    
    ds = load_dataset("wikipedia", config_name, split="train", streaming=True, trust_remote_code=True)
    
    for example in ds:
        yield example


def tokenize_dataset(
    tokenizer_path: str,
    output_prefix: str,
    max_tokens: int = 10_000_000,
    language: str = "en",
    date: str = "20220301",
    text_column: str = "text",
):
    """Tokenize Wikipedia from HuggingFace into Megatron indexed dataset format.
    
    Args:
        tokenizer_path: Path to the tokenizer (local or HuggingFace name)
        output_prefix: Output prefix for .bin/.idx files
        max_tokens: Maximum number of tokens to process (default: 10M)
        language: Wikipedia language code (default: en)
        date: Wikipedia dump date (default: 20220301)
        text_column: Column name containing the text
    """
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    except ImportError:
        from megatron.core.datasets.indexed_dataset import MMapIndexedDatasetBuilder as IndexedDatasetBuilder

    # Load tokenizer
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    
    # Check if this is a SentencePiece model directory
    sp_model_path = os.path.join(tokenizer_path, "tokenizer.model")
    if os.path.isdir(tokenizer_path) and os.path.exists(sp_model_path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_model_path)
        logger.info(f"Loaded SentencePiece tokenizer with vocab size: {sp.GetPieceSize()}")
        
        class SPWrapper:
            def __init__(self, sp_processor):
                self.sp = sp_processor
                self.eos_id = sp_processor.eos_id()
                self.bos_id = sp_processor.bos_id()
            
            def text_to_ids(self, text):
                return self.sp.EncodeAsIds(text)
        
        tokenizer = SPWrapper(sp)
        eos_id = tokenizer.eos_id
    else:
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
        tokenizer = get_tokenizer(tokenizer_name=tokenizer_path, use_fast=True)
        eos_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None)
    
    logger.info(f"EOS token ID: {eos_id}")

    # Create output directory
    os.makedirs(Path(output_prefix).parent, exist_ok=True)

    # Build indexed dataset
    builder = IndexedDatasetBuilder(f"{output_prefix}.bin")
    token_count = 0
    docs_processed = 0
    
    logger.info(f"Starting tokenization (target: {max_tokens:,} tokens)...")

    try:
        for example in iter_huggingface_wikipedia(language=language, date=date):
            text = example.get(text_column, "")
            if not text or not text.strip():
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
            if docs_processed % 5000 == 0:
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
    parser = argparse.ArgumentParser(description="Tokenize Wikipedia from HuggingFace into Megatron indexed dataset")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/work/m24047/m24047brmn/tokenizers/luciole_50k",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="/tmpdir/m24047brmn/nemo_1b/data_wiki/wikipedia_en_text_document",
        help="Output prefix for .bin/.idx files",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10_000_000,
        help="Maximum tokens to process (default: 10M)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Wikipedia language code (default: en)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="20220301",
        help="Wikipedia dump date (default: 20220301)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name containing the text",
    )
    args = parser.parse_args()

    tokenize_dataset(
        tokenizer_path=args.tokenizer,
        output_prefix=args.output_prefix,
        max_tokens=args.max_tokens,
        language=args.language,
        date=args.date,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()

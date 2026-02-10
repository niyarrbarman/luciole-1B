"""
Train a SentencePiece tokenizer on FineWeb-Edu dataset.

Creates a tokenizer similar to Minitron but with ~50k vocab size.
Output format is compatible with NeMo/Megatron.

Usage:
    python train_tokenizer.py \
        --input_dir /work/shares/IA-Datasets/fineweb_edu_10bt_shuffled \
        --output_dir /work/m24047/m24047brmn/tokenizers/luciole_50k \
        --vocab_size 50000 \
        --num_threads 80
"""

import argparse
import glob
import json
import logging
import os
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_text_from_jsonl(input_dir: str, output_file: str, max_bytes: int = 50_000_000_000, file_pattern: str = "*.jsonl"):
    """Extract text from JSONL files into a single text file for tokenizer training.
    
    Args:
        input_dir: Directory containing JSONL files
        output_file: Path to output text file
        max_bytes: Maximum bytes to extract (default: 50GB)
        file_pattern: Glob pattern for JSONL files
    """
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, "**", file_pattern), recursive=True))
    if not jsonl_files:
        # Try non-recursive
        jsonl_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    
    logger.info(f"Found {len(jsonl_files)} JSONL files")
    
    total_bytes = 0
    docs_count = 0
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for filepath in jsonl_files:
            logger.info(f"Reading: {os.path.basename(filepath)}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            example = json.loads(line)
                            text = example.get("text", "")
                            if text.strip():
                                out_f.write(text + "\n")
                                total_bytes += len(text.encode("utf-8")) + 1
                                docs_count += 1
                                
                                if docs_count % 100000 == 0:
                                    logger.info(f"  Extracted {docs_count:,} docs, {total_bytes / 1e9:.2f} GB")
                                
                                if total_bytes >= max_bytes:
                                    logger.info(f"Reached max_bytes limit ({max_bytes / 1e9:.1f} GB)")
                                    return total_bytes, docs_count
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")
                continue
    
    return total_bytes, docs_count


def train_sentencepiece_tokenizer(
    text_file: str,
    output_dir: str,
    model_prefix: str = "tokenizer",
    vocab_size: int = 50000,
    num_threads: int = 80,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
):
    """Train a SentencePiece tokenizer.
    
    Args:
        text_file: Path to input text file
        output_dir: Directory to save tokenizer files
        model_prefix: Prefix for output files
        vocab_size: Vocabulary size
        num_threads: Number of CPU threads
        character_coverage: Character coverage (0.9995 for multilingual)
        model_type: Model type ('bpe' or 'unigram')
    """
    import sentencepiece as spm
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, model_prefix)
    
    logger.info(f"Training SentencePiece tokenizer:")
    logger.info(f"  Input: {text_file}")
    logger.info(f"  Output: {model_path}.model")
    logger.info(f"  Vocab size: {vocab_size}")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Threads: {num_threads}")
    
    # Train tokenizer with settings similar to Minitron/Nemotron
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        # Minitron-style settings
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        input_sentence_size=50_000_000,  # Use 50M sentences max
        # Special tokens (NeMo/Megatron compatible)
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        # Byte fallback for unknown chars
        byte_fallback=True,
        # Normalization
        normalization_rule_name="identity",  # No normalization
        remove_extra_whitespaces=False,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        split_digits=True,
        # Control tokens (reserved for future use)
        control_symbols=["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<extra_id_3>"],
    )
    
    logger.info(f"Tokenizer training complete!")
    logger.info(f"  Model: {model_path}.model")
    logger.info(f"  Vocab: {model_path}.vocab")
    
    # Create a tokenizer_config.json for HuggingFace compatibility
    config = {
        "tokenizer_class": "SentencePieceTokenizer",
        "vocab_size": vocab_size,
        "model_type": model_type,
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token_id": 0,
        "unk_token_id": 1,
        "bos_token_id": 2,
        "eos_token_id": 3,
    }
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"  Config: {config_path}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer on FineWeb-Edu")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/work/shares/IA-Datasets/fineweb_edu_10bt_shuffled",
        help="Directory containing JSONL files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/work/m24047/m24047brmn/tokenizers/luciole_50k",
        help="Directory to save tokenizer files",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50000,
        help="Vocabulary size (default: 50000)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=80,
        help="Number of CPU threads (default: 80)",
    )
    parser.add_argument(
        "--max_gb",
        type=float,
        default=50.0,
        help="Maximum GB of text to use for training (default: 50)",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.chunk.*.jsonl",
        help="Glob pattern for JSONL files",
    )
    parser.add_argument(
        "--keep_text_file",
        action="store_true",
        help="Keep the extracted text file after training",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract text to temporary file
    if args.keep_text_file:
        text_file = os.path.join(args.output_dir, "training_corpus.txt")
    else:
        text_file = os.path.join(args.output_dir, "_temp_corpus.txt")
    
    logger.info("=" * 60)
    logger.info("Step 1: Extracting text from JSONL files...")
    logger.info("=" * 60)
    
    max_bytes = int(args.max_gb * 1e9)
    total_bytes, docs_count = extract_text_from_jsonl(
        input_dir=args.input_dir,
        output_file=text_file,
        max_bytes=max_bytes,
        file_pattern=args.file_pattern,
    )
    logger.info(f"Extracted {docs_count:,} documents, {total_bytes / 1e9:.2f} GB")
    
    logger.info("=" * 60)
    logger.info("Step 2: Training SentencePiece tokenizer...")
    logger.info("=" * 60)
    
    train_sentencepiece_tokenizer(
        text_file=text_file,
        output_dir=args.output_dir,
        model_prefix="tokenizer",
        vocab_size=args.vocab_size,
        num_threads=args.num_threads,
    )
    
    # Cleanup temp file
    if not args.keep_text_file and os.path.exists(text_file):
        os.remove(text_file)
        logger.info(f"Cleaned up temporary text file")
    
    logger.info("=" * 60)
    logger.info("Done! Tokenizer files:")
    logger.info(f"  {args.output_dir}/tokenizer.model")
    logger.info(f"  {args.output_dir}/tokenizer.vocab")
    logger.info(f"  {args.output_dir}/tokenizer_config.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

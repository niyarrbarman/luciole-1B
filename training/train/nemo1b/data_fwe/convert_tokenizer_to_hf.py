"""
Convert SentencePiece tokenizer to HuggingFace format.

This creates a HuggingFace-compatible tokenizer that NeMo can load.
"""

import argparse
import os
import json
import shutil


def convert_sp_to_hf(input_dir: str, output_dir: str = None):
    """Convert SentencePiece tokenizer to HuggingFace LlamaTokenizer format.
    
    Args:
        input_dir: Directory containing tokenizer.model
        output_dir: Output directory (default: same as input with _hf suffix or in-place)
    """
    from transformers import LlamaTokenizerFast, LlamaTokenizer
    
    sp_model_path = os.path.join(input_dir, "tokenizer.model")
    if not os.path.exists(sp_model_path):
        raise FileNotFoundError(f"No tokenizer.model found in {input_dir}")
    
    if output_dir is None:
        output_dir = input_dir  # Convert in-place
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading SentencePiece model from: {sp_model_path}")
    
    # Use LlamaTokenizer which is designed for SentencePiece models
    # This is compatible with most decoder-only models
    try:
        tokenizer = LlamaTokenizer(
            vocab_file=sp_model_path,
            legacy=False,
            add_bos_token=False,  # We add BOS/EOS manually in preprocessing
            add_eos_token=False,
        )
    except Exception as e:
        print(f"LlamaTokenizer failed: {e}")
        print("Trying with legacy=True...")
        tokenizer = LlamaTokenizer(
            vocab_file=sp_model_path,
            legacy=True,
            add_bos_token=False,
            add_eos_token=False,
        )
    
    # Set special tokens
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: pad={tokenizer.pad_token_id}, unk={tokenizer.unk_token_id}, bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}")
    
    # Save in HuggingFace format
    tokenizer.save_pretrained(output_dir)
    
    # Copy the original .model and .vocab files too for reference
    if output_dir != input_dir:
        vocab_path = os.path.join(input_dir, "tokenizer.vocab")
        if os.path.exists(vocab_path):
            shutil.copy(vocab_path, os.path.join(output_dir, "tokenizer.vocab"))
    
    print(f"\nConverted tokenizer saved to: {output_dir}")
    print(f"Files created:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")
    
    # Quick test
    print(f"\nQuick test:")
    test_text = "Hello, world! This is a test."
    tokens = tokenizer.encode(test_text)
    print(f"  Input: {test_text}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: {tokenizer.decode(tokens)}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert SentencePiece tokenizer to HuggingFace format")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/work/m24047/m24047brmn/tokenizers/luciole_50k",
        help="Directory containing tokenizer.model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: convert in-place)",
    )
    args = parser.parse_args()
    
    convert_sp_to_hf(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

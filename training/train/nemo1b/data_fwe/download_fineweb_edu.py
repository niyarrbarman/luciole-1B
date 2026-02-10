"""
Download FineWeb-Edu dataset from HuggingFace.

Downloads the dataset and saves it locally for later preprocessing.

Usage:
    python download_fineweb_edu.py \
        --output_dir /tmpdir/m24047brmn/nemo_1b/data_fwe/raw \
        --cache_dir /tmpdir/m24047brmn/hf_cache
"""

import argparse
import logging
import os
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(
    output_dir: str,
    cache_dir: str = None,
    streaming: bool = False,
    num_shards: int = 10,
):
    """Download FineWeb-Edu dataset and save locally.
    
    Args:
        output_dir: Directory to save the downloaded dataset
        cache_dir: HuggingFace cache directory
        streaming: Whether to use streaming mode (set False to download fully)
        num_shards: Number of shards to split the dataset into
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load FineWeb-Edu dataset
    logger.info("Loading FineWeb-Edu dataset from HuggingFace...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # Use the 10B token sample for faster download
        split="train",
        streaming=streaming,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    logger.info("Dataset loaded successfully")

    # Save to disk
    output_path = Path(output_dir) / "fineweb_edu_sample_10BT"
    logger.info(f"Saving dataset to: {output_path}")
    
    if streaming:
        # If streaming, convert to regular dataset first (iterate and save)
        logger.warning("Streaming mode - this may take a while to materialize the dataset...")
        # For streaming datasets, we need to iterate and save in chunks
        from datasets import Dataset
        
        batch = []
        batch_idx = 0
        for i, example in enumerate(dataset):
            batch.append(example)
            if len(batch) >= 100000:  # Save every 100k examples
                batch_dataset = Dataset.from_list(batch)
                shard_path = output_path / f"shard_{batch_idx:04d}"
                batch_dataset.save_to_disk(str(shard_path))
                logger.info(f"Saved shard {batch_idx} with {len(batch)} examples")
                batch = []
                batch_idx += 1
        
        # Save remaining
        if batch:
            batch_dataset = Dataset.from_list(batch)
            shard_path = output_path / f"shard_{batch_idx:04d}"
            batch_dataset.save_to_disk(str(shard_path))
            logger.info(f"Saved final shard {batch_idx} with {len(batch)} examples")
    else:
        # Regular dataset - save directly
        dataset.save_to_disk(str(output_path), num_shards=num_shards)
    
    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info(f"  Output directory: {output_path}")
    logger.info("=" * 60)
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmpdir/m24047brmn/nemo_1b/data_fwe/raw",
        help="Directory to save the downloaded dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmpdir/m24047brmn/hf_cache",
        help="HuggingFace cache directory for dataset download",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (slower but uses less memory)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=10,
        help="Number of shards to split the dataset into",
    )
    args = parser.parse_args()

    download_dataset(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()

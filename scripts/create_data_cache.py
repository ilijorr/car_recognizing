#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import pickle
import json
from tqdm import tqdm

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset import CarDataModule

def create_dataset_cache(data_path: str, cache_dir: str = "cache"):
    """
    Create and save dataset cache for fast loading

    Args:
        data_path: Path to raw dataset
        cache_dir: Directory to save cache files
    """

    print("🔄 Creating dataset cache (this will take a while, but only once)...")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Create data module (slow first time)
    print("Loading and processing dataset...")
    data_module = CarDataModule(
        data_path=data_path,
        batch_size=32,  # Doesn't matter for caching
        image_size=224, # Doesn't matter for caching
        num_workers=0   # Single thread for stability
    )

    # Save the splits and encoders
    cache_data = {
        'train_samples': data_module.train_samples,
        'val_samples': data_module.val_samples,
        'test_samples': data_module.test_samples,
        'make_encoder': data_module.make_encoder,
        'model_encoder': data_module.model_encoder,
        'year_encoder': data_module.year_encoder,
        'num_makes': data_module.num_makes,
        'num_models': data_module.num_models,
        'num_years': data_module.num_years
    }

    # Save to cache
    cache_file = os.path.join(cache_dir, "dataset_cache.pkl")
    print(f"💾 Saving cache to {cache_file}...")

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    # Save metadata for inspection
    metadata = {
        'total_samples': len(data_module.train_samples) + len(data_module.val_samples) + len(data_module.test_samples),
        'train_samples': len(data_module.train_samples),
        'val_samples': len(data_module.val_samples),
        'test_samples': len(data_module.test_samples),
        'num_makes': data_module.num_makes,
        'num_models': data_module.num_models,
        'num_years': data_module.num_years,
        'make_classes': data_module.make_encoder.classes_.tolist(),
        'year_classes': data_module.year_encoder.classes_.tolist()
    }

    metadata_file = os.path.join(cache_dir, "dataset_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Cache created successfully!")
    print(f"📊 Dataset summary:")
    print(f"  Total samples: {metadata['total_samples']:,}")
    print(f"  Train/Val/Test: {metadata['train_samples']:,} / {metadata['val_samples']:,} / {metadata['test_samples']:,}")
    print(f"  Classes: {metadata['num_makes']} makes, {metadata['num_models']} models, {metadata['num_years']} years")
    print(f"\n💡 Now use CachedCarDataModule for instant loading!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create dataset cache')
    parser.add_argument('--data_path', type=str, default='data/raw',
                       help='Path to raw dataset')
    parser.add_argument('--cache_dir', type=str, default='cache',
                       help='Directory to save cache')

    args = parser.parse_args()

    create_dataset_cache(args.data_path, args.cache_dir)
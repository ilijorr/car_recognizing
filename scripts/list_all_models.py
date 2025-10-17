#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import pickle
from collections import Counter

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

def list_all_models(cache_file: str = "cache/dataset_cache.pkl"):
    """
    List all unique models in the dataset with their sample counts
    """

    if not os.path.exists(cache_file):
        print(f"❌ Cache file {cache_file} not found.")
        print("💡 Run 'python scripts/create_data_cache.py' first.")
        return

    print(f"📋 Loading models from {cache_file}...")

    # Load cached data
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)

    # Extract all samples
    all_samples = (cache_data['train_samples'] +
                  cache_data['val_samples'] +
                  cache_data['test_samples'])

    # Count models
    model_counts = Counter(s['model'] for s in all_samples)

    print(f"\n📊 Found {len(model_counts)} unique models:")
    print("=" * 80)

    # Sort by count (descending) then by name
    for model, count in sorted(model_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{model:<30} : {count:>6} samples")

    print("=" * 80)
    print(f"Total models: {len(model_counts)}")

    # Show models with very few samples
    few_samples = {model: count for model, count in model_counts.items() if count < 10}
    print(f"\nModels with <10 samples ({len(few_samples)}):")
    for model, count in sorted(few_samples.items(), key=lambda x: x[1]):
        print(f"  {model:<30} : {count} samples")

if __name__ == "__main__":
    list_all_models()
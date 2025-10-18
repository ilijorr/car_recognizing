#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import pickle
import json
from collections import Counter
import numpy as np
from tqdm import tqdm

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

def check_cache_quality(cache_file: str = "cache/dataset_cache.pkl"):
    """
    Check data quality issues in the cached dataset

    Args:
        cache_file: Path to cached dataset file
    """

    if not os.path.exists(cache_file):
        print(f"❌ Cache file {cache_file} not found.")
        print("💡 Run 'python scripts/create_data_cache.py' first.")
        return

    print(f"🔍 Checking data quality in {cache_file}...")

    # Load cached data
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)

    # Extract data
    all_samples = (cache_data['train_samples'] +
                  cache_data['val_samples'] +
                  cache_data['test_samples'])

    make_encoder = cache_data['make_encoder']
    model_encoder = cache_data['model_encoder']
    year_encoder = cache_data['year_encoder']

    print(f"📊 Total samples: {len(all_samples):,}")
    print(f"📊 Classes: {len(make_encoder.classes_)} makes, {len(model_encoder.classes_)} models, {len(year_encoder.classes_)} years")

    # Show unique counts
    unique_makes = len(set(s['make'] for s in all_samples))
    unique_models = len(set(s['model'] for s in all_samples))
    unique_years = len(set(s['year'] for s in all_samples))
    print(f"📊 Unique values: {unique_makes} makes, {unique_models} models, {unique_years} years")

    # Check for data quality issues
    print("\n" + "="*60)
    print("🔍 DATA QUALITY CHECKS")
    print("="*60)

    # 1. Check for missing/corrupt image paths
    print("\n1️⃣ Checking image file existence...")
    missing_files = []
    corrupt_paths = []

    for i, sample in enumerate(tqdm(all_samples[:1000], desc="Checking files")):  # Check first 1000
        image_path = sample['image_path']

        # Check if path exists
        if not os.path.exists(image_path):
            missing_files.append(image_path)

        # Check for weird characters
        if any(char in str(image_path) for char in ['?', '*', '<', '>', '|']):
            corrupt_paths.append(image_path)

    print(f"   Missing files (first 1000): {len(missing_files)}")
    print(f"   Corrupt paths (first 1000): {len(corrupt_paths)}")

    # 2. Check for NaN/None values in labels
    print("\n2️⃣ Checking for NaN/None values...")
    nan_makes = sum(1 for s in all_samples if s['make'] in [None, '', 'nan', 'NaN'])
    nan_models = sum(1 for s in all_samples if s['model'] in [None, '', 'nan', 'NaN'])
    nan_years = sum(1 for s in all_samples if s['year'] in [None, '', 'nan', 'NaN'])

    print(f"   NaN makes: {nan_makes}")
    print(f"   NaN models: {nan_models}")
    print(f"   NaN years: {nan_years}")

    # 3. Check for UNKNOWN values
    print("\n3️⃣ Checking for UNKNOWN labels...")
    unknown_makes = sum(1 for s in all_samples if s['make'] == 'UNKNOWN')
    unknown_models = sum(1 for s in all_samples if s['model'] == 'UNKNOWN')
    unknown_years = sum(1 for s in all_samples if s['year'] == 'Unknown')

    print(f"   UNKNOWN makes: {unknown_makes} ({unknown_makes/len(all_samples)*100:.1f}%)")
    print(f"   UNKNOWN models: {unknown_models} ({unknown_models/len(all_samples)*100:.1f}%)")
    print(f"   Unknown years: {unknown_years} ({unknown_years/len(all_samples)*100:.1f}%)")

    # 4. Check encoding integrity
    print("\n4️⃣ Checking label encoding integrity...")
    encoding_errors = 0

    for sample in all_samples[:1000]:  # Check first 1000
        try:
            # Check if encoded values are within bounds
            if 'make_encoded' in sample:
                if sample['make_encoded'] >= len(make_encoder.classes_):
                    encoding_errors += 1
            if 'model_encoded' in sample:
                if sample['model_encoded'] >= len(model_encoder.classes_):
                    encoding_errors += 1
            if 'year_encoded' in sample:
                if sample['year_encoded'] >= len(year_encoder.classes_):
                    encoding_errors += 1
        except:
            encoding_errors += 1

    print(f"   Encoding errors (first 1000): {encoding_errors}")

    # 5. Check for data leakage between splits
    print("\n5️⃣ Checking for data leakage...")
    train_combos = set()
    val_combos = set()
    test_combos = set()

    for sample in cache_data['train_samples']:
        train_combos.add(f"{sample['make']}_{sample['model']}")

    for sample in cache_data['val_samples']:
        val_combos.add(f"{sample['make']}_{sample['model']}")

    for sample in cache_data['test_samples']:
        test_combos.add(f"{sample['make']}_{sample['model']}")

    train_val_overlap = train_combos & val_combos
    train_test_overlap = train_combos & test_combos
    val_test_overlap = val_combos & test_combos

    print(f"   Train-Val overlap: {len(train_val_overlap)} combinations")
    print(f"   Train-Test overlap: {len(train_test_overlap)} combinations")
    print(f"   Val-Test overlap: {len(val_test_overlap)} combinations")

    # 6. Check class distribution extremes
    print("\n6️⃣ Checking class distribution extremes...")

    # Make distribution
    make_counts = Counter(s['make'] for s in all_samples)
    model_counts = Counter(s['model'] for s in all_samples)
    year_counts = Counter(s['year'] for s in all_samples)

    print(f"\n   📈 Make distribution:")
    print(f"      Most common: {make_counts.most_common(1)[0]} samples")
    print(f"      Least common: {make_counts.most_common()[-1]} samples")
    print(f"      Makes with <10 samples: {sum(1 for count in make_counts.values() if count < 10)}")

    print(f"\n   📈 Model distribution:")
    print(f"      Most common: {model_counts.most_common(1)[0]} samples")
    print(f"      Least common: {model_counts.most_common()[-1]} samples")
    print(f"      Models with <5 samples: {sum(1 for count in model_counts.values() if count < 5)}")
    print(f"      Models with <10 samples: {sum(1 for count in model_counts.values() if count < 10)}")

    print(f"\n   📈 Year distribution:")
    print(f"      Most common: {year_counts.most_common(1)[0]} samples")
    print(f"      Least common: {year_counts.most_common()[-1]} samples")

    # 7. Summary and recommendations
    print("\n" + "="*60)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*60)

    total_issues = (len(missing_files) + len(corrupt_paths) + nan_makes +
                   nan_models + nan_years + unknown_makes + unknown_models +
                   unknown_years + encoding_errors)

    if total_issues == 0:
        print("✅ No major data quality issues found!")
    else:
        print(f"⚠️  Found {total_issues} potential data quality issues")

    # Model class recommendations
    models_with_few_samples = sum(1 for count in model_counts.values() if count < 10)
    if models_with_few_samples > 100:
        print(f"💡 RECOMMENDATION: {models_with_few_samples} models have <10 samples")
        print("   Consider model consolidation (A4, A4_Avant → A4)")

    # Data leakage check
    if len(train_val_overlap) > 0 or len(train_test_overlap) > 0:
        print("⚠️  Data leakage detected! This could explain poor validation performance.")

    if unknown_makes > len(all_samples) * 0.1:  # More than 10% unknown
        print("⚠️  High percentage of UNKNOWN makes - check brand recognition")

    return {
        'total_samples': len(all_samples),
        'missing_files': len(missing_files),
        'unknown_makes': unknown_makes,
        'unknown_models': unknown_models,
        'models_with_few_samples': models_with_few_samples,
        'data_leakage': len(train_val_overlap) + len(train_test_overlap) > 0
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check dataset quality')
    parser.add_argument('--cache_file', type=str, default='cache/dataset_cache.pkl',
                       help='Path to cache file')

    args = parser.parse_args()
    check_cache_quality(args.cache_file)

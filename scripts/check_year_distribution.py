#!/usr/bin/env python3

import sys
from pathlib import Path
from collections import Counter

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from utils.folder_normalizer import CarFolderNormalizer

    print("Checking year distribution for entire dataset...")

    # Use the normalizer directly on directory names (much faster)
    data_path = Path("data/raw")
    if not data_path.exists():
        print(f"Data path {data_path} not found. Please check your dataset location.")
        exit(1)

    normalizer = CarFolderNormalizer(str(data_path))

    makes = []
    models = []
    years = []
    make_models = []
    total_images = 0
    directory_count = 0

    print("Scanning directories...")

    # Check all directories
    for dir_path in data_path.iterdir():
        if not dir_path.is_dir():
            continue

        directory_count += 1

        # Extract car info using our updated parser
        make, model, year = normalizer.extract_car_info(dir_path.name)

        # Count images in this directory
        image_count = len(list(dir_path.glob('*.jpg')))
        total_images += image_count

        # Add labels for each image (to get proper distribution)
        makes.extend([make] * image_count)
        models.extend([model] * image_count)
        years.extend([year] * image_count)
        make_models.extend([f"{make}_{model}"] * image_count)

    # Count distributions
    make_counts = Counter(makes)
    model_counts = Counter(models)
    year_counts = Counter(years)
    make_model_counts = Counter(make_models)

    print(f"\n📊 Complete Dataset Analysis:")
    print("=" * 60)
    print(f"Total images: {total_images:,}")
    print(f"Total directories: {directory_count:,}")
    print(f"Unique makes: {len(make_counts)}")
    print(f"Unique models: {len(model_counts)}")
    print(f"Unique make-model combinations: {len(make_model_counts)}")
    print(f"Unique year categories: {len(year_counts)}")

    # Top Makes
    print(f"\n🚗 Top 10 Makes:")
    print("-" * 30)
    for make, count in make_counts.most_common(10):
        percentage = (count / total_images) * 100
        print(f"  {make:15}: {count:6,} images ({percentage:5.1f}%)")

    # Top Models
    print(f"\n🔧 Top 10 Models:")
    print("-" * 30)
    for model, count in model_counts.most_common(10):
        percentage = (count / total_images) * 100
        print(f"  {model:15}: {count:6,} images ({percentage:5.1f}%)")

    # Year Distribution
    print(f"\n📅 Year Category Distribution:")
    print("-" * 35)
    for year, count in sorted(year_counts.items()):
        percentage = (count / total_images) * 100
        print(f"  {year:10}: {count:6,} images ({percentage:5.1f}%)")

    # Top Make-Model Combinations
    print(f"\n🎯 Top 10 Make-Model Combinations:")
    print("-" * 40)
    for make_model, count in make_model_counts.most_common(10):
        percentage = (count / total_images) * 100
        print(f"  {make_model:20}: {count:6,} images ({percentage:5.1f}%)")

    # Dataset Balance Analysis
    print(f"\n⚖️  Dataset Balance Analysis:")
    print("-" * 35)

    # Make balance
    make_percentages = [(count/total_images)*100 for count in make_counts.values()]
    max_make_pct = max(make_percentages)
    min_make_pct = min(make_percentages)
    print(f"Make imbalance: {max_make_pct:.1f}% (max) vs {min_make_pct:.1f}% (min)")

    # Model balance
    model_percentages = [(count/total_images)*100 for count in model_counts.values()]
    max_model_pct = max(model_percentages)
    min_model_pct = min(model_percentages)
    print(f"Model imbalance: {max_model_pct:.1f}% (max) vs {min_model_pct:.1f}% (min)")

    # Year balance
    year_percentages = [(count/total_images)*100 for count in year_counts.values()]
    max_year_pct = max(year_percentages)
    min_year_pct = min(year_percentages)
    print(f"Year imbalance: {max_year_pct:.1f}% (max) vs {min_year_pct:.1f}% (min)")

    # Show impact of year parsing change
    print(f"\n🔄 Impact of Year Parsing Change:")
    print("-" * 35)
    print(f"Before: ~100+ exact year classes")
    print(f"After:  {len(year_counts)} decade classes")
    print(f"Class reduction: {100 - (len(year_counts)/100)*100:.0f}%")

except Exception as e:
    print(f"Could not analyze dataset: {e}")
    import traceback
    traceback.print_exc()
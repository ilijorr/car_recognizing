#!/usr/bin/env python3

import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_year_classes():
    """Test that the dataset creates the correct number of year classes"""

    try:
        from dataset import CarDataModule

        print("🧪 Testing Year Classes with New Parsing...")
        print("=" * 50)

        # Create progress bar for major steps
        steps = ['Loading dataset', 'Processing labels', 'Creating data splits', 'Analyzing results']

        with tqdm(total=len(steps), desc="Testing", unit="step") as pbar:

            # Step 1: Create data module
            pbar.set_description("Loading dataset")
            print("\nCreating data module...")
            data_module = CarDataModule(
                data_path="data/raw",
                batch_size=16,  # Small batch for testing
                image_size=128,  # Small size for speed
                num_workers=0    # No multiprocessing for testing
            )
            pbar.update(1)

            # Step 2: Get class info
            pbar.set_description("Processing labels")
            time.sleep(0.5)  # Small delay to show progress
            class_info = data_module.get_class_info()
            pbar.update(1)

            # Step 3: Analyze splits
            pbar.set_description("Creating data splits")
            time.sleep(0.5)
            pbar.update(1)

            # Step 4: Show results
            pbar.set_description("Analyzing results")
            time.sleep(0.5)
            pbar.update(1)

        print(f"\n📊 Class Information:")
        print(f"Number of makes: {class_info['num_makes']}")
        print(f"Number of models: {class_info['num_models']}")
        print(f"Number of years: {class_info['num_years']}")

        print(f"\n📅 Year Categories Found:")
        year_classes = class_info['year_classes']
        for i, year_class in enumerate(year_classes):
            print(f"  {i}: {year_class}")

        # Verify we have decade-style years
        decade_years = [y for y in year_classes if y.endswith('s')]
        print(f"\n✅ Decade-style years: {len(decade_years)} out of {len(year_classes)}")

        if len(year_classes) < 20:  # Should be way fewer than before
            print(f"✅ SUCCESS: Year classes reduced from ~104 to {len(year_classes)}")
        else:
            print(f"❌ ISSUE: Still have {len(year_classes)} year classes, expected ~6")

        # Show sample data point with progress
        print(f"\n🔍 Sample Data Point:")
        print("Loading sample...")
        with tqdm(total=1, desc="Getting sample", unit="sample") as sample_pbar:
            sample_image, sample_labels = data_module.train_dataset[0]
            sample_pbar.update(1)

        print(f"Image shape: {sample_image.shape}")
        print(f"Make label: {sample_labels['make'].item()} → {class_info['make_classes'][sample_labels['make'].item()]}")
        print(f"Model label: {sample_labels['model'].item()} → {class_info['model_classes'][sample_labels['model'].item()]}")
        print(f"Year label: {sample_labels['year'].item()} → {class_info['year_classes'][sample_labels['year'].item()]}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_year_classes()
    if success:
        print("\n🎉 Year class testing completed!")
    else:
        print("\n💥 Year class testing failed!")
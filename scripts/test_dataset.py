#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset import CarDataset, CarDataModule


def test_dataset():
    """Test the CarDataset and CarDataModule"""

    # Set your data path here
    data_path = "data/raw"  # Adjust this to your actual data path

    print("Testing CarDataset...")

    try:
        # Test single dataset
        dataset = CarDataset(data_path, split='train', image_size=224)

        print("Dataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        print(f"Number of makes: {dataset.num_makes}")
        print(f"Number of models: {dataset.num_models}")
        print(f"Number of years: {dataset.num_years}")

        # Test getting first sample
        if len(dataset) > 0:
            image, labels = dataset[0]
            print("\nFirst sample:")
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Labels: {labels}")

            # Check image tensor values are normalized
            print(f"Image min/max: {image.min():.3f} / {image.max():.3f}")

        print("\n" + "="*50)
        print("Testing CarDataModule...")

        # Test data module
        data_module = CarDataModule(data_path, batch_size=4, num_workers=0)

        print("Data module created successfully!")
        print(f"Class info: {data_module.get_class_info()}")

        # Test data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        print("\nData loader sizes:")
        print(f"Train: {len(train_loader.dataset)} samples")
        print(f"Val: {len(val_loader.dataset)} samples")
        print(f"Test: {len(test_loader.dataset)} samples")

        # Test getting a batch
        print("\nTesting batch loading...")
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels shapes: {[k + ': ' + str(v.shape) for k, v in labels.items()]}")
            break

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()

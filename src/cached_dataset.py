import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader

from dataset import CarDataset

class CachedCarDataModule:
    def __init__(self,
                 data_path: str,
                 cache_file: str,
                 batch_size: int = 32,
                 image_size: int = 224,
                 num_workers: int = 4):
        """
        Fast data module that loads from pre-computed cache

        Args:
            data_path: Path to dataset directory (for image loading)
            cache_file: Path to cached dataset file
            batch_size: Batch size for data loaders
            image_size: Target image size
            num_workers: Number of workers for data loading
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

        print(f"📦 Loading dataset cache from {cache_file}...")

        # Load cached data
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"Cache file {cache_file} not found. "
                f"Please run 'python scripts/create_data_cache.py' first."
            )

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        # Extract cached components
        self.train_samples = cache_data['train_samples']
        self.val_samples = cache_data['val_samples']
        self.test_samples = cache_data['test_samples']
        self.make_encoder = cache_data['make_encoder']
        self.model_encoder = cache_data['model_encoder']
        self.year_encoder = cache_data['year_encoder']
        self.num_makes = cache_data['num_makes']
        self.num_models = cache_data['num_models']
        self.num_years = cache_data['num_years']

        print(f"✅ Cache loaded! Train: {len(self.train_samples):,}, "
              f"Val: {len(self.val_samples):,}, Test: {len(self.test_samples):,}")

        # Create datasets for each split
        self._create_datasets()

    def _create_datasets(self):
        """Create dataset objects for each split"""

        # Create train dataset
        self.train_dataset = CarDataset(self.data_path, split='train', image_size=self.image_size)
        self.train_dataset.samples = self.train_samples
        self.train_dataset.make_encoder = self.make_encoder
        self.train_dataset.model_encoder = self.model_encoder
        self.train_dataset.year_encoder = self.year_encoder
        self.train_dataset.num_makes = self.num_makes
        self.train_dataset.num_models = self.num_models
        self.train_dataset.num_years = self.num_years

        # Create val dataset
        self.val_dataset = CarDataset(self.data_path, split='val', image_size=self.image_size)
        self.val_dataset.samples = self.val_samples
        self.val_dataset.make_encoder = self.make_encoder
        self.val_dataset.model_encoder = self.model_encoder
        self.val_dataset.year_encoder = self.year_encoder
        self.val_dataset.num_makes = self.num_makes
        self.val_dataset.num_models = self.num_models
        self.val_dataset.num_years = self.num_years

        # Create test dataset
        self.test_dataset = CarDataset(self.data_path, split='test', image_size=self.image_size)
        self.test_dataset.samples = self.test_samples
        self.test_dataset.make_encoder = self.make_encoder
        self.test_dataset.model_encoder = self.model_encoder
        self.test_dataset.year_encoder = self.year_encoder
        self.test_dataset.num_makes = self.num_makes
        self.test_dataset.num_models = self.num_models
        self.test_dataset.num_years = self.num_years

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_class_info(self) -> Dict[str, Any]:
        """Get information about classes"""
        return {
            'num_makes': self.num_makes,
            'num_models': self.num_models,
            'num_years': self.num_years,
            'make_classes': self.make_encoder.classes_.tolist(),
            'model_classes': self.model_encoder.classes_.tolist(),
            'year_classes': self.year_encoder.classes_.tolist()
        }
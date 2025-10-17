from pathlib import Path
from typing import Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.folder_normalizer import CarFolderNormalizer


class CarDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 transform: transforms.Compose = None,
                 image_size: int = 224):
        """
        Car dataset for multi-output classification

        Args:
            data_path: Path to dataset directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
            image_size: Target image size (224 for ResNet-50)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size

        # Initialize normalizer to extract car info
        self.normalizer = CarFolderNormalizer(data_path)

        # Load and process data
        self.samples = self._load_samples()
        self.make_encoder = LabelEncoder()
        self.model_encoder = LabelEncoder()
        self.year_encoder = LabelEncoder()

        # Fit encoders and encode labels
        self._fit_encoders()
        self._encode_labels()

        # Set transforms
        self.transform = transform or self._get_default_transforms()

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load all image samples with their labels"""
        samples = []

        for dir_path in self.data_path.iterdir():
            if not dir_path.is_dir():
                continue

            # Extract car info from directory name
            make, model, year = self.normalizer.extract_car_info(dir_path.name)

            # Get all images in directory
            for img_path in dir_path.glob('*.jpg'):
                samples.append({
                    'image_path': img_path,
                    'make': make,
                    'model': model,
                    'year': year,
                    'directory': dir_path.name
                })

        return samples

    def _fit_encoders(self):
        """Fit label encoders on all data"""
        makes = [sample['make'] for sample in self.samples]
        models = [sample['model'] for sample in self.samples]
        years = [sample['year'] for sample in self.samples]

        self.make_encoder.fit(makes)
        self.model_encoder.fit(models)
        self.year_encoder.fit(years)

        self.num_makes = len(self.make_encoder.classes_)
        self.num_models = len(self.model_encoder.classes_)
        self.num_years = len(self.year_encoder.classes_)

    def _encode_labels(self):
        """Encode string labels to integers"""
        for sample in self.samples:
            sample['make_encoded'] = self.make_encoder.transform([sample['make']])[0]
            sample['model_encoded'] = self.model_encoder.transform([sample['model']])[0]
            sample['year_encoded'] = self.year_encoder.transform([sample['year']])[0]

    def _get_default_transforms(self) -> transforms.Compose:
        """Get default image transformations based on split"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(degrees=10)],
                                       p=0.3),
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1)], p=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[
            torch.Tensor, Dict[str, torch.Tensor]
            ]:
        sample = self.samples[idx]

        # Load and transform image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Return image and labels
        labels = {
            'make': torch.tensor(sample['make_encoded'], dtype=torch.long),
            'model': torch.tensor(sample['model_encoded'], dtype=torch.long),
            'year': torch.tensor(sample['year_encoded'], dtype=torch.long)
        }

        return image, labels


class CarDataModule:
    def __init__(self,
                 data_path: str,
                 batch_size: int = 32,
                 image_size: int = 224,
                 num_workers: int = 4):
        """
        Data module for car dataset with train/val/test splits

        Args:
            data_path: Path to dataset directory
            batch_size: Batch size for data loaders
            image_size: Target image size
            num_workers: Number of workers for data loading
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

        # Create full dataset to get samples
        full_dataset = CarDataset(
                data_path,
                split='train',
                image_size=image_size
                )

        # Store encoders and class counts
        self.make_encoder = full_dataset.make_encoder
        self.model_encoder = full_dataset.model_encoder
        self.year_encoder = full_dataset.year_encoder
        self.num_makes = full_dataset.num_makes
        self.num_models = full_dataset.num_models
        self.num_years = full_dataset.num_years

        # Split data by make-model to prevent leakage
        self.train_samples, self.val_samples, test_samples = self._split_by_make_model(
            full_dataset.samples
        )

        print(f"Data splits: Train={len(self.train_samples)}, "
              f"Val={len(self.val_samples)}, Test={len(test_samples)}")

        # Create datasets for each split
        self.train_dataset = self._create_split_dataset('train', self.train_samples)
        self.val_dataset = self._create_split_dataset('val', self.val_samples)
        self.test_dataset = self._create_split_dataset('test', test_samples)

    def _split_by_make_model(self, all_samples):
        """Split data by make-model combinations to prevent leakage"""

        # Group samples by make-model combination
        make_model_groups = {}
        for sample in all_samples:
            make_model_key = f"{sample['make']}_{sample['model']}"
            if make_model_key not in make_model_groups:
                make_model_groups[make_model_key] = []
            make_model_groups[make_model_key].append(sample)

        # Get unique make-model combinations and their makes for stratification
        combinations = list(make_model_groups.keys())
        combination_makes = [combo.split('_')[0] for combo in combinations]

        print(f"Found {len(combinations)} unique make-model combinations")

        # Check if stratification is possible for makes
        from collections import Counter
        make_counts = Counter(combination_makes)
        min_make_count = min(make_counts.values())

        # Only stratify if all makes have at least 2 samples
        stratify_makes = combination_makes if min_make_count >= 2 else None

        if stratify_makes is None:
            print("⚠️  Some makes have too few samples for stratification - using random split")
        else:
            print(f"✅ Using make stratification ({min_make_count} min samples per make)")

        # Split combinations (not individual samples)
        train_combos, test_combos = train_test_split(
            combinations,
            test_size=0.2,
            random_state=42,
            stratify=stratify_makes
        )

        # Further split train combinations into train/val
        train_combo_makes = [combo.split('_')[0] for combo in train_combos]
        train_make_counts = Counter(train_combo_makes)
        min_train_make_count = min(train_make_counts.values())

        stratify_train_makes = train_combo_makes if min_train_make_count >= 2 else None

        train_combos, val_combos = train_test_split(
            train_combos,
            test_size=0.2,  # 20% of remaining 80% = 16% of total
            random_state=42,
            stratify=stratify_train_makes
        )

        # Convert combination splits to sample lists
        train_samples = []
        val_samples = []
        test_samples = []

        for combo in train_combos:
            train_samples.extend(make_model_groups[combo])

        for combo in val_combos:
            val_samples.extend(make_model_groups[combo])

        for combo in test_combos:
            test_samples.extend(make_model_groups[combo])

        # Verify no leakage
        train_combos_set = set(train_combos)
        val_combos_set = set(val_combos)
        test_combos_set = set(test_combos)

        assert len(train_combos_set & val_combos_set) == 0, "Train-Val leakage detected!"
        assert len(train_combos_set & test_combos_set) == 0, "Train-Test leakage detected!"
        assert len(val_combos_set & test_combos_set) == 0, "Val-Test leakage detected!"

        print("✅ No data leakage detected - make-model combinations are properly separated")

        return train_samples, val_samples, test_samples

    def _create_split_dataset(self, split: str, samples: List[Dict]) -> CarDataset:
        """Create dataset for specific split with pre-filtered samples"""
        dataset = CarDataset(self.data_path, split=split, image_size=self.image_size)
        dataset.samples = samples
        dataset.make_encoder = self.make_encoder
        dataset.model_encoder = self.model_encoder
        dataset.year_encoder = self.year_encoder
        dataset.num_makes = self.num_makes
        dataset.num_models = self.num_models
        dataset.num_years = self.num_years
        dataset._encode_labels()
        return dataset

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

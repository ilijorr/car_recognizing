import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from utils.folder_normalizer import CarFolderNormalizer


class FastCarDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 transform: transforms.Compose = None,
                 image_size: int = 224,
                 samples: List[Dict] = None,
                 encoders: Dict = None):
        """
        Optimized Car dataset for multi-output classification

        Args:
            data_path: Path to dataset directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
            image_size: Target image size (224 for ResNet-50)
            samples: Pre-loaded samples list (for speed)
            encoders: Pre-fitted encoders (for speed)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size

        # Use pre-loaded data if available
        if samples is not None and encoders is not None:
            self.samples = samples
            self.make_encoder = encoders['make']
            self.model_encoder = encoders['model']
            self.year_encoder = encoders['year']
            self.num_makes = len(self.make_encoder.classes_)
            self.num_models = len(self.model_encoder.classes_)
            self.num_years = len(self.year_encoder.classes_)
        else:
            # Fallback to slow loading
            self.normalizer = CarFolderNormalizer(data_path)
            self.samples = self._load_samples()
            self.make_encoder = LabelEncoder()
            self.model_encoder = LabelEncoder()
            self.year_encoder = LabelEncoder()
            self._fit_encoders()

        self._encode_labels()
        self.transform = transform or self._get_default_transforms()

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load all image samples with their labels - SLOW"""
        print("Loading samples... (this may take a while)")
        samples = []

        for dir_path in self.data_path.iterdir():
            if not dir_path.is_dir():
                continue

            # Extract car info from directory name
            make, model, year = self.normalizer.extract_car_info(dir_path.name)

            # Count images without loading them (faster)
            image_files = list(dir_path.glob('*.jpg'))
            for img_path in image_files:
                samples.append({
                    'image_path': str(img_path),  # Store as string to save memory
                    'make': make,
                    'model': model,
                    'year': year,
                    'directory': dir_path.name
                })

        print(f"Loaded {len(samples)} samples")
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
                transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
                transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.3),
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]

        # Load and transform image (only when needed)
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


class FastCarDataModule:
    def __init__(self,
                 data_path: str,
                 batch_size: int = 32,
                 image_size: int = 224,
                 num_workers: int = 4,
                 cache_file: str = None):
        """
        Fast data module with caching

        Args:
            data_path: Path to dataset directory
            batch_size: Batch size for data loaders
            image_size: Target image size
            num_workers: Number of workers for data loading
            cache_file: Path to cache file (speeds up repeated runs)
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

        # Try to load from cache first
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}...")
            self._load_from_cache(cache_file)
        else:
            print("Building dataset from scratch...")
            self._build_dataset()

            # Save to cache for next time
            if cache_file:
                self._save_to_cache(cache_file)

        self._create_datasets()

    def _build_dataset(self):
        """Build dataset from scratch - SLOW"""
        # Create full dataset to get samples
        full_dataset = FastCarDataset(self.data_path, split='train', image_size=self.image_size)

        # Store data
        self.all_samples = full_dataset.samples
        self.make_encoder = full_dataset.make_encoder
        self.model_encoder = full_dataset.model_encoder
        self.year_encoder = full_dataset.year_encoder
        self.num_makes = full_dataset.num_makes
        self.num_models = full_dataset.num_models
        self.num_years = full_dataset.num_years

        # Split data: 64% train, 16% val, 20% test
        self.train_samples, test_samples = train_test_split(
            self.all_samples, test_size=0.2, random_state=42,
            stratify=[s['make'] for s in self.all_samples]
        )

        self.train_samples, self.val_samples = train_test_split(
            self.train_samples, test_size=0.2, random_state=42,
            stratify=[s['make'] for s in self.train_samples]
        )

        self.test_samples = test_samples

        print(f"Data splits: Train={len(self.train_samples)}, "
              f"Val={len(self.val_samples)}, Test={len(self.test_samples)}")

    def _save_to_cache(self, cache_file: str):
        """Save processed data to cache"""
        print(f"Saving to cache: {cache_file}")
        cache_data = {
            'train_samples': self.train_samples,
            'val_samples': self.val_samples,
            'test_samples': self.test_samples,
            'encoders': {
                'make': self.make_encoder,
                'model': self.model_encoder,
                'year': self.year_encoder
            },
            'num_classes': {
                'makes': self.num_makes,
                'models': self.num_models,
                'years': self.num_years
            }
        }

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def _load_from_cache(self, cache_file: str):
        """Load processed data from cache - FAST"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        self.train_samples = cache_data['train_samples']
        self.val_samples = cache_data['val_samples']
        self.test_samples = cache_data['test_samples']
        self.make_encoder = cache_data['encoders']['make']
        self.model_encoder = cache_data['encoders']['model']
        self.year_encoder = cache_data['encoders']['year']
        self.num_makes = cache_data['num_classes']['makes']
        self.num_models = cache_data['num_classes']['models']
        self.num_years = cache_data['num_classes']['years']

        print(f"Loaded from cache: Train={len(self.train_samples)}, "
              f"Val={len(self.val_samples)}, Test={len(self.test_samples)}")

    def _create_datasets(self):
        """Create dataset objects for each split"""
        encoders = {
            'make': self.make_encoder,
            'model': self.model_encoder,
            'year': self.year_encoder
        }

        self.train_dataset = FastCarDataset(
            self.data_path, 'train', image_size=self.image_size,
            samples=self.train_samples, encoders=encoders
        )

        self.val_dataset = FastCarDataset(
            self.data_path, 'val', image_size=self.image_size,
            samples=self.val_samples, encoders=encoders
        )

        self.test_dataset = FastCarDataset(
            self.data_path, 'test', image_size=self.image_size,
            samples=self.test_samples, encoders=encoders
        )

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

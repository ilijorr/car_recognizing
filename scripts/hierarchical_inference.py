#!/usr/bin/env python3

import sys
import os
import argparse
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hierarchical_model import HierarchicalCarClassifier, create_make_to_models_mapping
from cached_dataset import CachedCarDataModule


class HierarchicalCarInference:
    def __init__(self, model_path: str, cache_file: str = None):
        """
        Initialize hierarchical car classifier for inference

        Args:
            model_path: Path to saved hierarchical model (.pth file)
            cache_file: Path to dataset cache (for encoders)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load checkpoint
        print(f"Loading model from {model_path}...")
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Try to load encoders from checkpoint first (FAST!)
        if all(key in self.checkpoint for key in ['make_encoder', 'model_encoder', 'year_encoder']):
            print("Loading encoders from checkpoint (fast)...")
            self.make_encoder = self.checkpoint['make_encoder']
            self.model_encoder = self.checkpoint['model_encoder']
            self.year_encoder = self.checkpoint['year_encoder']
        else:
            # Fallback: load from cache (SLOW!)
            print("⚠️  Encoders not in checkpoint, loading from cache (slow)...")
            if cache_file is None:
                config = self.checkpoint.get('config', {})
                cache_file = config.get('cache_file', 'cache/dataset_cache.pkl')

            import time
            start_time = time.time()
            data_module = CachedCarDataModule(
                data_path='data/raw',
                cache_file=cache_file,
                batch_size=1
            )
            self.make_encoder = data_module.make_encoder
            self.model_encoder = data_module.model_encoder
            self.year_encoder = data_module.year_encoder
            print(f"  Encoders loaded in {time.time() - start_time:.2f}s")

        # Get make-to-models mapping from checkpoint or recreate
        if 'make_to_models' in self.checkpoint:
            self.make_to_models = self.checkpoint['make_to_models']
        else:
            print("⚠️  make_to_models not in checkpoint - this will cause errors!")
            self.make_to_models = {}

        # Initialize model
        print("Initializing hierarchical model...")
        class_info = self.checkpoint.get('class_info', self.data_module.get_class_info())

        self.model = HierarchicalCarClassifier(
            num_makes=class_info['num_makes'],
            num_years=class_info['num_years'],
            make_to_models=self.make_to_models,
            pretrained=False
        ).to(self.device)

        # Load trained weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ Model loaded successfully!")
        if 'best_val_emr' in self.checkpoint:
            print(f"   Best validation EMR: {self.checkpoint['best_val_emr']:.4f}")

    def preprocess_image(self, image_path: str):
        """Load and preprocess image for inference"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return input_tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def predict(self, image_path: str):
        """
        Perform hierarchical prediction on image

        Args:
            image_path: Path to image file

        Returns:
            dict: Prediction results with confidences
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image_path)
        if input_tensor is None:
            return None

        print(f"\n🔮 Running hierarchical inference on {Path(image_path).name}...")

        with torch.no_grad():
            # Single forward pass for all stages (MUCH FASTER!)
            print("  Running single forward pass for all stages...")
            make_ids = torch.tensor([0], device=self.device)  # Dummy for initial call
            make_names = ['BMW']  # Dummy for initial call

            # Get all predictions at once
            all_outputs = self.model(input_tensor, make_ids=make_ids,
                                   make_names=make_names, stage='all')

            # Stage 1: Process make prediction
            print("  Stage 1 - Make prediction...")
            make_output = all_outputs['make']
            make_probs = torch.softmax(make_output, dim=1)
            make_pred_idx = torch.argmax(make_output, dim=1).item()
            make_confidence = make_probs[0, make_pred_idx].item()

            make_name = self.make_encoder.inverse_transform([make_pred_idx])[0]
            print(f"    Predicted: {make_name} (confidence: {make_confidence:.3f})")

            # Stage 2: Re-run model prediction with correct make (if needed)
            print("  Stage 2 - Model prediction...")
            if make_name != 'BMW':  # Re-run only if make changed
                make_ids = torch.tensor([make_pred_idx], device=self.device)
                make_names = [make_name]
                model_output = self.model(input_tensor, make_ids=make_ids,
                                        make_names=make_names, stage='model')['model']
            else:
                model_output = all_outputs['model']

            if make_name in self.make_to_models:
                valid_models = self.make_to_models[make_name]
                num_valid = len(valid_models)

                model_probs = torch.softmax(model_output[0, :num_valid], dim=0)
                model_pred_idx = torch.argmax(model_probs).item()
                model_confidence = model_probs[model_pred_idx].item()

                model_name = valid_models[model_pred_idx]
                print(f"    Predicted: {model_name} (confidence: {model_confidence:.3f})")
                print(f"    Available models for {make_name}: {len(valid_models)}")
            else:
                model_name = "UNKNOWN"
                model_confidence = 0.0
                print(f"    ERROR: Make {make_name} not found in model mapping")

            # Stage 3: Process year prediction
            print("  Stage 3 - Year prediction...")
            year_output = all_outputs['year']
            year_probs = torch.softmax(year_output, dim=1)
            year_pred_idx = torch.argmax(year_output, dim=1).item()
            year_confidence = year_probs[0, year_pred_idx].item()

            year_name = self.year_encoder.inverse_transform([year_pred_idx])[0]
            print(f"    Predicted: {year_name} (confidence: {year_confidence:.3f})")

        # Final results
        results = {
            'make': {
                'name': make_name,
                'confidence': make_confidence
            },
            'model': {
                'name': model_name,
                'confidence': model_confidence
            },
            'year': {
                'name': year_name,
                'confidence': year_confidence
            }
        }

        return results


def main():
    parser = argparse.ArgumentParser(description='Hierarchical Car Classifier Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to saved hierarchical model (.pth file)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file for inference')
    parser.add_argument('--cache', type=str, default=None,
                        help='Path to dataset cache file (optional)')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return

    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return

    try:
        # Initialize inference system
        inference = HierarchicalCarInference(args.model, args.cache)

        # Run prediction
        results = inference.predict(args.image)

        if results:
            print(f"\n🎯 Final Prediction for {Path(args.image).name}:")
            print(f"  Make: {results['make']['name']} ({results['make']['confidence']:.3f})")
            print(f"  Model: {results['model']['name']} ({results['model']['confidence']:.3f})")
            print(f"  Year: {results['year']['name']} ({results['year']['confidence']:.3f})")

            # Overall confidence (geometric mean)
            overall_conf = (results['make']['confidence'] *
                          results['model']['confidence'] *
                          results['year']['confidence']) ** (1/3)
            print(f"  Overall: {overall_conf:.3f}")
        else:
            print("❌ Prediction failed")

    except Exception as e:
        print(f"💥 Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
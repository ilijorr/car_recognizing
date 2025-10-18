#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import argparse
import torch
from torchvision import transforms
from PIL import Image
import json

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import MultiOutputCarClassifier


class CarClassifierInference:
    def __init__(self, model_path: str):
        """
        Initialize the car classifier for inference

        Args:
            model_path: Path to the saved model file (.pth)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract saved information
        self.config = checkpoint['config']
        self.class_info = checkpoint['class_info']

        # Reconstruct model
        self.model = MultiOutputCarClassifier(
            num_makes=self.class_info['num_makes'],
            num_models=self.class_info['num_models'],
            num_years=self.class_info['num_years'],
            backbone=self.config['backbone'],
            pretrained=False  # We're loading trained weights
        ).to(self.device)

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Setup image transforms (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Create reverse mappings for decoding predictions
        self.make_classes = self.class_info['make_classes']
        self.model_classes = self.class_info['model_classes']
        self.year_classes = self.class_info['year_classes']

        print("Model loaded successfully!")
        print(f"Model knows {len(self.make_classes)} makes, "
              f"{len(self.model_classes)} models, "
              f"{len(self.year_classes)} years")

    def predict_image(self, image_path: str) -> dict:
        """
        Predict car make, model, and year from an image

        Args:
            image_path: Path to the car image

        Returns:
            Dictionary with predictions and confidence scores
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")

        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Get probabilities and predictions
        make_probs = torch.softmax(predictions['make'], dim=1)
        model_probs = torch.softmax(predictions['model'], dim=1)
        year_probs = torch.softmax(predictions['year'], dim=1)

        # Get top predictions
        make_pred = torch.argmax(make_probs, dim=1).item()
        model_pred = torch.argmax(model_probs, dim=1).item()
        year_pred = torch.argmax(year_probs, dim=1).item()

        # Get confidence scores
        make_confidence = make_probs[0, make_pred].item()
        model_confidence = model_probs[0, model_pred].item()
        year_confidence = year_probs[0, year_pred].item()

        # Decode predictions to readable labels
        result = {
            'predictions': {
                'make': self.make_classes[make_pred],
                'model': self.model_classes[model_pred],
                'year': self.year_classes[year_pred]
            },
            'confidence_scores': {
                'make': round(make_confidence, 4),
                'model': round(model_confidence, 4),
                'year': round(year_confidence, 4)
            },
            'overall_confidence': round((make_confidence + model_confidence + year_confidence) / 3, 4)
        }

        return result

    def predict_batch(self, image_paths: list) -> list:
        """
        Predict multiple images at once

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results

    def get_top_k_predictions(self, image_path: str, k: int = 3) -> dict:
        """
        Get top-k predictions for each category

        Args:
            image_path: Path to the car image
            k: Number of top predictions to return

        Returns:
            Dictionary with top-k predictions for each category
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Get probabilities
        make_probs = torch.softmax(predictions['make'], dim=1)
        model_probs = torch.softmax(predictions['model'], dim=1)
        year_probs = torch.softmax(predictions['year'], dim=1)

        # Get top-k for each category
        make_topk = torch.topk(make_probs, k, dim=1)
        model_topk = torch.topk(model_probs, k, dim=1)
        year_topk = torch.topk(year_probs, k, dim=1)

        result = {
            'top_makes': [
                {
                    'prediction': self.make_classes[idx.item()],
                    'confidence': prob.item()
                }
                for prob, idx in zip(make_topk.values[0], make_topk.indices[0])
            ],
            'top_models': [
                {
                    'prediction': self.model_classes[idx.item()],
                    'confidence': prob.item()
                }
                for prob, idx in zip(model_topk.values[0], model_topk.indices[0])
            ],
            'top_years': [
                {
                    'prediction': self.year_classes[idx.item()],
                    'confidence': prob.item()
                }
                for prob, idx in zip(year_topk.values[0], year_topk.indices[0])
            ]
        }

        return result


def main():
    parser = argparse.ArgumentParser(description='Car Classifier Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved model (.pth file)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to car image')
    parser.add_argument('--top_k', type=int, default=1,
                       help='Show top-k predictions (default: 1)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to JSON file')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    # Initialize classifier
    try:
        classifier = CarClassifierInference(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Make prediction
    try:
        if args.top_k == 1:
            result = classifier.predict_image(args.image)
            print(f"\n🚗 Car Classification Results for: {args.image}")
            print("=" * 60)
            print(f"Make:  {result['predictions']['make']} ({result['confidence_scores']['make']:.2%})")
            print(f"Model: {result['predictions']['model']} ({result['confidence_scores']['model']:.2%})")
            print(f"Year:  {result['predictions']['year']} ({result['confidence_scores']['year']:.2%})")
            print(f"\nOverall Confidence: {result['overall_confidence']:.2%}")
        else:
            result = classifier.get_top_k_predictions(args.image, args.top_k)
            print(f"\n🚗 Top-{args.top_k} Car Classification Results for: {args.image}")
            print("=" * 60)

            print(f"\nTop {args.top_k} Makes:")
            for i, pred in enumerate(result['top_makes'], 1):
                print(f"  {i}. {pred['prediction']} ({pred['confidence']:.2%})")

            print(f"\nTop {args.top_k} Models:")
            for i, pred in enumerate(result['top_models'], 1):
                print(f"  {i}. {pred['prediction']} ({pred['confidence']:.2%})")

            print(f"\nTop {args.top_k} Years:")
            for i, pred in enumerate(result['top_years'], 1):
                print(f"  {i}. {pred['prediction']} ({pred['confidence']:.2%})")

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
"""
Simple predictor class for easy integration into other applications
"""

import sys
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

# Add src to path if needed
sys.path.append(str(Path(__file__).parent))

from model import MultiOutputCarClassifier


class CarPredictor:
    """
    Simple car classifier predictor for easy use

    Example usage:
        predictor = CarPredictor('results/best_model.pth')
        result = predictor.predict('path/to/car/image.jpg')
        print(f"This is a {result['make']} {result['model']} from {result['year']}")
    """

    def __init__(self, model_path: str):
        """
        Initialize predictor

        Args:
            model_path: Path to saved model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.class_info = checkpoint['class_info']

        # Create model
        self.model = MultiOutputCarClassifier(
            num_makes=self.class_info['num_makes'],
            num_models=self.class_info['num_models'],
            num_years=self.class_info['num_years'],
            backbone=self.config['backbone'],
            pretrained=False
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Class mappings
        self.make_classes = self.class_info['make_classes']
        self.model_classes = self.class_info['model_classes']
        self.year_classes = self.class_info['year_classes']

    def predict(self, image_path: str) -> dict:
        """
        Predict car from image

        Args:
            image_path: Path to car image

        Returns:
            Dictionary with make, model, year predictions and confidence scores
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Get predictions and confidence
        make_probs = torch.softmax(predictions['make'], dim=1)
        model_probs = torch.softmax(predictions['model'], dim=1)
        year_probs = torch.softmax(predictions['year'], dim=1)

        make_pred = torch.argmax(make_probs, dim=1).item()
        model_pred = torch.argmax(model_probs, dim=1).item()
        year_pred = torch.argmax(year_probs, dim=1).item()

        return {
            'make': self.make_classes[make_pred],
            'model': self.model_classes[model_pred],
            'year': self.year_classes[year_pred],
            'confidence': {
                'make': make_probs[0, make_pred].item(),
                'model': model_probs[0, model_pred].item(),
                'year': year_probs[0, year_pred].item()
            }
        }

    def predict_simple(self, image_path: str) -> str:
        """
        Get simple string prediction

        Args:
            image_path: Path to car image

        Returns:
            String like "BMW X5 2018"
        """
        result = self.predict(image_path)
        return f"{result['make']} {result['model']} {result['year']}"
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple


class MultiOutputCarClassifier(nn.Module):
    def __init__(self,
                 num_makes: int,
                 num_models: int,
                 num_years: int,
                 backbone: str = 'resnet50',
                 pretrained: bool = True):
        """
        Multi-output car classifier based on ResNet-50

        Args:
            num_makes: Number of car manufacturers
            num_models: Number of car models
            num_years: Number of year categories
            backbone: Model backbone ('resnet50' or 'efficientnet_b3')
            pretrained: Use pretrained weights
        """
        super(MultiOutputCarClassifier, self).__init__()

        self.num_makes = num_makes
        self.num_models = num_models
        self.num_years = num_years

        # Initialize backbone
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = self.backbone.fc.in_features
            # Remove the final classification layer
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b3':
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            self.feature_dim = self.backbone.classifier[1].in_features
            # Remove the final classification layer
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Multi-output heads
        self.make_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_makes)
        )

        self.model_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_models)
        )

        self.year_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_years)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Dictionary with predictions for make, model, year
        """
        # Extract features using backbone
        features = self.backbone(x)

        # Get predictions from each head
        make_logits = self.make_head(features)
        model_logits = self.model_head(features)
        year_logits = self.year_head(features)

        return {
            'make': make_logits,
            'model': model_logits,
            'year': year_logits
        }

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> Tuple[list, list]:
        """
        Get trainable parameters separated by backbone and heads

        Returns:
            backbone_params: List of backbone parameters
            head_params: List of head parameters
        """
        backbone_params = list(self.backbone.parameters())
        head_params = (list(self.make_head.parameters()) +
                       list(self.model_head.parameters()) +
                       list(self.year_head.parameters()))

        return backbone_params, head_params


class MultiOutputLoss(nn.Module):
    def __init__(self,
                 make_weight: float = 1.0,
                 model_weight: float = 1.0,
                 year_weight: float = 1.0,
                 make_class_weights: torch.Tensor = None,
                 model_class_weights: torch.Tensor = None,
                 year_class_weights: torch.Tensor = None):
        """
        Combined loss for multi-output classification

        Args:
            make_weight: Weight for make classification loss
            model_weight: Weight for model classification loss
            year_weight: Weight for year classification loss
            make_class_weights: Class weights for make imbalance
            model_class_weights: Class weights for model imbalance
            year_class_weights: Class weights for year imbalance
        """
        super(MultiOutputLoss, self).__init__()

        self.make_weight = make_weight
        self.model_weight = model_weight
        self.year_weight = year_weight

        # Create separate loss functions with class weights
        self.make_loss = nn.CrossEntropyLoss(weight=make_class_weights)
        self.model_loss = nn.CrossEntropyLoss(weight=model_class_weights)
        self.year_loss = nn.CrossEntropyLoss(weight=year_class_weights)

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss

        Args:
            predictions: Model predictions
            targets: Ground truth labels

        Returns:
            Dictionary with individual and total losses
        """
        make_loss = self.make_loss(predictions['make'], targets['make'])
        model_loss = self.model_loss(predictions['model'], targets['model'])
        year_loss = self.year_loss(predictions['year'], targets['year'])

        total_loss = (self.make_weight * make_loss +
                      self.model_weight * model_loss +
                      self.year_weight * year_loss)

        return {
            'total_loss': total_loss,
            'make_loss': make_loss,
            'model_loss': model_loss,
            'year_loss': year_loss
        }


def calculate_class_weights(targets: list, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets using sklearn

    Args:
        targets: List of target labels
        num_classes: Total number of classes

    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # Get unique classes and compute weights
    unique_classes = np.unique(targets)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=targets
    )

    # Create weight tensor for all classes (in case some classes are missing)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for i, class_idx in enumerate(unique_classes):
        if class_idx < num_classes:
            weights[class_idx] = class_weights[i]

    return weights


def create_model(num_makes: int,
                 num_models: int,
                 num_years: int,
                 backbone: str = 'resnet50',
                 pretrained: bool = True) -> MultiOutputCarClassifier:
    """
    Factory function to create model

    Args:
        num_makes: Number of car manufacturers
        num_models: Number of car models
        num_years: Number of year categories
        backbone: Model backbone
        pretrained: Use pretrained weights

    Returns:
        Initialized model
    """
    return MultiOutputCarClassifier(
        num_makes=num_makes,
        num_models=num_models,
        num_years=num_years,
        backbone=backbone,
        pretrained=pretrained
    )

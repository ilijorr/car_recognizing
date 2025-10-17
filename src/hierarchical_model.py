import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, List, Any


class MakeClassifier(nn.Module):
    """Stage 1: Classify car make (manufacturer)"""

    def __init__(self, num_makes: int, pretrained: bool = True):
        super().__init__()
        self.num_makes = num_makes

        # Load ResNet-50 backbone
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50(weights=None)

        # Replace final layer
        self.backbone.fc = nn.Linear(2048, num_makes)

    def forward(self, x):
        return self.backbone(x)


class ModelClassifier(nn.Module):
    """Stage 2: Classify car model given the make"""

    def __init__(self, make_to_models: Dict[str, List[str]], num_makes: int, pretrained: bool = True):
        super().__init__()
        self.make_to_models = make_to_models
        self.num_makes = num_makes

        # Load ResNet-50 backbone (without final FC layer)
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50(weights=None)

        # Remove final FC layer to get features
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

        # Make embedding
        self.make_embedding = nn.Embedding(num_makes, 128)

        # Create separate classifier for each make
        self.model_classifiers = nn.ModuleDict()
        for make, models in make_to_models.items():
            self.model_classifiers[make] = nn.Linear(2048 + 128, len(models))

    def forward(self, x, make_ids, make_names):
        # Extract image features
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)  # [batch_size, 2048]

        # Get make embeddings
        make_emb = self.make_embedding(make_ids)  # [batch_size, 128]

        # Combine features
        combined = torch.cat([features, make_emb], dim=1)  # [batch_size, 2176]

        # Find max number of models across all makes for consistent output size
        max_models = max(len(models) for models in self.make_to_models.values())
        batch_size = x.size(0)

        # Create output tensor with consistent size
        output = torch.zeros(batch_size, max_models, device=x.device)

        # Apply appropriate classifier for each sample based on its make
        for i, make_name in enumerate(make_names):
            if make_name in self.model_classifiers:
                classifier = self.model_classifiers[make_name]
                single_output = classifier(combined[i:i+1])  # [1, num_models_for_make]
                # Copy to the appropriate slice of the output tensor
                num_models_for_make = single_output.size(1)
                output[i, :num_models_for_make] = single_output[0]
            # For unknown makes, leave as zeros (already initialized)

        return output


class YearClassifier(nn.Module):
    """Stage 3: Classify car year (decade)"""

    def __init__(self, num_years: int, pretrained: bool = True):
        super().__init__()
        self.num_years = num_years

        # Load ResNet-50 backbone
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50(weights=None)

        # Replace final layer
        self.backbone.fc = nn.Linear(2048, num_years)

    def forward(self, x):
        return self.backbone(x)


class HierarchicalCarClassifier(nn.Module):
    """Complete hierarchical classification system"""

    def __init__(self, num_makes: int, num_years: int, make_to_models: Dict[str, List[str]],
                 pretrained: bool = True):
        super().__init__()

        self.num_makes = num_makes
        self.num_years = num_years
        self.make_to_models = make_to_models

        # Initialize individual classifiers
        self.make_classifier = MakeClassifier(num_makes, pretrained)
        self.model_classifier = ModelClassifier(make_to_models, num_makes, pretrained)
        self.year_classifier = YearClassifier(num_years, pretrained)

    def forward(self, x, make_ids=None, make_names=None, stage='all'):
        """
        Forward pass through hierarchical classifier

        Args:
            x: Input images
            make_ids: Ground truth make IDs (for training model classifier)
            make_names: Ground truth make names (for training model classifier)
            stage: Which stage to run ('make', 'model', 'year', 'all')
        """
        results = {}

        if stage in ['make', 'all']:
            make_output = self.make_classifier(x)
            results['make'] = make_output

            if make_ids is None:
                # Inference mode - use predicted makes
                predicted_makes = torch.argmax(make_output, dim=1)
                make_ids = predicted_makes

        if stage in ['model', 'all']:
            if make_ids is not None and make_names is not None:
                model_output = self.model_classifier(x, make_ids, make_names)
                results['model'] = model_output

        if stage in ['year', 'all']:
            year_output = self.year_classifier(x)
            results['year'] = year_output

        return results

    def predict(self, x, make_encoder, model_encoders):
        """
        Complete inference pipeline

        Args:
            x: Input images
            make_encoder: LabelEncoder for makes
            model_encoders: Dict of LabelEncoders for models per make
        """
        self.eval()
        with torch.no_grad():
            # Stage 1: Predict make
            make_logits = self.make_classifier(x)
            make_probs = torch.softmax(make_logits, dim=1)
            predicted_make_ids = torch.argmax(make_logits, dim=1)

            # Convert to make names
            predicted_make_names = [make_encoder.inverse_transform([mid.item()])[0]
                                  for mid in predicted_make_ids]

            # Stage 2: Predict model given make
            model_outputs = []
            for i, make_name in enumerate(predicted_make_names):
                if make_name in self.make_to_models:
                    single_input = x[i:i+1]
                    single_make_id = predicted_make_ids[i:i+1]
                    model_logits = self.model_classifier(single_input, single_make_id, [make_name])
                    model_probs = torch.softmax(model_logits, dim=1)
                    model_outputs.append(model_probs)
                else:
                    # Unknown make
                    dummy_output = torch.zeros(1, 1, device=x.device)
                    model_outputs.append(dummy_output)

            # Stage 3: Predict year
            year_logits = self.year_classifier(x)
            year_probs = torch.softmax(year_logits, dim=1)
            predicted_year_ids = torch.argmax(year_logits, dim=1)

            return {
                'make_probs': make_probs,
                'predicted_makes': predicted_make_names,
                'model_outputs': model_outputs,
                'year_probs': year_probs,
                'predicted_year_ids': predicted_year_ids
            }


def create_make_to_models_mapping(samples: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Create mapping from makes to their models from dataset samples"""
    make_to_models = {}

    for sample in samples:
        make = sample['make']
        model = sample['model']

        if make not in make_to_models:
            make_to_models[make] = set()
        make_to_models[make].add(model)

    # Convert sets to sorted lists
    for make in make_to_models:
        make_to_models[make] = sorted(list(make_to_models[make]))

    return make_to_models


class HierarchicalLoss(nn.Module):
    """Combined loss for hierarchical classification"""

    def __init__(self, make_weight: float = 1.0, model_weight: float = 2.0, year_weight: float = 1.0):
        super().__init__()
        self.make_weight = make_weight
        self.model_weight = model_weight
        self.year_weight = year_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """
        Compute hierarchical loss

        Args:
            predictions: Dict with 'make', 'model', 'year' tensors
            targets: Dict with 'make', 'model', 'year' tensors
        """
        total_loss = 0.0
        loss_dict = {}

        if 'make' in predictions:
            make_loss = self.criterion(predictions['make'], targets['make'])
            loss_dict['make_loss'] = make_loss
            total_loss += self.make_weight * make_loss

        if 'model' in predictions:
            model_loss = self.criterion(predictions['model'], targets['model'])
            loss_dict['model_loss'] = model_loss
            total_loss += self.model_weight * model_loss

        if 'year' in predictions:
            year_loss = self.criterion(predictions['year'], targets['year'])
            loss_dict['year_loss'] = year_loss
            total_loss += self.year_weight * year_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict
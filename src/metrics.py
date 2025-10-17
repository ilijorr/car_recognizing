import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, List


class MultiOutputMetrics:
    def __init__(self):
        """Metrics calculator for multi-output classification"""
        self.reset()

    def reset(self):
        """Reset all stored predictions and targets"""
        self.make_preds = []
        self.make_targets = []
        self.model_preds = []
        self.model_targets = []
        self.year_preds = []
        self.year_targets = []

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """
        Update metrics with new batch

        Args:
            predictions: Model predictions dictionary
            targets: Ground truth targets dictionary
        """
        # Convert logits to predictions
        make_pred = torch.argmax(predictions['make'], dim=1)
        model_pred = torch.argmax(predictions['model'], dim=1)
        year_pred = torch.argmax(predictions['year'], dim=1)

        # Store predictions and targets
        self.make_preds.extend(make_pred.cpu().numpy())
        self.make_targets.extend(targets['make'].cpu().numpy())

        self.model_preds.extend(model_pred.cpu().numpy())
        self.model_targets.extend(targets['model'].cpu().numpy())

        self.year_preds.extend(year_pred.cpu().numpy())
        self.year_targets.extend(targets['year'].cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics

        Returns:
            Dictionary with computed metrics
        """
        metrics = {}

        # Individual accuracies
        metrics['make_accuracy'] = accuracy_score(self.make_targets, self.make_preds)
        metrics['model_accuracy'] = accuracy_score(self.model_targets, self.model_preds)
        metrics['year_accuracy'] = accuracy_score(self.year_targets, self.year_preds)

        # F1-scores (weighted for imbalanced data)
        metrics['make_f1'] = f1_score(self.make_targets, self.make_preds, average='weighted')
        metrics['model_f1'] = f1_score(self.model_targets, self.model_preds, average='weighted')
        metrics['year_f1'] = f1_score(self.year_targets, self.year_preds, average='weighted')

        # RMSE for year (treating as regression-like metric)
        year_rmse = np.sqrt(np.mean((np.array(self.year_targets) - np.array(self.year_preds)) ** 2))
        metrics['year_rmse'] = year_rmse

        # Exact Match Ratio (EMR) - all outputs must be correct
        exact_matches = 0
        total_samples = len(self.make_targets)

        for i in range(total_samples):
            if (self.make_preds[i] == self.make_targets[i] and
                self.model_preds[i] == self.model_targets[i] and
                self.year_preds[i] == self.year_targets[i]):
                exact_matches += 1

        metrics['emr'] = exact_matches / total_samples if total_samples > 0 else 0.0

        # Overall accuracy (average of individual accuracies)
        metrics['overall_accuracy'] = (metrics['make_accuracy'] +
                                       metrics['model_accuracy'] +
                                       metrics['year_accuracy']) / 3

        return metrics

    def get_confusion_info(self) -> Dict[str, Dict]:
        """
        Get confusion matrix information for each output

        Returns:
            Dictionary with confusion information
        """
        from sklearn.metrics import classification_report

        info = {}

        # Make confusion info
        if len(set(self.make_targets)) > 1:
            info['make'] = classification_report(
                self.make_targets, self.make_preds, output_dict=True, zero_division=0
            )

        # Model confusion info
        if len(set(self.model_targets)) > 1:
            info['model'] = classification_report(
                self.model_targets, self.model_preds, output_dict=True, zero_division=0
            )

        # Year confusion info
        if len(set(self.year_targets)) > 1:
            info['year'] = classification_report(
                self.year_targets, self.year_preds, output_dict=True, zero_division=0
            )

        return info


def calculate_class_weights(targets: List[int], num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets

    Args:
        targets: List of target labels
        num_classes: Number of classes

    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight

    unique_classes = np.unique(targets)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=targets
    )

    # Create weight tensor for all classes
    weights = torch.ones(num_classes)
    for i, weight in enumerate(class_weights):
        if i < len(unique_classes):
            weights[unique_classes[i]] = weight

    return weights


def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Calculate top-k accuracy

    Args:
        predictions: Model predictions (logits)
        targets: Ground truth targets
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred_indices = predictions.topk(k, dim=1, largest=True, sorted=True)
        correct = pred_indices.eq(targets.view(-1, 1).expand_as(pred_indices))
        correct_k = correct[:, :k].contiguous().view(-1).float().sum(0, keepdim=True)
        return (correct_k / batch_size).item()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


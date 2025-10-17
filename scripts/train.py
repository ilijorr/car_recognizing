#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset import CarDataModule
from model import MultiOutputCarClassifier, MultiOutputLoss
from metrics import MultiOutputMetrics, AverageMeter


class CarTrainer:
    def __init__(self, config):
        """
        Car classifier trainer

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize data module
        self.data_module = CarDataModule(
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            num_workers=config['num_workers']
        )

        # Get class information
        class_info = self.data_module.get_class_info()
        self.num_makes = class_info['num_makes']
        self.num_models = class_info['num_models']
        self.num_years = class_info['num_years']

        print(f"Dataset loaded: {self.num_makes} makes, {self.num_models} models, {self.num_years} years")

        # Initialize model
        self.model = MultiOutputCarClassifier(
            num_makes=self.num_makes,
            num_models=self.num_models,
            num_years=self.num_years,
            backbone=config['backbone'],
            pretrained=config['pretrained']
        ).to(self.device)

        # Initialize loss function
        self.criterion = MultiOutputLoss(
            make_weight=config['loss_weights']['make'],
            model_weight=config['loss_weights']['model'],
            year_weight=config['loss_weights']['year']
        )

        # Training history
        self.train_history = {
            'loss': [], 'make_acc': [], 'model_acc': [], 'year_acc': [], 'emr': []
        }
        self.val_history = {
            'loss': [], 'make_acc': [], 'model_acc': [], 'year_acc': [], 'emr': []
        }

        # Best metrics tracking
        self.best_val_emr = 0.0
        self.best_model_state = None

    def train_phase_1(self):
        """
        Phase 1: Train with frozen backbone (only head layers)
        """
        print("\n" + "="*50)
        print("PHASE 1: Training with frozen backbone")
        print("="*50)

        # Freeze backbone
        self.model.freeze_backbone()

        # Setup optimizer for heads only
        head_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                head_params.append(param)

        optimizer = optim.Adam(head_params, lr=self.config['phase1_lr'])
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

        # Train for specified epochs
        for epoch in range(self.config['phase1_epochs']):
            train_metrics = self._train_epoch(optimizer, epoch + 1, "Phase 1")
            val_metrics = self._validate_epoch(epoch + 1, "Phase 1")

            # Update learning rate
            scheduler.step(val_metrics['emr'])

            # Save history
            self._update_history(train_metrics, val_metrics)

            # Check for best model
            if val_metrics['emr'] > self.best_val_emr:
                self.best_val_emr = val_metrics['emr']
                self.best_model_state = self.model.state_dict().copy()

            print(f"Phase 1 Epoch {epoch+1}: Val EMR = {val_metrics['emr']:.4f}")

    def train_phase_2(self):
        """
        Phase 2: Fine-tune entire model (unfreeze backbone)
        """
        print("\n" + "="*50)
        print("PHASE 2: Fine-tuning entire model")
        print("="*50)

        # Unfreeze backbone
        self.model.unfreeze_backbone()

        # Setup optimizer for all parameters
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['phase2_lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['phase2_epochs'])

        # Train for specified epochs
        for epoch in range(self.config['phase2_epochs']):
            train_metrics = self._train_epoch(optimizer, epoch + 1, "Phase 2")
            val_metrics = self._validate_epoch(epoch + 1, "Phase 2")

            # Update learning rate
            scheduler.step()

            # Save history
            self._update_history(train_metrics, val_metrics)

            # Check for best model
            if val_metrics['emr'] > self.best_val_emr:
                self.best_val_emr = val_metrics['emr']
                self.best_model_state = self.model.state_dict().copy()

            print(f"Phase 2 Epoch {epoch+1}: Val EMR = {val_metrics['emr']:.4f}")

    def _train_epoch(self, optimizer, epoch, phase):
        """Train for one epoch"""
        self.model.train()

        train_loader = self.data_module.train_dataloader()
        metrics = MultiOutputMetrics()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"{phase} Epoch {epoch}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)

            # Backward pass
            loss_dict['total_loss'].backward()
            optimizer.step()

            # Update metrics
            loss_meter.update(loss_dict['total_loss'].item(), images.size(0))
            metrics.update(predictions, targets)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # Compute final metrics
        final_metrics = metrics.compute()
        final_metrics['loss'] = loss_meter.avg

        return final_metrics

    def _validate_epoch(self, epoch, phase):
        """Validate for one epoch"""
        self.model.eval()

        val_loader = self.data_module.val_dataloader()
        metrics = MultiOutputMetrics()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"{phase} Val {epoch}"):
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                # Forward pass
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)

                # Update metrics
                loss_meter.update(loss_dict['total_loss'].item(), images.size(0))
                metrics.update(predictions, targets)

        # Compute final metrics
        final_metrics = metrics.compute()
        final_metrics['loss'] = loss_meter.avg

        return final_metrics

    def _update_history(self, train_metrics, val_metrics):
        """Update training history"""
        self.train_history['loss'].append(train_metrics['loss'])
        self.train_history['make_acc'].append(train_metrics['make_accuracy'])
        self.train_history['model_acc'].append(train_metrics['model_accuracy'])
        self.train_history['year_acc'].append(train_metrics['year_accuracy'])
        self.train_history['emr'].append(train_metrics['emr'])

        self.val_history['loss'].append(val_metrics['loss'])
        self.val_history['make_acc'].append(val_metrics['make_accuracy'])
        self.val_history['model_acc'].append(val_metrics['model_accuracy'])
        self.val_history['year_acc'].append(val_metrics['year_accuracy'])
        self.val_history['emr'].append(val_metrics['emr'])

    def test(self):
        """Test the best model"""
        print("\n" + "="*50)
        print("TESTING BEST MODEL")
        print("="*50)

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        test_loader = self.data_module.test_dataloader()
        metrics = MultiOutputMetrics()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)

                loss_meter.update(loss_dict['total_loss'].item(), images.size(0))
                metrics.update(predictions, targets)

        test_metrics = metrics.compute()
        test_metrics['loss'] = loss_meter.avg

        print("\nTest Results:")
        print(f"Loss: {test_metrics['loss']:.4f}")
        print(f"Make Accuracy: {test_metrics['make_accuracy']:.4f}")
        print(f"Model Accuracy: {test_metrics['model_accuracy']:.4f}")
        print(f"Year Accuracy: {test_metrics['year_accuracy']:.4f}")
        print(f"EMR: {test_metrics['emr']:.4f}")
        print(f"Overall Accuracy: {test_metrics['overall_accuracy']:.4f}")

        return test_metrics

    def save_model(self, filepath):
        """Save the best model"""
        if self.best_model_state is not None:
            torch.save({
                'model_state_dict': self.best_model_state,
                'config': self.config,
                'class_info': self.data_module.get_class_info(),
                'best_val_emr': self.best_val_emr
            }, filepath)
            print(f"Model saved to {filepath}")

    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.train_history['loss'], label='Train')
        axes[0, 0].plot(self.val_history['loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()

        # EMR
        axes[0, 1].plot(self.train_history['emr'], label='Train')
        axes[0, 1].plot(self.val_history['emr'], label='Validation')
        axes[0, 1].set_title('Exact Match Ratio (EMR)')
        axes[0, 1].legend()

        # Make Accuracy
        axes[1, 0].plot(self.train_history['make_acc'], label='Train')
        axes[1, 0].plot(self.val_history['make_acc'], label='Validation')
        axes[1, 0].set_title('Make Accuracy')
        axes[1, 0].legend()

        # Model Accuracy
        axes[1, 1].plot(self.train_history['model_acc'], label='Train')
        axes[1, 1].plot(self.val_history['model_acc'], label='Validation')
        axes[1, 1].set_title('Model Accuracy')
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Training plots saved to {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train Car Classifier')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/raw',
                        help='Path to dataset')

    args = parser.parse_args()

    # Default configuration
    config = {
        'data_path': args.data_path,
        'batch_size': 16,
        'image_size': 128,
        'num_workers': 4,
        'backbone': 'resnet50',
        'pretrained': True,
        'phase1_epochs': 2,
        'phase2_epochs': 3,
        'phase1_lr': 0.001,
        'phase2_lr': 0.0001,
        'loss_weights': {
            'make': 1.0,
            'model': 1.0,
            'year': 1.0
        }
    }

    # Load config file if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)

    print("Training Configuration:")
    print(json.dumps(config, indent=2))

    # Initialize trainer
    trainer = CarTrainer(config)

    # Train
    trainer.train_phase_1()
    trainer.train_phase_2()

    # Test
    test_metrics = trainer.test()

    # Save results
    os.makedirs('results', exist_ok=True)
    trainer.save_model('results/best_model.pth')
    trainer.plot_training_history('results/training_history.png')

    # Save test results
    with open('results/test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print("Training completed!")


if __name__ == "__main__":
    main()


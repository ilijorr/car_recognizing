#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cached_dataset import CachedCarDataModule
from hierarchical_model import HierarchicalCarClassifier, HierarchicalLoss, create_make_to_models_mapping
from metrics import MultiOutputMetrics, AverageMeter


class HierarchicalTrainer:
    def __init__(self, config):
        """
        Hierarchical car classifier trainer

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize data module
        cache_file = config.get('cache_file', 'cache_consolidated/dataset_cache.pkl')

        print(f"🚀 Using cached dataset from {cache_file}")
        self.data_module = CachedCarDataModule(
            data_path=config['data_path'],
            cache_file=cache_file,
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            num_workers=config['num_workers']
        )

        # Get class information
        class_info = self.data_module.get_class_info()
        self.num_makes = class_info['num_makes']
        self.num_models = class_info['num_models']
        self.num_years = class_info['num_years']

        # Create make-to-models mapping
        all_samples = (self.data_module.train_samples +
                      self.data_module.val_samples +
                      self.data_module.test_samples)
        self.make_to_models = create_make_to_models_mapping(all_samples)

        print(f"📊 Dataset: {self.num_makes} makes, {self.num_models} models, {self.num_years} years")
        print(f"🗂️ Make-to-models mapping: {len(self.make_to_models)} makes")

        # Initialize hierarchical model
        self.model = HierarchicalCarClassifier(
            num_makes=self.num_makes,
            num_years=self.num_years,
            make_to_models=self.make_to_models,
            pretrained=config['pretrained']
        ).to(self.device)

        # Initialize loss function
        self.criterion = HierarchicalLoss(
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

    def train_stage(self, stage: str, epochs: int, lr: float):
        """
        Train specific stage of hierarchical model

        Args:
            stage: 'make', 'model', 'year', or 'all'
            epochs: Number of epochs to train
            lr: Learning rate
        """
        print(f"\n{'='*50}")
        print(f"TRAINING STAGE: {stage.upper()}")
        print(f"{'='*50}")

        # Setup optimizer based on stage
        if stage == 'make':
            optimizer = optim.Adam(self.model.make_classifier.parameters(), lr=lr)
        elif stage == 'model':
            optimizer = optim.Adam(self.model.model_classifier.parameters(), lr=lr)
        elif stage == 'year':
            optimizer = optim.Adam(self.model.year_classifier.parameters(), lr=lr)
        else:  # 'all'
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        # Train for specified epochs
        for epoch in range(epochs):
            train_metrics = self._train_epoch(optimizer, epoch + 1, stage)
            val_metrics = self._validate_epoch(epoch + 1, stage)

            # Update learning rate
            scheduler.step(val_metrics.get('emr', 0))

            # Save history
            self._update_history(train_metrics, val_metrics)

            # Check for best model
            current_emr = val_metrics.get('emr', 0)
            if current_emr > self.best_val_emr:
                self.best_val_emr = current_emr
                self.best_model_state = self.model.state_dict().copy()

            print(f"{stage.capitalize()} Epoch {epoch+1}: Val EMR = {current_emr:.4f}")

    def _train_epoch(self, optimizer, epoch, stage):
        """Train for one epoch"""
        self.model.train()

        train_loader = self.data_module.train_dataloader()
        metrics = MultiOutputMetrics()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"{stage.capitalize()} Train {epoch}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            # Prepare make names for model classifier
            make_names = []
            if stage in ['model', 'all']:
                make_ids = targets['make'].cpu().numpy()
                make_names = [self.data_module.make_encoder.inverse_transform([mid])[0]
                             for mid in make_ids]

            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(images, targets['make'], make_names, stage=stage)

            # Compute loss only for current stage
            stage_targets = {}
            if stage == 'make' or stage == 'all':
                stage_targets['make'] = targets['make']
            if stage == 'model' or stage == 'all':
                if 'model' in predictions:
                    # For model stage, we need to remap targets to make-specific indices
                    stage_targets['model'] = self._remap_model_targets(targets['model'], make_names)
            if stage == 'year' or stage == 'all':
                stage_targets['year'] = targets['year']

            loss_dict = self.criterion(predictions, stage_targets)

            # Backward pass
            loss_dict['total_loss'].backward()
            optimizer.step()

            # Update metrics
            loss_meter.update(loss_dict['total_loss'].item(), images.size(0))

            # Only compute metrics for complete predictions
            if stage == 'all':
                metrics.update(predictions, targets)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # Compute final metrics
        final_metrics = {'loss': loss_meter.avg}
        if stage == 'all':
            computed_metrics = metrics.compute()
            final_metrics.update(computed_metrics)

        return final_metrics

    def _remap_model_targets(self, global_model_targets, make_names):
        """Remap global model targets to make-specific model indices"""
        batch_size = global_model_targets.size(0)
        remapped_targets = torch.full_like(global_model_targets, -1)  # Use -1 as invalid marker

        for i, make_name in enumerate(make_names):
            if make_name in self.make_to_models:
                # Get the global model name
                global_model_idx = global_model_targets[i].item()
                try:
                    global_model_name = self.data_module.model_encoder.inverse_transform([global_model_idx])[0]

                    # Find the make-specific index
                    make_models = self.make_to_models[make_name]
                    if global_model_name in make_models:
                        make_specific_idx = make_models.index(global_model_name)
                        # Double check - this should always be true, but being safe
                        if 0 <= make_specific_idx < len(make_models):
                            remapped_targets[i] = make_specific_idx
                        else:
                            print(f"ERROR: Index {make_specific_idx} out of bounds for {make_name} ({len(make_models)} models)")
                            remapped_targets[i] = -1  # Mark as invalid
                    else:
                        # Model not found in this make - this indicates data inconsistency
                        print(f"ERROR: Model {global_model_name} not found in {make_name} models")
                        print(f"Available models for {make_name}: {make_models[:5]}...")  # Show first 5
                        remapped_targets[i] = -1  # Mark as invalid
                except Exception as e:
                    print(f"ERROR: Exception remapping model target {global_model_idx}: {e}")
                    remapped_targets[i] = -1  # Mark as invalid
            else:
                print(f"ERROR: Make {make_name} not found in make_to_models mapping")
                remapped_targets[i] = -1  # Mark as invalid

        return remapped_targets

    def _validate_epoch(self, epoch, stage):
        """Validate for one epoch"""
        self.model.eval()

        val_loader = self.data_module.val_dataloader()
        metrics = MultiOutputMetrics()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"{stage.capitalize()} Val {epoch}"):
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                # Prepare make names for model classifier
                make_names = []
                if stage in ['model', 'all']:
                    make_ids = targets['make'].cpu().numpy()
                    make_names = [self.data_module.make_encoder.inverse_transform([mid])[0]
                                 for mid in make_ids]

                # Forward pass
                predictions = self.model(images, targets['make'], make_names, stage=stage)

                # Compute loss only for current stage
                stage_targets = {}
                if stage == 'make' or stage == 'all':
                    stage_targets['make'] = targets['make']
                if stage == 'model' or stage == 'all':
                    if 'model' in predictions:
                        # For model stage, we need to remap targets to make-specific indices
                        stage_targets['model'] = self._remap_model_targets(targets['model'], make_names)
                if stage == 'year' or stage == 'all':
                    stage_targets['year'] = targets['year']

                loss_dict = self.criterion(predictions, stage_targets)

                # Update metrics
                loss_meter.update(loss_dict['total_loss'].item(), images.size(0))

                # Only compute metrics for complete predictions
                if stage == 'all':
                    metrics.update(predictions, targets)

        # Compute final metrics
        final_metrics = {'loss': loss_meter.avg}
        if stage == 'all':
            computed_metrics = metrics.compute()
            final_metrics.update(computed_metrics)

        return final_metrics

    def _update_history(self, train_metrics, val_metrics):
        """Update training history"""
        self.train_history['loss'].append(train_metrics['loss'])
        self.val_history['loss'].append(val_metrics['loss'])

        if 'make_accuracy' in train_metrics:
            self.train_history['make_acc'].append(train_metrics['make_accuracy'])
            self.train_history['model_acc'].append(train_metrics['model_accuracy'])
            self.train_history['year_acc'].append(train_metrics['year_accuracy'])
            self.train_history['emr'].append(train_metrics['emr'])

        if 'make_accuracy' in val_metrics:
            self.val_history['make_acc'].append(val_metrics['make_accuracy'])
            self.val_history['model_acc'].append(val_metrics['model_accuracy'])
            self.val_history['year_acc'].append(val_metrics['year_accuracy'])
            self.val_history['emr'].append(val_metrics['emr'])

    def test(self):
        """Test the best model"""
        print(f"\n{'='*50}")
        print("TESTING BEST MODEL")
        print(f"{'='*50}")

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

                # Prepare make names
                make_ids = targets['make'].cpu().numpy()
                make_names = [self.data_module.make_encoder.inverse_transform([mid])[0]
                             for mid in make_ids]

                # Forward pass
                predictions = self.model(images, targets['make'], make_names, stage='all')

                # Remap model targets for hierarchical loss computation
                test_targets = targets.copy()
                if 'model' in predictions:
                    test_targets['model'] = self._remap_model_targets(targets['model'], make_names)

                loss_dict = self.criterion(predictions, test_targets)

                loss_meter.update(loss_dict['total_loss'].item(), images.size(0))
                # Use original targets for metrics computation (not remapped)
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

    def save_model(self, filepath):
        """Save the best model"""
        if self.best_model_state is not None:
            torch.save({
                'model_state_dict': self.best_model_state,
                'config': self.config,
                'class_info': self.data_module.get_class_info(),
                'make_to_models': self.make_to_models,
                'best_val_emr': self.best_val_emr
            }, filepath)
            print(f"Model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Car Classifier')
    parser.add_argument('--config', type=str, default='configs/hierarchical.json',
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, default='data/raw',
                        help='Path to dataset')
    parser.add_argument('--cache_file', type=str, default='cache_consolidated/dataset_cache.pkl',
                        help='Path to dataset cache file')

    args = parser.parse_args()

    # Default configuration optimized for hierarchical training
    config = {
        'data_path': args.data_path,
        'cache_file': args.cache_file,
        'batch_size': 32,
        'image_size': 224,
        'num_workers': 8,
        'pretrained': True,

        # Stage-specific training
        'make_epochs': 5,
        'model_epochs': 8,
        'year_epochs': 5,
        'final_epochs': 10,

        # Learning rates
        'make_lr': 0.001,
        'model_lr': 0.0005,
        'year_lr': 0.001,
        'final_lr': 0.0001,

        # Loss weights
        'loss_weights': {
            'make': 1.0,
            'model': 2.0,
            'year': 1.0
        }
    }

    # Load config file if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)

    print("Hierarchical Training Configuration:")
    print(json.dumps(config, indent=2))

    # Initialize trainer
    trainer = HierarchicalTrainer(config)

    # Stage-wise training
    print("\n🎯 Starting hierarchical training...")

    # Stage 1: Train make classifier
    trainer.train_stage('make', config['make_epochs'], config['make_lr'])

    # Stage 2: Train model classifier
    trainer.train_stage('model', config['model_epochs'], config['model_lr'])

    # Stage 3: Train year classifier
    trainer.train_stage('year', config['year_epochs'], config['year_lr'])

    # Stage 4: Fine-tune all together
    trainer.train_stage('all', config['final_epochs'], config['final_lr'])

    # Test
    test_metrics = trainer.test()

    # Save results
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # Save model
    model_path = results_dir / 'hierarchical_model.pth'
    trainer.save_model(str(model_path))

    # Save test results
    results_file = results_dir / 'hierarchical_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Plot and save training history
    plot_path = results_dir / 'hierarchical_training_history.png'
    trainer.plot_training_history(str(plot_path))

    print("Hierarchical training completed!")


if __name__ == "__main__":
    main()
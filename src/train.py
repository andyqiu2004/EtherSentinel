import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import os
from models.gnn_bert import GNNBERT, TransactionClassifier, AccountAnalyzer
from typing import Dict, List, Tuple

class ModelTrainer:
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, 
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in progress_bar:
            # Prepare batch data
            x, edge_index, y = self._prepare_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x, edge_index)
            loss = self.criterion(output, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Prepare batch data
                x, edge_index, y = self._prepare_batch(batch)
                
                # Forward pass
                output = self.model(x, edge_index)
                loss = self.criterion(output, y)
                
                # Update statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += int((pred == y).sum())
                total += y.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }
    
    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data for model input"""
        x, edge_index, y = batch
        return x.to(self.device), edge_index.to(self.device), y.to(self.device)
    
    def save_checkpoint(self, 
                       epoch: int,
                       metrics: Dict[str, float],
                       save_dir: str = 'models/checkpoints'):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'checkpoint_epoch_{epoch}_{timestamp}.pt'
        path = os.path.join(save_dir, filename)
        
        torch.save(checkpoint, path)
        self.logger.info(f'Saved checkpoint to {path}')
    
    def load_checkpoint(self, 
                       path: str,
                       device: str = None):
        """Load model checkpoint"""
        if device is None:
            device = self.device
            
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f'Loaded checkpoint from {path}')
        return checkpoint['epoch'], checkpoint['metrics']

def train_model(train_loader: DataLoader,
                val_loader: DataLoader,
                model: nn.Module,
                num_epochs: int = 100,
                save_dir: str = 'models/checkpoints',
                early_stopping_patience: int = 10):
    """Train the model with early stopping"""
    trainer = ModelTrainer(model)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Log metrics
        logging.info(f'Epoch {epoch}:')
        logging.info(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                    f'Accuracy: {train_metrics["accuracy"]:.2f}%')
        logging.info(f'Val - Loss: {val_metrics["loss"]:.4f}, '
                    f'Accuracy: {val_metrics["accuracy"]:.2f}%')
        
        # Save checkpoint
        trainer.save_checkpoint(epoch, {
            'train': train_metrics,
            'val': val_metrics
        }, save_dir)
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize model
    num_features = 4  # Adjust based on your feature set
    num_classes = 2   # Adjust based on your classification task
    model = GNNBERT(num_features, num_classes)
    
    # TODO: Load your data and create DataLoader instances
    # train_loader = ...
    # val_loader = ...
    
    # Train model
    # train_model(train_loader, val_loader, model) 
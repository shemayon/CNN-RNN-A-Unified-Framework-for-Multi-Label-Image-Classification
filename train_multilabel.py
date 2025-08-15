import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import CNNRNNMultiLabel
from data_loader import get_multilabel_loader

class MultiLabelLoss(nn.Module):
    """
    Custom loss function for multi-label classification
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super(MultiLabelLoss, self).__init__()
        self.alpha = alpha  # Weight for positive samples
        self.beta = beta    # Weight for negative samples
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size, num_classes]
        """
        # Calculate BCE loss
        bce_loss = self.bce(outputs, targets)
        
        # Apply different weights to positive and negative samples
        positive_mask = (targets == 1).float()
        negative_mask = (targets == 0).float()
        
        weighted_loss = (self.alpha * positive_mask * bce_loss + 
                        self.beta * negative_mask * bce_loss)
        
        return weighted_loss.mean()

def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    Evaluate the model on validation/test set
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _, _ in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs, _ = model(images)  # No label sequences for evaluation
            predictions = (torch.sigmoid(outputs) > threshold).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='micro', zero_division=0
    )
    
    # Calculate accuracy (exact match)
    accuracy = accuracy_score(all_targets, all_predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs, save_dir, threshold=0.5):
    """
    Train the CNN-RNN model
    """
    best_f1 = 0.0
    train_losses = []
    val_metrics = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, targets, label_sequences, label_lengths) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            label_lengths = label_lengths.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(images, label_sequences, label_lengths)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_metrics_dict = evaluate_model(model, val_loader, device, threshold)
        val_metrics.append(val_metrics_dict)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Precision: {val_metrics_dict["precision"]:.4f}')
        print(f'  Val Recall: {val_metrics_dict["recall"]:.4f}')
        print(f'  Val F1: {val_metrics_dict["f1"]:.4f}')
        print(f'  Val Accuracy: {val_metrics_dict["accuracy"]:.4f}')
        
        # Save best model
        if val_metrics_dict['f1'] > best_f1:
            best_f1 = val_metrics_dict['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics_dict
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'  New best model saved with F1: {best_f1:.4f}')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_metrics': val_metrics
        }, os.path.join(save_dir, 'checkpoint.pth'))
    
    return train_losses, val_metrics

def plot_training_curves(train_losses, val_metrics, save_dir):
    """
    Plot training curves
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Validation metrics
    epochs = range(1, len(val_metrics) + 1)
    precision = [m['precision'] for m in val_metrics]
    recall = [m['recall'] for m in val_metrics]
    f1 = [m['f1'] for m in val_metrics]
    accuracy = [m['accuracy'] for m in val_metrics]
    
    ax2.plot(epochs, precision, label='Precision')
    ax2.plot(epochs, recall, label='Recall')
    ax2.plot(epochs, f1, label='F1')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    ax3.plot(epochs, f1)
    ax3.set_title('Validation F1 Score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    
    ax4.plot(epochs, accuracy)
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train CNN-RNN for Multi-Label Classification')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--annotation_file', type=str, default=None,
                        help='Path to annotation file (JSON)')
    parser.add_argument('--num_classes', type=int, default=80,
                        help='Number of classes')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size for RNN')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of RNN layers')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (minimum 2 for training)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device configuration - prioritize GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        device = torch.device('cpu')
        print('GPU not available, using CPU')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ensure minimum batch size for training
    batch_size = max(2, args.batch_size)
    if batch_size != args.batch_size:
        print(f"Warning: Batch size adjusted from {args.batch_size} to {batch_size} for training stability")
    
    # Create data loaders
    train_loader = get_multilabel_loader(
        args.image_dir, args.annotation_file, transform,
        batch_size=batch_size, shuffle=True, num_workers=4,
        num_classes=args.num_classes
    )
    
    # For simplicity, use the same data for validation (in practice, you'd split the data)
    val_loader = get_multilabel_loader(
        args.image_dir, args.annotation_file, transform,
        batch_size=batch_size, shuffle=False, num_workers=4,
        num_classes=args.num_classes
    )
    
    # Create model
    model = CNNRNNMultiLabel(
        num_classes=args.num_classes,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Using batch size: {batch_size}")
    
    # Loss function and optimizer
    criterion = MultiLabelLoss(alpha=1.0, beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    print('Starting training...')
    train_losses, val_metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        device, args.num_epochs, args.save_dir, args.threshold
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_metrics, args.save_dir)
    
    print('Training completed!')
    print(f'Best F1 Score: {max([m["f1"] for m in val_metrics]):.4f}')

if __name__ == '__main__':
    main() 
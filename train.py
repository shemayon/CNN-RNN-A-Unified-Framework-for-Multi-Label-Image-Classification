import argparse
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from build_vocab import Vocabulary
from data_loader import CaptionDataset, get_loader  # Update the import statement
from model import CaptionModel

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a captioning model')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image for training')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path to the vocabulary file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of the data loader')
    args = parser.parse_args()

    # Load the vocabulary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Define the image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the image dataset and data loader
    image_dataset = CaptionDataset(args.image_path, vocab, transform=transform)
    data_loader = get_loader(args.image_path, vocab, transform,
                             batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Create the captioning model
    model = CaptionModel(vocab_size=len(vocab), embedding_size=256,
                         hidden_size=512, num_layers=2)

    # Send the model to the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    for epoch in range(args.num_epochs):
        for i, (images, targets, lengths) in enumerate(data_loader):
            # Move the images and targets to the GPU
            images = images.to(device)
            targets = targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(images, lengths)

            # Calculate the loss
            loss = criterion(outputs.reshape(-1, len(vocab)), targets.reshape(-1))

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Print the loss
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    main()

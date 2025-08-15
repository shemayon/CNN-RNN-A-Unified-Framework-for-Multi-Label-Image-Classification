#!/usr/bin/env python3
"""
Test script to demonstrate model predictions
"""

import torch
import json
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

from model import CNNRNNMultiLabel

def load_model_and_predict():
    """Load the trained model and make predictions"""
    
    # Configuration
    model_path = "./checkpoints/best_model.pth"
    num_classes = 10
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using:")
        print("python train_multilabel.py --image_dir ./images --num_classes 10 --batch_size 4 --num_epochs 5")
        return
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading trained model...")
    model = CNNRNNMultiLabel(
        num_classes=num_classes,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"Best F1 Score: {checkpoint.get('best_f1', 'N/A')}")
    
    # Load annotations to see what labels were assigned
    annotations_file = "./test_annotations.json"
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        print(f"✅ Loaded annotations for {len(annotations)} images")
    else:
        annotations = {}
        print("⚠️ No annotations file found")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test on each image
    image_dir = "./images"
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n=== Making Predictions on {len(image_files)} Images ===")
    
    for i, image_file in enumerate(image_files):
        print(f"\n--- Image {i+1}: {image_file} ---")
        
        # Load and preprocess image
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs, _ = model(image_tensor)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Get predicted class indices
            predicted_indices = torch.where(predictions[0] == 1)[0].cpu().numpy()
            confidence_scores = probabilities[0][predicted_indices].cpu().numpy()
        
        # Display results
        print(f"Predicted labels: {predicted_indices}")
        print(f"Confidence scores: {confidence_scores}")
        
        # Compare with ground truth if available
        if image_file in annotations:
            gt_labels = annotations[image_file]['labels']
            print(f"Ground truth labels: {gt_labels}")
            
            # Calculate overlap
            overlap = set(predicted_indices) & set(gt_labels)
            print(f"Correct predictions: {len(overlap)}/{len(gt_labels)}")
        
        # Show top 3 predictions by confidence
        all_scores = probabilities[0].cpu().numpy()
        top_indices = np.argsort(all_scores)[-3:][::-1]
        print(f"Top 3 predictions:")
        for j, idx in enumerate(top_indices):
            print(f"  {j+1}. Class {idx}: {all_scores[idx]:.3f}")
    
    print(f"\n=== Prediction Summary ===")
    print(f"Model successfully made predictions on {len(image_files)} images")
    print(f"Model file: {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def visualize_predictions():
    """Create a visualization of predictions"""
    
    # Load model
    model_path = "./checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train first.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNRNNMultiLabel(num_classes=10, embed_size=256, hidden_size=256).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test on one image
    image_dir = "./images"
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found")
        return
    
    # Use first image
    image_file = image_files[0]
    image_path = os.path.join(image_dir, image_file)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs, _ = model(image_tensor)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).float()
        
        predicted_indices = torch.where(predictions[0] == 1)[0].cpu().numpy()
        confidence_scores = probabilities[0][predicted_indices].cpu().numpy()
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(image)
    ax1.set_title(f'Input Image: {image_file}')
    ax1.axis('off')
    
    # Show predictions
    if len(predicted_indices) > 0:
        y_pos = np.arange(len(predicted_indices))
        ax2.barh(y_pos, confidence_scores)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'Class {i}' for i in predicted_indices])
        ax2.set_xlabel('Confidence Score')
        ax2.set_title('Predicted Labels')
        ax2.set_xlim(0, 1)
        
        # Add confidence scores as text
        for i, (idx, score) in enumerate(zip(predicted_indices, confidence_scores)):
            ax2.text(score + 0.01, i, f'{score:.3f}', va='center')
    else:
        ax2.text(0.5, 0.5, 'No labels predicted\n(threshold too high)', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Predicted Labels')
    
    plt.tight_layout()
    plt.savefig('./prediction_demo.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'prediction_demo.png'")
    plt.show()

if __name__ == "__main__":
    print("CNN-RNN Prediction Test")
    print("=" * 40)
    
    # Test predictions
    load_model_and_predict()
    
    # Create visualization
    print(f"\nCreating visualization...")
    visualize_predictions()
    
    print(f"\n🎉 Testing completed!")
    print(f"Next steps:")
    print(f"1. Check the predictions above")
    print(f"2. Look at 'prediction_demo.png' for visualization")
    print(f"3. Use 'predict_multilabel.py' for more detailed predictions") 
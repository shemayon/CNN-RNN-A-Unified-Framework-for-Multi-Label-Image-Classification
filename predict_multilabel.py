import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from model import CNNRNNMultiLabel

def load_model(model_path, num_classes, embed_size=512, hidden_size=512, num_layers=1, device='cpu'):
    """
    Load a trained CNN-RNN model
    """
    model = CNNRNNMultiLabel(
        num_classes=num_classes,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def preprocess_image(image_path, transform=None):
    """
    Preprocess image for model input
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image

def predict_labels(model, image_tensor, device, threshold=0.5, class_names=None):
    """
    Predict labels for a single image
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs, hidden_states = model(image_tensor)
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs)
        
        # Get predictions based on threshold
        predictions = (probabilities > threshold).float()
        
        # Get predicted class indices
        predicted_indices = torch.where(predictions[0] == 1)[0].cpu().numpy()
        
        # Get confidence scores for predicted classes
        confidence_scores = probabilities[0][predicted_indices].cpu().numpy()
        
        # Create results dictionary
        results = {
            'predicted_indices': predicted_indices.tolist(),
            'confidence_scores': confidence_scores.tolist(),
            'all_probabilities': probabilities[0].cpu().numpy().tolist()
        }
        
        # Add class names if provided
        if class_names is not None:
            results['predicted_classes'] = [class_names[i] for i in predicted_indices]
        
        return results

def visualize_predictions(image, results, class_names=None, save_path=None):
    """
    Visualize image with predicted labels
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Display predictions
    if class_names is not None and 'predicted_classes' in results:
        labels = results['predicted_classes']
        scores = results['confidence_scores']
        
        # Create bar plot of confidence scores
        y_pos = np.arange(len(labels))
        ax2.barh(y_pos, scores)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Confidence Score')
        ax2.set_title('Predicted Labels')
        ax2.set_xlim(0, 1)
        
        # Add confidence scores as text
        for i, (label, score) in enumerate(zip(labels, scores)):
            ax2.text(score + 0.01, i, f'{score:.3f}', va='center')
    else:
        # Display indices and scores
        indices = results['predicted_indices']
        scores = results['confidence_scores']
        
        y_pos = np.arange(len(indices))
        ax2.barh(y_pos, scores)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'Class {i}' for i in indices])
        ax2.set_xlabel('Confidence Score')
        ax2.set_title('Predicted Labels')
        ax2.set_xlim(0, 1)
        
        # Add confidence scores as text
        for i, (idx, score) in enumerate(zip(indices, scores)):
            ax2.text(score + 0.01, i, f'{score:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Visualization saved to {save_path}')
    
    plt.show()

def batch_predict(model, image_dir, device, threshold=0.5, class_names=None, save_dir=None):
    """
    Predict labels for all images in a directory
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = {}
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # Preprocess image
            image_tensor, original_image = preprocess_image(image_path, transform)
            
            # Predict labels
            prediction_results = predict_labels(model, image_tensor, device, threshold, class_names)
            
            # Store results
            results[image_file] = prediction_results
            
            # Visualize if save_dir is provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{os.path.splitext(image_file)[0]}_predictions.png')
                visualize_predictions(original_image, prediction_results, class_names, save_path)
            
            print(f'Processed {image_file}: {len(prediction_results["predicted_indices"])} labels predicted')
            
        except Exception as e:
            print(f'Error processing {image_file}: {str(e)}')
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Predict labels using trained CNN-RNN model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--num_classes', type=int, default=80,
                        help='Number of classes')
    parser.add_argument('--embed_size', type=int, default=512,
                        help='Embedding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden size for RNN')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of RNN layers')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--class_names_file', type=str, default=None,
                        help='Path to JSON file containing class names')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save visualizations')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process all images in directory')
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load class names if provided
    class_names = None
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r') as f:
            class_names = json.load(f)
        print(f'Loaded {len(class_names)} class names')
    
    # Load model
    print('Loading model...')
    model = load_model(
        args.model_path, args.num_classes, args.embed_size,
        args.hidden_size, args.num_layers, device
    )
    print('Model loaded successfully!')
    
    if args.batch_mode:
        # Batch prediction mode
        if not os.path.isdir(args.image_path):
            print(f'Error: {args.image_path} is not a directory')
            return
        
        print(f'Processing all images in {args.image_path}...')
        results = batch_predict(
            model, args.image_path, device, args.threshold,
            class_names, args.save_dir
        )
        
        # Save results to JSON
        if args.save_dir:
            results_file = os.path.join(args.save_dir, 'batch_predictions.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f'Results saved to {results_file}')
    
    else:
        # Single image prediction mode
        if not os.path.isfile(args.image_path):
            print(f'Error: {args.image_path} is not a file')
            return
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(args.image_path)
        
        # Predict labels
        results = predict_labels(model, image_tensor, device, args.threshold, class_names)
        
        # Print results
        print(f'\nPrediction Results for {args.image_path}:')
        print(f'Number of predicted labels: {len(results["predicted_indices"])}')
        
        if class_names is not None and 'predicted_classes' in results:
            for i, (class_name, score) in enumerate(zip(results['predicted_classes'], results['confidence_scores'])):
                print(f'  {i+1}. {class_name}: {score:.3f}')
        else:
            for i, (idx, score) in enumerate(zip(results['predicted_indices'], results['confidence_scores'])):
                print(f'  {i+1}. Class {idx}: {score:.3f}')
        
        # Visualize results
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'prediction_visualization.png')
            visualize_predictions(original_image, results, class_names, save_path)
        else:
            visualize_predictions(original_image, results, class_names)

if __name__ == '__main__':
    main() 
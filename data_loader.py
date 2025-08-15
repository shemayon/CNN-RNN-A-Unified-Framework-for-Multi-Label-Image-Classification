import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
import random

class MultiLabelDataset(Dataset):
    """
    Dataset for multi-label image classification
    """
    def __init__(self, image_dir, annotation_file=None, transform=None, num_classes=80):
        self.image_dir = image_dir
        self.transform = transform
        self.num_classes = num_classes
        
        # Load annotations
        if annotation_file is not None and os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            # Create dummy annotations for demo
            self.annotations = self._create_dummy_annotations()
        
        self.image_files = list(self.annotations.keys())
    
    def _create_dummy_annotations(self):
        """Create dummy annotations for demonstration"""
        annotations = {}
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            # Randomly assign 1-5 labels per image
            num_labels = random.randint(1, 5)
            labels = random.sample(range(self.num_classes), num_labels)
            annotations[img_file] = {
                'labels': labels,
                'label_names': [f'class_{i}' for i in labels]
            }
        
        return annotations
    
    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Get labels
        annotation = self.annotations[image_name]
        labels = annotation['labels']
        
        # Create multi-label target vector (keep original labels for targets)
        target = torch.zeros(self.num_classes)
        target[labels] = 1.0
        
        # Create label sequence for RNN training (sorted by label index)
        # Add 1 to all labels to avoid 0 (which will be used for padding)
        label_sequence = torch.tensor(sorted(labels), dtype=torch.long) + 1
        label_length = len(labels)
        
        return image, target, label_sequence, label_length
    
    def __len__(self):
        return len(self.image_files)

def collate_fn(batch):
    """
    Custom collate function for variable length label sequences
    """
    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    label_sequences = [item[2] for item in batch]  # Keep as list for variable length
    label_lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)
    
    return images, targets, label_sequences, label_lengths

def get_multilabel_loader(image_dir, annotation_file, transform=None, 
                         batch_size=32, shuffle=True, num_workers=4, 
                         num_classes=80):
    """
    Create a DataLoader for multi-label classification
    
    Args:
        image_dir: Directory containing images
        annotation_file: Path to annotation file (JSON)
        transform: Image transformations
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        num_classes: Number of classes
    
    Returns:
        DataLoader for multi-label classification
    """
    dataset = MultiLabelDataset(image_dir, annotation_file, transform, num_classes)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return loader



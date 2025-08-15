# CNN-RNN: A Unified Framework for Multi-Label Image Classification

This repository implements the CNN-RNN framework for multi-label image classification as described in the paper: [CNN-RNN: A Unified Framework for Multi-Label Image Classification](https://arxiv.org/pdf/1604.04573).

## Overview

The CNN-RNN framework combines Convolutional Neural Networks (CNNs) for feature extraction with Recurrent Neural Networks (RNNs) for modeling label dependencies. This approach learns a joint image-label embedding space that captures both visual features and semantic label relationships.

### Key Features

- **CNN Encoder**: Uses ResNet-152 for robust feature extraction
- **RNN Decoder**: LSTM-based model for capturing label dependencies
- **Joint Embedding**: Unified space for images and labels
- **Multi-Label Classification**: Handles multiple labels per image
- **Label Dependency Modeling**: RNN learns relationships between labels

## Architecture

The model consists of three main components:

1. **CNN Encoder**: Extracts visual features from input images using ResNet-152
2. **Feature Projection**: Maps CNN features to a shared embedding space
3. **RNN Decoder**: Processes label sequences to model dependencies and generate predictions

![image](https://github.com/shemayon/CNN-RNN-A-Unified-Framework-for-Multi-Label-Image-Classification/architecture)

### Key Interactions:

1. **Training Flow**:
   - User provides images and annotations
   - Model processes both images and label sequences
   - RNN learns label dependencies during training
   - Optimizer updates weights based on loss

2. **Prediction Flow**:
   - User provides new images
   - Model uses only image features for inference
   - RNN generates predictions based on learned dependencies
   - Results include confidence scores for each class

3. **Data Flow**:
   - Images: RGB → CNN → Feature Vector → RNN → Predictions
   - Labels: Label Sequence → Embedding → RNN → Dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CNN-RNN-A-Unified-Framework-for-Multi-Label-Image-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the CNN-RNN model:

```bash
python train_multilabel.py \
    --image_dir ./images \
    --annotation_file ./annotations.json \
    --num_classes 80 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --save_dir ./checkpoints
```

**Parameters:**
- `--image_dir`: Directory containing training images
- `--annotation_file`: JSON file with image annotations (optional, will create dummy data if not provided)
- `--num_classes`: Number of classes in your dataset
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate for optimization
- `--num_epochs`: Number of training epochs
- `--save_dir`: Directory to save model checkpoints

### Prediction

To predict labels for a single image:

```bash
python predict_multilabel.py \
    --model_path ./checkpoints/best_model.pth \
    --image_path ./test_image.jpg \
    --num_classes 80 \
    --threshold 0.5
```

To predict labels for all images in a directory:

```bash
python predict_multilabel.py \
    --model_path ./checkpoints/best_model.pth \
    --image_path ./test_images/ \
    --num_classes 80 \
    --threshold 0.5 \
    --batch_mode \
    --save_dir ./predictions
```

### Data Format

#### Image Directory
Place your training images in a directory. Supported formats: `.jpg`, `.jpeg`, `.png`

#### Annotation File (Optional)
If you have annotations, provide them in JSON format:

```json
{
    "image1.jpg": {
        "labels": [0, 5, 12],
        "label_names": ["person", "car", "building"]
    },
    "image2.jpg": {
        "labels": [3, 8],
        "label_names": ["dog", "cat"]
    }
}
```

If no annotation file is provided, the system will create dummy annotations for demonstration.

## Model Details

### CNN-RNN Architecture

The model implements the following key components:

1. **ResNet-152 Encoder**: Pre-trained CNN for feature extraction
2. **Feature Projection Layer**: Maps CNN features to embedding space
3. **Label Embedding Layer**: Embeds class labels into the same space
4. **LSTM Decoder**: Processes label sequences to model dependencies
5. **Classification Layer**: Final layer for multi-label prediction

### Loss Function

The model uses a custom multi-label loss function that:
- Applies different weights to positive and negative samples
- Uses Binary Cross-Entropy with Logits
- Handles class imbalance through weighted loss

### Training Process

1. **Image Encoding**: CNN extracts features from input images
2. **Label Sequence Processing**: RNN processes label sequences during training
3. **Joint Learning**: Model learns both visual features and label dependencies
4. **Multi-Label Prediction**: Outputs probabilities for all classes

## Evaluation Metrics

The model is evaluated using:
- **Precision**: Ratio of correctly predicted positive labels
- **Recall**: Ratio of actual positive labels that were predicted
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Exact match accuracy across all labels

## Results

The model achieves competitive performance on multi-label classification tasks by:
- Capturing label dependencies through RNN processing
- Learning joint image-label representations
- Handling variable numbers of labels per image

## Files Structure

```
├── model.py                 # Core CNN-RNN model implementation (161 lines)
├── data_loader.py           # Multi-label dataset and data loading (116 lines)
├── train_multilabel.py      # Training script with evaluation (295 lines)
├── predict_multilabel.py    # Prediction script with visualization (267 lines)
├── test_predictions.py      # Test script for model predictions (214 lines)
├── requirements.txt         # Python dependencies
├── README.md               # Documentation with diagrams
├── images/                 # Sample images directory
├── checkpoints/            # Model checkpoints (created during training)
├── test_annotations.json   # Sample annotations (auto-generated)
└── prediction_demo.png     # Example prediction visualization
```

### Core Components:

- **`model.py`**: Contains the `CNNRNNMultiLabel` class with CNN encoder and RNN decoder
- **`data_loader.py`**: `MultiLabelDataset` class and `get_multilabel_loader` function
- **`train_multilabel.py`**: Complete training pipeline with loss functions and evaluation
- **`predict_multilabel.py`**: Prediction pipeline with visualization capabilities
- **`test_predictions.py`**: Simple test script for model validation



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Original paper authors for the CNN-RNN framework
- PyTorch team for the deep learning framework
- COCO dataset for evaluation benchmarks


# CNN-RNN: A Unified Framework for Multi Label Image Classification

Real-world images often have multiple labels, representing various objects, scenes, actions, and attributes. While single-label image classification with Convolutional Neural Networks (CNNs) has seen success, traditional methods for multi-label classification lack explicit handling of label dependencies.

This project introduces a CNN-RNN framework to address this gap. By combining CNNs for feature extraction with RNNs for modeling label dependencies, our approach learns a joint image-label embedding. The framework allows end-to-end training, integrating image and label information seamlessly.

Experimental results on benchmark datasets demonstrate that our CNN-RNN framework achieves superior performance compared to existing models in multi-label image classification.


```markdown
This repository contains the source code for a CNN-RNN model for multi-label image classification. It's important to note that the model is not generating captions, but rather predicting labels for a given image.

## Requirements

To run the code, you need to install the following dependencies:

- Python 3.6 or higher
- PyTorch 1.6.0 or higher
- torchvision 0.7.0 or higher
- numpy 1.18.1 or higher
- scikit-image 0.16.2 or higher
- scipy 1.5.2 or higher
- nltk 3.4.5 or higher
```

## Data

The model is trained and evaluated on the [COCO dataset](http://cocodataset.org/#home), which consists of 82,081 training images and 40,137 validation images. Each image is annotated with multiple labels from a set of 80 categories.


## Model

The model is a unified framework for multi-label image classification that combines a CNN and an RNN. The CNN extracts features from the image, and the RNN learns a joint image-label embedding to characterize the semantic label dependency as well as the image-label relevance.

The model is implemented in the `model.py` file.

## Training

The model is trained using the `train.py` script. You can specify the hyperparameters using command-line arguments.

Here's an example command to train the model:

```bash
python train.py --image_path data/images/img.jpg --vocab_path data/vocab.pkl --batch_size 256 --learning_rate 0.001 --num_epochs 10 --num_workers 4
```


## Results

The results of the evaluation are printed to the console and saved to a log file.



## Citation

If you use the code in your research, please cite the following paper:

Chen, X., & Gupta, A. (2017). CNN-RNN: A unified framework for multi-label image classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 4961-4969).


# SqueezeNet PyTorch

## About the Project

This project implements the SqueezeNet image classification model
using PyTorch.

The SqueezeNet architecture was created manually and pretrained
weights were loaded from a `.pth` file.

The project takes an image as input and predicts the object
present in the image along with its confidence score.

## How the Project Works

Image
↓
Image Preprocessing
↓
SqueezeNet Model
↓
Pretrained Weights
↓
Prediction
↓
Object Name + Confidence

## SqueezeNet

SqueezeNet is a lightweight Convolutional Neural Network (CNN).

It uses a special block called the Fire Module.

The Fire Module contains:

- Squeeze 1x1 convolution
- Expand 1x1 convolution
- Expand 3x3 convolution

## Image Preprocessing

The input image is:

1. Converted to RGB
2. Resized to 224 × 224
3. Converted into a tensor
4. Normalized
5. Given to the model

The final input shape is:

[1, 3, 224, 224]

## Pretrained Weights

The project uses:

squeezenet1_0-b66bff10.pth

The pretrained model contains 1000 ImageNet classes.

The weights are loaded manually into our SqueezeNet model.

## Dataset

The `dataset.py` file handles images and labels.

Example:

```python
data = [
    {
        "image": "image.png",
        "label": 207
    }
]
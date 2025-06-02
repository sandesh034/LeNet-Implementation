# LeNet-5 Implementation in PyTorch

A complete implementation of LeNet-5 Convolutional Neural Network using PyTorch, trained and evaluated on both MNIST and EMNIST datasets.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Results](#results)
- [Model Architecture Details](#model-architecture-details)
- [Training Details](#training-details)
- [References](#references)

## 📖 Overview

This project implements the classic LeNet-5 architecture proposed by Yann LeCun et al. in 1998. The implementation includes:

- Complete LeNet-5 architecture with proper layer dimensions
- Training on MNIST (handwritten digits) dataset
- Training on EMNIST Letters (handwritten letters) dataset
- Visualization of training progress and sample predictions
- Model performance evaluation

## 🏗️ Architecture

LeNet-5 consists of 7 layers:

1. **C1**: Convolutional layer (1→6 feature maps, 5×5 kernel)
2. **S2**: Average pooling layer (2×2, stride=2)
3. **C3**: Convolutional layer (6→16 feature maps, 5×5 kernel)
4. **S4**: Average pooling layer (2×2, stride=2)
5. **C5**: Fully connected layer (400→120 neurons)
6. **F6**: Fully connected layer (120→84 neurons)
7. **Output**: Fully connected layer (84→num_classes)

**Activation Function**: Tanh (as in original paper)

## 🔧 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
torchinfo>=1.6.0
numpy>=1.21.0
jupyter>=1.0.0
```

## 🚀 Installation

1. Clone this repository:

```bash
git clone https://github.com/sandesh034/LeNet-Implementation.git
cd LeNet-Implementation
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook lenet.ipynb
```

## 📊 Dataset Information

### MNIST Dataset

- **Classes**: 10 (digits 0-9)
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image size**: 28×28 (resized to 32×32 for LeNet-5)
- **Format**: Grayscale

### EMNIST Letters Dataset

- **Classes**: 26 (letters A-Z)
- **Training samples**: 124,800
- **Test samples**: 20,800
- **Image size**: 28×28 (resized to 32×32 for LeNet-5)
- **Format**: Grayscale
- **Note**: Labels are converted from 1-26 to 0-25 for compatibility

### Key Features

- **GPU Support**: Automatically detects and uses CUDA if available
- **Data Preprocessing**: Includes proper normalization and resizing
- **Training Monitoring**: Real-time loss tracking and visualization
- **Model Summary**: Detailed architecture overview using torchinfo

## 📈 Results

### MNIST Performance

- **Training**: 20 epochs
- **Optimizer**: SGD with learning rate 0.001
- **Expected Accuracy**: ~88-92%

### EMNIST Letters Performance

- **Training**: 20 epochs
- **Optimizer**: SGD with learning rate 0.001
- **Expected Accuracy**: ~60-80%

## 🔍 Model Architecture Details

```python
class Lenet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            # C1: Convolutional Layer
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            # S2: Average Pooling
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: Convolutional Layer
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            # S4: Average Pooling
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Flatten and Fully Connected Layers
            nn.Flatten(),
            nn.Linear(400, 120, bias=True),
            nn.Tanh(),
            nn.Linear(120, 84, bias=True),
            nn.Tanh(),
            nn.Linear(84, num_classes, bias=True)
        )
```

**Input Dimensions**: 32×32×1 (grayscale images)
**Total Parameters**: ~61,706 (for 10 classes)

## 🎯 Training Details

### Hyperparameters

- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Epochs**: 20
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: CrossEntropyLoss

### Data Augmentation

- Resize to 32×32 (LeNet-5 input requirement)
- Normalization (MNIST: mean=0.1383, std=0.2935)
- Tensor conversion

## 📚 References

1. **Original Paper**: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. _Proceedings of the IEEE_, 86(11), 2278-2324. [PDF](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

2. **Datasets**:

   - [MNIST Database](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
   - [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

3. **PyTorch Documentation**:
   - [PyTorch Official Docs](https://pytorch.org/docs/)
   - [Torchvision Datasets](https://pytorch.org/vision/stable/datasets.html)

## 📁 Project Structure

```
LeNet/
├── lenet.ipynb          # Main implementation notebook
├── README.md            # This file
├── data/               # Dataset storage (auto-created)
│   ├── MNIST/
│   └── EMNIST/
└── requirements.txt    # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- Yann LeCun and colleagues for the original LeNet-5 architecture
- PyTorch team for the excellent deep learning framework
- NIST for providing the EMNIST dataset

---

⭐ If you found this implementation helpful, please give it a star!

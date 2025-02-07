# Convolutional Neural Network (CNN) from Scratch

This project implements a Convolutional Neural Network (CNN) from scratch using only fundamental libraries like `numpy`. The network is trained and tested on the MNIST dataset to recognize handwritten digits, specifically focusing on binary classification (digits 0 and 1).

## Features
- Fully custom implementation of CNN components, including convolutional, dense, reshape, and activation layers.
- Custom loss functions for binary cross-entropy and mean squared error.
- Preprocessing and handling of the MNIST dataset for digit classification.
- Training and evaluation scripts with adjustable parameters.

## Project Structure

```
project/
├── mnist_convo.py        # Main script for training and testing the CNN
├── activation.py         # Base Activation class
├── activations.py        # Implementations of Tanh, Sigmoid, and Softmax functions
├── convolutional.py      # Implementation of the Convolutional layer
├── dense.py              # Implementation of the Dense (fully connected) layer
├── layer.py              # Base Layer class
├── losses.py             # Loss functions (binary cross-entropy, MSE) and their derivatives
├── model.py              # Functions for training and predicting with the network
├── reshape.py            # Reshape layer implementation
```

## Installation

This project requires Python and the following libraries:
- `numpy`
- `scipy`
- `keras`

Install the dependencies using:
```bash
pip install numpy scipy keras
```

## How It Works

### 1. Preprocessing Data
- The MNIST dataset is loaded and filtered for digits 0 and 1.
- Images are reshaped to match the input dimensions of the network.
- Labels are one-hot encoded for binary classification.

### 2. Network Architecture
The CNN architecture used in this project:
1. **Convolutional Layer**:
   - Input: `(1, 28, 28)` (grayscale images)
   - Kernel size: `3x3`
   - Depth: `5`
   - Output: `(5, 26, 26)`
2. **Sigmoid Activation**
3. **Reshape Layer**:
   - Flattens the convolutional output into `(5 * 26 * 26, 1)`.
4. **Dense Layer**:
   - Fully connected layer with 100 neurons.
5. **Sigmoid Activation**
6. **Dense Layer**:
   - Fully connected layer with 2 neurons (for binary classification).
7. **Sigmoid Activation**

### 3. Training
- Loss Function: Binary Cross-Entropy
- Backpropagation is implemented manually to update weights and biases.
- The network is trained over 20 epochs with a learning rate of 0.1.

### 4. Testing
- The trained network is tested on the test set.
- Outputs the predicted and true labels for each test image.

## Example Usage

Run the `mnist_convo.py` script to train and test the CNN:
```bash
python mnist_convo.py
```

## Key Components

### `mnist_convo.py`
- Preprocesses the MNIST dataset for digit classification.
- Defines the CNN architecture.
- Trains the network using binary cross-entropy loss.
- Tests and prints the predictions for the test set.

### `activation.py`
- Base class for activation layers, enabling custom activation functions.

### `activations.py`
- Implements the `Tanh`, `Sigmoid`, and `Softmax` activation functions.

### `convolutional.py`
- Implements the Convolutional layer with forward and backward propagation.

### `dense.py`
- Implements the Dense (fully connected) layer with forward and backward propagation.

### `layer.py`
- Base Layer class with placeholder methods for forward and backward propagation.

### `losses.py`
- Contains loss functions:
  - **Binary Cross-Entropy**: Suitable for classification tasks.
  - **Mean Squared Error (MSE)**: Useful for regression tasks.

### `model.py`
- Contains the `predict` function for forward propagation.
- Implements the `train` function for network training with loss computation and backpropagation.

### `reshape.py`
- Reshapes data between layers during forward and backward passes.

## Results
After training, the CNN achieves binary classification on the MNIST dataset. Predicted labels are compared against true labels, and the performance can be observed in the printed output.

## Contributing
Feel free to fork the repository and submit pull requests to enhance the project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.


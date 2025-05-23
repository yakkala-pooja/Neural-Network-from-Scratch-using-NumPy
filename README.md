
# Neural Network from Scratch using NumPy

This project demonstrates a fully functional feedforward neural network built from scratch using only **NumPy**. It showcases the implementation of dense layers, activation functions, forward and backward propagation, and training using gradient descent.

## Overview

The neural network architecture and learning pipeline are entirely coded from the ground up, without using any machine learning libraries. The implementation supports a modular layer structure and is designed to help understand the inner workings of a neural network.

This project includes:

- Manual forward and backward propagation
- Custom loss computation using Mean Squared Error (MSE)
- Simple gradient descent optimization
- Testing and experiments done in a Jupyter notebook

---

## File Structure

```bash
.
├── nnet.py           # Core neural network module with all layer and training logic
├── NeuralNet.ipynb   # Jupyter notebook for testing and training (OR, XOR, and classification)
└── README.md         # Project description
```

---

## Key Components in `nnet.py`

### 1. `Layer` (Base Class)
- Interface class for layers (`init`, `forward`, `backward`)

### 2. `FCLayer` (Fully Connected Layer)
- Initializes weights and biases
- Implements forward propagation: `output = np.dot(input, weights) + bias`
- Backward propagation through weight updates

### 3. `ActivationLayer`
- Applies an activation function (`tanh`, `sigmoid`, etc.)
- Stores and uses the derivative for backpropagation

### 4. `Network`
- Core model class to:
  - Add layers
  - Train with `.fit()`
  - Predict with `.predict()`

---

## Experiments in Notebook

The Jupyter notebook (`NeuralNet.ipynb`) includes:

- **Manual test for OR gate**
  - Input: `[1, 0]`
  - Hidden and output layers use `tanh`
  - Forward pass output ≈ `0.831`
- **Weight update using gradient descent**
  - New weights and biases computed after one step
- **XOR gate training**
  - Shows convergence over epochs with loss curve
- **Stress classification task**
  - Gaussian data used to distinguish "stressed" vs "not stressed"

---

## Sample Use-Case: OR Gate

Given:
- 2 input neurons
- 4 hidden neurons with `tanh` activation
- 1 output neuron with `tanh` activation

Manual forward pass:
```python
input = [1, 0]
# Using given weights, biases, and tanh activations
# Resulting output: ≈ 0.831
```

---

## Requirements

- Python 3.x
- NumPy
- Jupyter Notebook (for running `ipynb`)

Install dependencies:
```bash
pip install numpy notebook
```

---

## Getting Started

1. Clone the repo or download the files.
2. Open the notebook:
```bash
jupyter notebook NeuralNet.ipynb
```
3. Explore experiments and modify parameters to test various configurations.

---

## Future Enhancements

- Add support for batch training and optimizers (e.g., Adam)
- Include cross-entropy loss and softmax for classification
- Extend to support multi-class problems

---

## License

This project is open-source and intended for educational and exploratory use.

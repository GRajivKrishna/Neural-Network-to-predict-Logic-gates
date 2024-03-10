# Neural Network Implementation in C

This project entails a basic implementation of a neural network in the C programming language. The neural network is designed to learn from a small dataset using backpropagation with sigmoid activation functions. Below is a breakdown of the key components and functionalities of the neural network:

## Overview

The neural network implemented in this project consists of an input layer, a hidden layer, and an output layer. It uses the sigmoid activation function and backpropagation algorithm to learn from the provided training data.

## Components

### Activation Functions

- **Sigmoid Function:** Used for both hidden and output layers to introduce non-linearity into the network.

### Weight Initialization

- **Random Initialization:** Weights and biases are initialized randomly to break symmetry and prevent convergence to local minima.

### Training

- **Backpropagation:** The network is trained using the backpropagation algorithm to adjust weights and biases based on the error between predicted and actual outputs.

### Input and Output

- **Training Data:** Input and output training sets are provided to the network for learning purposes.

## Files

- **neural_network.c:** Contains the main implementation of the neural network.
- **Makefile:** Builds the project.

## Usage

1. Compile the program using the provided Makefile: `make`.
2. Execute the compiled program: `./neural_network`.

## Parameters

- **Learning Rate:** Adjustable parameter for controlling the rate at which weights are updated during training.
- **Number of Epochs:** The number of iterations over the entire training dataset during training.

## Dependencies

- **Standard C Libraries:** `stdio.h`, `math.h`, `stdlib.h`
- **External Libraries:** None




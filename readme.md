# Learning Pytorch 

In May-June 2025, I embarked on a self-initiated project to learn the basics of neural networks and deep learning using Pytorch. I created several basic models and uploaded them here. 

This repository is dvided into several folders, each containing the relevant files for the model or category of models I created. The details of the file structure are described below. 

## Linear

This folder contains 2 models. 

### `linear_regression.py`
<b> How to use it: </b>
> ```python linear_regression.py [gradient] [intercept] [epochs] [learning_rate]```

<b> What it does: </b>
It manually creates a model with weight as the gradient parameter and bias as the intercept parameter. It then trains the model over the specified number of epochs with the specified learning rate. It outputs the gradient and intercept it learnt over these epochs, illustrated by a `matplotlib` graph. 

 ### `linear_regression2.py`
 <b> How to use it: </b>
> ```python linear_regression2.py [gradient] [intercept] [epochs] [learning_rate]```

<b> What it does: </b>
The same thing as `linear_regression.py` except using the built-in `nn.Linear` that comes with PyTorch. 

## Classification 

### `binary_classification.py`
 <b> How to use it: </b>
> ```python binary_classification.py [epochs] [learning_rate]```

<b> What it does: </b>
Uses `nn.Linear`, `nn.Sequential` and `nn.ReLU` to create a model capable of classifying objects into 2 classes. The data is 2 circles of different radii. The program lets the user see how varying the number of epochs and learning rate helps the model learn. 
### `multiclassification.py`
 <b> How to use it: </b>
> ```python multiclassification.py [epochs] [learning_rate] [num_classes]```

<b> What it does: </b>
The program lets the user see how varying the number of epochs and learning rate helps the model learn. This time, the user can also customise the number of classes that the data is split into. 

## CNN 
### `fashion.py`
 <b> How to use it: </b>
> ```python fashion.py```

<b> What it does: </b>
Trains a convolutional neural network on the FashionMNIST dataset available through `torchvision.datasets`. 

### `fashioneval.py`
 <b> How to use it: </b>
> ```python fashioneval.py```

<b> What it does: </b>
Evaluates the model created by `fashion.py` and outputs the results.
## LSTM 

## Frankenstein
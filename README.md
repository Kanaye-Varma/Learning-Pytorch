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

# Adaptive Wiener Loss

Data comparison lies at the heart of machine learning: for many applications, simplistic loss
functions - such as the L2 loss that rely on local element-wise differences between samples - have
taken preference. Such metrics are notorious for producing low-quality results. The proposed Adaptive Wiener Loss (AWLoss) addresses this issue by introducing a new convolutional approach to data comparison; one that uses a Wiener filter approach to naturally incorporate global information and promote spatial awareness within the compared samples. 

This repository contains an implementation of this loss in a natural [`Pytorch`](https://github.com/pytorch/pytorch) as here it is promoted as a loss function to drive deep learning problems.


## Installation
```
git clone https://github.com/dekape/AWLoss.git
cd AWLoss

# create conda environment (recommended)
conda create --name awloss
conda activate awloss

# install dependencies
pip install -r requirements.txt

# for running examples and performance notebooks
pip install -r requirements-optional.txt

# install package
pip install -e .
```

## Example usage
```
import torch
from awloss import AWLoss

awloss = AWLoss()
x = torch.rand([1, 3, 28, 28])
y = torch.rand([1, 3, 28, 28])

awloss(x, y)
>> tensor(1.4073, grad_fn=<DivBackward0>)

awloss(x, x)
>> tensor(0., grad_fn=<DivBackward0>)
```

## Method Overview
The main idea behind this comparison method, firstly introduced by [Warner and Guasch (2014)](https://www.s-cube.com/media/1204/segam2014-03712e1.pdf), is that two signals are considered identical when their corresponding matching filter is an dirac delta at zero lag (i.e. convolutional idendity). We start by considering two signals, $\mathbf{x}$ and $\mathbf{d}$, that are not identical. A convolutional Wiener filter $\mathbf{v}$ that provides the best least squares match between the two samples is computed by the well known equation:

$$
\mathbf{v} = (\mathbf{D}^{T} \mathbf{D})^{-1} \mathbf{D}^{T} \mathbf{x}
$$

where $\mathbf{D}$ is the Toeplitz matrix of signal $\mathbf{d}$ that achieves a convolution operation in matrix form.

We then act on this filter through an arbitrary a penalty function $\mathbf{T}$ that rewards energy at zero lag, and monotonically penalises energy at non-zero lags:

$$
L = \frac{1}{2}||\mathbf{T} * (\mathbf{v} - \mathbf{\delta)}||^{2}_{2}
$$

By minimising $L$, we implicitly drive signal $\mathbf{d}$ to become more similar to signal $\mathbf{x}$

## Input Arguments


## Filter Dimensions
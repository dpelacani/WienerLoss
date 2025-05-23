# Wiener Loss

This repository contains the implementation of the `WienerLoss` as presented in [Convolve and Conquer: Data Comparison with Wiener Filters](https://arxiv.org/abs/2311.06558) and at the workshop [Medical Imaging meets NeurIPS 2023](https://neurips.cc/virtual/2023/82490).
little change test

---

Data comparison lies at the heart of machine learning: for many applications, simplistic loss
functions - such as the L2 loss that rely on local element-wise differences between samples - have
taken preference. Such metrics are notorious for producing low-quality results. The proposed Wiener Loss addresses this issue by introducing a new convolutional approach to data comparison; one that uses a Wiener filter approach to naturally incorporate global information and promote spatial awareness within the compared samples. 


This repository contains an implementation of this loss in a natural [`Pytorch`](https://github.com/pytorch/pytorch) as here it is promoted as a loss function to drive deep learning problems. The source code is a single file that contains a single class named [`WienerLoss`](wiener_loss/wiener_loss.py). Its usage and customisation are described below.

A demonstration of this loss in a deep learning context is shown in the following figure for a medical data imputation problem:

<img src="figs/cerebellum_samples2.png" alt="drawing" width="700"/>

In the figure, (e) and (f) are obtained through the training of a UNet.


## Installation
```
git clone https://github.com/dpelacani/WienerLoss.git
cd WienerLoss

# create conda environment (recommended)
conda create --name wiener
conda activate wiener

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
from wiener_loss import WienerLoss

wiener_loss = WienerLoss()
x = torch.rand([1, 3, 28, 28])
y = torch.rand([1, 3, 28, 28])

wiener_loss(x, y)
>> tensor(1.4073, grad_fn=<DivBackward0>)

wiener_loss(x, x)
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

<img src="figs/scheme.png" alt="drawing" width="500"/>

## Input Arguments
On object initialistion:

    Args:
        method, optional
            "fft" for Fast Fourier Transform or "direct" for the
            Levinson-Durbin recurssion algorithm.
            In this version only "fft" is available. Defaults to "fft"
        filter_scale, optional
            the scale of the filters compared to the size of the data.
            Defaults to 2
        reduction, optional
            specifies the reduction to apply to the output, "mean" or "sum".
            Defaults to "mean"
        mode, optional
            "forward" or "reverse" computation of the filter. For details of
            the difference, refer to the original paper. Default "reverse".
        penalty_function, optional
            the penalty function to apply to the filter. It should be either:
            a python function that receives a mesh as input; "identity", 
            which creates a uniform mesh populated with the value 1; or 
            "gaussian", which creates a gaussian function with mean of 0 and
            standard deviation defined by the parameter "std". If None, the
            penalty function is the identity. Default None.
        store_filters, optional
            whether to store the filters in memory, useful for debugging.
            Option to store the filers before or after normalisation with
            "norm" and "unorm", respectively. Default False.
        lmbda, optional
            the stabilization value to compute the filter. It is used as
            a percentage value of the RMS of the cross-correlation and
            applied equally to the nominator and denominator for
            decorrelation. Default 1e-4.
        std, optional
            the standard deviation value of the zero-mean gaussian generated
            as a penalty function for the filter. Only applicable when
            'penalty_function' is passed as "gaussian". Default 1e-4.

On object calling

    Args:
        recon
            the reconstructed signal
        target
            the target signal
        lmbda, optional
            the stabilization value to compute the filter. It is used as
            a percentage value of the RMS of the cross-correlation and
            applied equally to the nominator and denominator for
            decorrelation. If passed, overwrites the class attribute
            of same name. Default None.
        gamma, optional
            noise to add to both target and reconstructed signals
            for training stabilization. Default 0.
        eta, optional
            noise to add to penalty function. Default 0.

## Filter Dimensions

The `WienerLoss` class automatically supports input data with up to 3 spatial dimensions. The dimensionality of the Wiener filter is inferred directly from the input shape:

| **Input Shape**                             | **Filter Dimensionality** |
|----------------------------------------------|---------------------------|
| `[batch, channels, length]`                  | 1D filters                |
| `[batch, channels, height, width]`           | 2D filters                |
| `[batch, channels, depth, height, width]`    | 3D filters                |

**Important:**  
For each case, the `WienerLoss` creates **one filter per channel per sample** in the batch. 

The batch dimension is always required.

## Example Notebooks:
*Needs updating*
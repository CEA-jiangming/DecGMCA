# DecGMCA

DecGMCA (Deconvolution Generalized Morphological Component Analysis) is a sparsity-based algorithm aiming at solving joint multichannel Deconvolution and Blind Source Separation (DBSS) problem.

*For more details, one can refer to the paper **Joint Multichannel Deconvolution and Blind Source Separation** (https://arxiv.org/abs/1703.02650)*

## Contents
1. [Introduction](#intro)
1. [Optimization](#opt)
1. [Python DecGMCA package](#pyDecGMCA)
1. [Installing](#install)
1. [Execution](#exec)
1. [Authors](#authors)
1. [Acknowledgments](#ack)

<a name="intro"></a>
## Introduction

Considering a multichannel DBSS problem:
> Y = H \* ( AS ) + N

Y: observation of size Nc by Np for an Nc-channel and Np-pixel imaging system.
H: a linear convolution kernel of size Nc by Np (*e.g.* downsampling matrix, psf), often ill-conditioned in practice.
A: unknown mixing matrix of size Nc by Ns, representing Ns sources are blindly linearly mixed (entities of row i represents the weighted contribution of sources at the given channel number i).
S: sources of size Ns by Np, representing Ns sources of Np pixels (sources are aligned to row vectors).
N: additional noise of size Nc by Np, supposed to be Gaussian.

This problem can be conveniently written in **Fourier space**, which has a lot of interests in Fourier imaging systems such as radio interferometry, MRI, etc. In Fourier space, noticing the convolution is transformed to product and A is unchanged as its entities are actually scalar factors applied to sources, the problem is written as
> \hat{Y} = \hat{H} ( A \hat{S} ) + \hat{N}

The sources are sparse in a given dictionary $\Phi$. Thus, the yielding optimization problem is written as follows:
> min\_{A,S} ||\hat{Y} - \hat{H} A \hat{S}||\_2^2 + \lambda ||S \Phi||\_p,
where p-norm is 0-norm or 1-norm.

<a name="opt"></a>
## Optimization

Challenges:
1. As the above optimization is non-convex, only critical point can be expected.  
1. Convolution kernel is ill-conditioned, leading to the unstability of the solution.

The main idea of solution is based on alternating minimization but we do not directly apply alternating proximity-based algorithms due to its computational demanding. Our DecGMCA employs an alternating projected least-squares procedure to approach the critical point plus a proximity-based procedure to finally refine the solution. Thus, DecGMCA is structured as:
- Intitialization
- Alternating projected least-squares
- Refinement step

### Initialization

A simple initialization (of matrix A) can be realized by randomization. One can also have more accurate initialization which depends on the form of the convolution kernel H:
- If H is a downsampling matrix, we apply several iterations of matrix completion scheme (*e.g.* SVT algorithm). Then the initialization is acheived by selecting Ns eigonvectors of left singular matrix after the application of singular value decomposition (svd) on the completed data.
- If H is a convolution kernel (not a downsampling matrix), we only keep Ns eigonvectors of left singular matrix as the initialization after svd on the completed data.

### Alternating projected least-squares

The procedure is based on the alternating update of one variable with respect to the other.
#### Update S with respect to A
This update can be divided into two steps: approximation of S via least-squares and sparsity thresholding. 

As for the approximation of S via least-squares, due to the ill-conditioned kernel H, a regularization parameter $\epsilon$ is involved to stablize the deconvolution. This parameter acts as a Tikhonov parameter: If $\epsilon$ is large, the system will be more regularized but the solution is less accurate. Conversely, if $\epsilon$ is small, the system will be less regularized but the solution is more accurate. The second step is sparsity thresholding to have a clean estimate of S. This step is realized by hard-thresholding with threshold $\lambda$. (Although soft-thresholding for l1-norm has more beautiful mathematical convergence proof, we argue that hard-thresholding has no biais effect and has empirical convergence according to our experiments). The consideration of $\epsilon$ and $\lambda$ is extensively studied in the paper.
#### Update A with respect to S
This update is just a simple least-squares. One should notice that columns of A should be l2 normalized.

### Refinement step
The alternating projected least-squares is efficient but does not necessarily ensure the optimal solution. This is owing to the fact that the projected least-squares has algorithmic biais compared to the exact projection realized by proximal algorithms. Thus, in order to refine the solution (often S), we resolve the problem of updating S with respect to A by using proximal algorithms such as Forward-Backward, Condat-Vu primal dual, etc.

<a name="pyDecGMCA"></a>
## Python DecGMCA package

### Prerequisites
#### Basic python environment
This package has been tested with python 2.7, some python libraries are required to ensure a correct working:
- numpy
- scipy
- matplotlib
- astropy
The above libraries are accessible in *macport*, *pip* or other package manager systems.
For instance, via *macport*:
```
port install some-package-name
```
or via *pip*:
```
pip install some-package-name
```
One may need root permission for the above operations.
#### Accelaration of codes? Interface python with C++ and paralization
For large-scale data, one may be not satisfied python (python can be up to 50 times slower than C/C++). In this DecGMCA package, we have an option to interface python with C++ and paralize the codes. The following packages are required:
- GCC (tested with GCC 4.9)
- CMake (tested with v3.9)
- Boost (tested with v1.58)
- Cfitsio
- OMP (tested with GCC 4.9)
These packages are easily installed via *apt-get* (Linux), *homebrew* (MacOS) or directly from the website.
For instance, via *apt-get*
```
apt-get install some-package-name
```
or via *homebrew*
```
brew install some-package-name
```
#### Problem with MacOS?
The default C compiler of MacOS is Clang. One should set GCC as the default C compiler and compile all dependencies (Boost, Cfitsio, etc.)

Among all packages, the Boost package is the most troublesome on MacOS. Here is a rapid solution:
- Download the [Boost source](http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz/download).
- Unzip and untar the downloaded file.
```
tar -xzvf boost_1_58_0.tar.gz
```
- Inside the boost directory run the *bootstrap.sh* script specifying a path to where the libraries are to be installed as follows:
```
./bootstrap.sh --prefix=/opt/local/ --with-toolset=gcc
```
- Run the *b2* script as follows:
```
./b2 install
```

<a name="install"></a>
## Installing

If all of the prerequisites are installed properly, one only needs to download the whole repository on his local machine.

The package includes:
- pyDecGMCA: main DecGMCA package
  - mathTools.py: some useful mathematical operations
  - pyUtils.py: useful routines such as two stages of update
  - algoDecG.py: python DecGMCA algorithm
  - pyProx.py: python proximal algorithms (used for the refinement step)
  - boost_Prox.py: partially accelerated proximal algorithms
  - boost_algoDecG.py: accelerated DecGMCA algorithm
  - boost_utils.py: accelerated routines such as two stages of update. One needs to compile DecGMCA_utils previously.
- pyWavelet: python wavelet tools
  - waveTools.py: some operations for wavelet coefficients
  - wav1d.py: 1D wavelet
  - wav2d.py: 2D wavelet
  - starlet_utils.py: accelerated starlet transform. One needs to compile pystarlet previously.
- simulationsTools: used to generate sources for tests
  - MakeExperiment.py
- simu_CS_deconv: used for running tests
  - param.py: global parameters for tests
  - test_CS.py: test script for compressed sensing test
  - test_deconv.py: test script for deconvolution test
- evaluation: used to evaluate results, such as criteria A and S.
  - evaluation.py

The following packages are needed to interface python with C++
- DecGMCA_utils
- pystarlet
Instructions for compilation (*e.g.* DecGMCA_utils):
- Inside the DecGMCA_utils directory create a *build* dossier
```
mkdir build
```
- Inside the build directory and run
```
cmake ..
make
```
<a name="exec"></a>
## Execution

<a name="authors"></a>
## Authors

* **Ming Jiang**

<a name="ack"></a>
## Acknowledgments
This work is supported by the CEA DRF impulsion project COSMIC and the European Community through the grants PHySIS (contract no. 60174), DEDALE (contract no. 665044) and LENA (contract no. 678282) within the H2020 Framework Programe.

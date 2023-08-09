# Pythonic Autograd
 Pythonic implementation for PyTorch autograd


<!-- generate table of content -->
## Table of Content
- [Pythonic Autograd](#pythonic-autograd)
  - [Table of Content](#table-of-content)
  - [Introduction](#introduction)
  - [Walkthrough](#walkthrough)
    - [Notebook 1 : Unsupervised Loss Implementation](#notebook-1--unsupervised-loss-implementation)
    - [Notebook 2 : Unsupervised Closefrom Loss](#notebook-2--unsupervised-closefrom-loss)
    - [Notebook 3 : Different Gradiant Decent Implementations](#notebook-3--different-gradiant-decent-implementations)
    - [Notebook 4 : Backpropagation Algorithm explained](#notebook-4--backpropagation-algorithm-explained)
    - [Notebook 5 : Pytorch Under The Hood](#notebook-5--pytorch-under-the-hood)
    - [Notebook 6 : Pytorch vs Numba](#notebook-6--pytorch-vs-numba)
    - [Notebook 7 : Back Propagation details P1](#notebook-7--back-propagation-details-p1)
    - [Notebook 8 : Back Propagation details P2](#notebook-8--back-propagation-details-p2)
    - [Notebook9 : Testing Implemented Autograd](#notebook9--testing-implemented-autograd)
    - [Extra - Topological Sort](#extra---topological-sort)
## Introduction 
This repository comprises numerous notebooks, each playing a significant role in enhancing comprehension of the PyTorch graph and facilitating the development of Pythonic autograd functionality.

## Walkthrough 
### [Notebook 1](1-Unsupervised-Loss-Implementation.ipynb) : Unsupervised Loss Implementation

This notebook addresses a target concerning an unsupervised training problem, which can be broken down as follows:

- **Problem**: find a solution to adjust an initially positioned point towards the central position amidst a distribution of randomly scattered points.
- **Goal**: Develop an understanding of the central objective underpinning gradient descent techniques, which revolves around the attainment of the optimal minimum point

### [Notebook 2](2-Unsupervised-Closefrom-Loss.ipynb) : Unsupervised Closefrom Loss
Refining the loss equation established in the previous notebook to a closed-form expression, as opposed to a numerical gradient loss, promises faster and more efficient computations.

### [Notebook 3](3-Different-Gradiant-Decent-Implementations.ipynb) : Different Gradiant Decent Implementations
- Implementation of the Full batch gradient decent
- Implementation of the Mini batch gradient decent
- Implementation of the Stochastic gradient decent

### [Notebook 4](4-Backpropagation-ALgorithm.ipynb) : Backpropagation Algorithm explained
A simplified mathematical breakdown of the backpropagation algorithm, which serves as the fundamental engine behind autograd functionality.
### [Notebook 5](5-Pytorch-Under-The-Hood.ipynb) : Pytorch Under The Hood
Explaination of some pytorch built-in tensor's related functions

### [Notebook 6](6-Pytorch-vs-Numba.ipynb) : Pytorch vs Numba
Creating the same function using both PyTorch and Numba, followed by a performance comparison between the two approaches.
>computionally Expensive to run, consume lot of disk memory, for the comparsion results.

### [Notebook 7](7-Back-Propagation-details-P1.ipynb) : Back Propagation details P1
Thoroughly elaborating the implementation process of the backpropagation algorithm using Torch tensors, applied to the moons dataset.





### [Notebook 8](8-Back-Propagation-details-P2.ipynb) : Back Propagation details P2
Continuing from the previous notebook, this iteration provides deeper insights into the backpropagation algorithm. It includes a comprehensive guide to implementing the backward function, leading to the creation and training of a distinct PyTorch model.


### [Notebook9](9-Testing-Implemented-Autograd.ipynb) : Testing Implemented Autograd
Assessing the implemented autograd feature using the Euclidean distance loss. This involves a comparison of outcomes against both the native PyTorch autograd capability and the previously developed loss and gradient calculation functions.

[Autograd Implementation](auto_grad.py)

### Extra - [Topological Sort](Extra-TopoSort.ipynb)

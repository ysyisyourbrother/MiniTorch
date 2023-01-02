# MiniTorch

<img src="https://minitorch.github.io/minitorch.svg" width="50%px">

[MiniTorch](https://github.com/minitorch/) is a diy teaching library for machine learning engineers who wish to learn about the internal concepts underlying deep learning systems. It is a pure Python re-implementation of the Torch API designed to be simple, easy-to-read, tested, and incremental. The final library can run Torch code.

## Guides

In this repository, each module builds a part of global MiniTorch framework.

- [Module-0](https://github.com/ysyisyourbrother/MiniTorch/tree/master/Module-0): Fundamental operators and help functions
- [Module-1](https://github.com/ysyisyourbrother/MiniTorch/tree/master/Module-1): Auto-Differentiation
- [Module-2](https://github.com/ysyisyourbrother/MiniTorch/tree/master/Module-2): Tensors
- [Module-3](https://github.com/ysyisyourbrother/MiniTorch/tree/master/Module-3): GPUs and Parallel Programming
- [Module-4](https://github.com/ysyisyourbrother/MiniTorch/tree/master/Module-4): Deep Neural Network

## Setup

MiniTorch requires Python 3.7 or higher. Please check your version of Python before start.

You can clone this repository to local by running following command:

```
$ git clone https://github.com/ysyisyourbrother/MiniTorch.git
```

Install the requirement packages:

```
$ pip3 install -r requirements.txt
$ pip3 install -r requirements.extra.txt
```

Execute following command everytime after you move to a new module:

```
$ pip3 install -Ue .
```
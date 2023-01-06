# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py

# Tensors

We now have a fully developed autodifferentiation system built around scalars. This system is correct, but you saw during training that it is inefficient. Every scalar number requires building an object, and each operation requires storing a graph of all the values that we have previously created. Training requires repeating the above operations, and running models, such as a linear model, requires a for loop over each of the terms in the network.

This module introduces and implements a tensor object that will solve these problems. Tensors group together many repeated operations to save Python overhead and to pass off grouped operations to faster implementations.

For this module we have implemented the skeleton tensor.py file for you. This is similar in spirit to scalar.py from the last assignment. Before starting, it is worth reading through this file to have a sense of what a Tensor does. Each of the following tasks asks you to implement the methods this file relies on:

- `tensor_data.py` : Indexing, strides, and storage
- `tensor_ops.py` : Higher-order tensor operations
- `tensor_functions.py` : Autodifferentiation-ready functions

## Tasks 2.1: Tensor Data - Indexing
The MiniTorch library implements the core tensor backend as minitorch.TensorData. This class handles indexing, storage, transposition, and low-level details such as strides. You will first implement these core functions before turning to the user-facing class minitorch.Tensor.

> TODO:
>
> Complete the following functions in minitorch/tensor_data.py, and pass tests marked as task2_1.

## Tasks 2.2: Tensor Broadcasting
> TODO:
>
> Complete following functions in minitorch/tensor_data.py and pass tests marked as task2_2.

## Tasks 2.3: Tensor Operations
Tensor operations apply high-level, higher-order operations to all elements in a tensor simultaneously. In particular, you can map, zip, and reduce tensor data objects together. On top of this foundation, we can build up a Function class for Tensor, similar to what we did for the ScalarFunction. In this task, you will first implement generic tensor operations and then use them to implement forward for specific operations.

> TODO:
>
> Add functions in minitorch/tensor_ops.py and minitorch/tensor_functions.py for each of the following, and pass tests marked as task2_3.

## Tasks 2.4: Gradients and Autograd
Similar to minitorch.Scalar, minitorch.Tensor is a Variable that supports autodifferentiation. In this task, you will implement backward functions for tensor operations.

> TODO:
>
> Complete following functions in minitorch/tensor_functions.py, and pass tests marked as task2_4.

## Task 2.5: Training
If your code works you should now be able to move on to the tensor training script in project/run_tensor.py. This code runs the same basic training setup as in module1, but now utilize your tensor code.

> TODO:
>
> Implement the missing forward functions in project/run_tensor.py. They should do exactly the same thing as the corresponding functions in project/run_scalar.py, but now use the tensor code base.

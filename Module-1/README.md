# MiniTorch Module 1

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module1/module1/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py tests/test_module.py tests/test_operators.py project/run_manual.py


# Autodiff
This module shows how to build the first version of MiniTorch using only simple values and functions. This covers key aspects of auto-differentiation: the key technique in the system. Then you will use your code to train a preliminary model.

## Task 1.1: Numerical Derivatives
Implement scalar numerical derivative calculation. This function will not be used in the main library but will be critical for testing the whole module.

> TODO:
>
> Complete the following function in minitorch/autodiff.py and pass tests marked as task1_1.

## Task 1.2: Scalars

Implement the overridden mathematical functions required for the minitorch.Scalar class. Each of these requires wiring the internal Python operator to the correct minitorch.Function.forward call.

Read the example ScalarFunctions that we have implemented for guidelines. You may find it useful to reuse the operators from Module 0.

We have built a debugging tool for you to observe the workings of your expressions to see how the graph is built. You can run it in the Autodiff Sandbox. You can alter the expression at the top of the file and then run the code to create a graph in Streamlit:

```
streamlit run app.py -- 1
```

> TODO:
>
> 1. Complete the following functions in minitorch/scalar_functions.py.
> 2. Complete the following function in minitorch/scalar.py, and pass tests marked as task1_2. See Python numerical overrides for the interface of these methods. All of these functions should return minitorch.Scalar arguments.


## Task 1.3: Chain Rule
Implement the chain_rule function in Scalar for functions of arbitrary arguments. This function should be able to backward process a function by passing it in a context and 
d
and then collecting the local derivatives. It should then pair these with the right variables and return them. This function is also where we filter out constants that were used on the forward pass, but do not need derivatives.

> TODO:
>
> Complete the following function in minitorch/scalar.py, and pass tests marked as task1_3.


## Task 1.4: Backpropagation
Implement backpropagation. Each of these requires wiring the internal Python operator to the correct minitorch.Function.backward call.

Read the example ScalarFunctions that we have implemented for guidelines. Feel free to also consult differentiation rules if you forget how these identities work.

> TODO:
>
> Complete the following functions in minitorch/autodiff.py and minitorch/scalar.py, and pass tests marked as task1_4.


## Task 1.5 Training
If your code works, you should now be able to run the training script. Study the code in project/run_scalar.py carefully to understand what the neural network is doing.

You will also need Module code to implement the parameters Network and for Linear. You can modify the dataset and the module with the parameters at the bottom of the file.

If your code is successful, you should be able to run the full visualization:

```
streamlit run app.py -- 1
```
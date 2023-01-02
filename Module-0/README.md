# MiniTorch Module 0

<img src="https://minitorch.github.io/minitorch.svg" width="50%px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module0.html


# Fundamentals
This introductory module is focused on introducing several core software engineering methods for testing and debugging, and also includes some basic mathematical foundations.

Before starting this assignment, make sure to set up your workspace following the setup guide, to understand how the code should be organized.

## Task 0.1: Operators
This task is designed to help you get comfortable with style checking and testing. We ask you to implement a series of basic mathematical functions. These functions are simple, but they form the basis of MiniTorch. Make sure that you understand each of them as some terminologies might be new.

> TODO:
>
> Complete the following functions in minitorch/operators.py and pass tests marked as task0_1.

## Task 0.2 Testing and Debugging
We ask you to implement property tests for your operators from Task 0.1. These tests should ensure that your functions not only work but also obey high-level mathematical properties for any input. Note that you need to change arguments for those test functions.

> TODO:
>
> Complete the test functions in tests/test_operators.py marked as task0_2.

## Task 0.3: Functional Python
To practice the use of higher-order functions in Python, implement three basic functional concepts. Use them in combination with operators described in Task 0.1 to build up more complex mathematical operations that work on lists instead of single values.

> TODO:
>
> Complete the following functions in minitorch/operators.py and pass tests marked as tasks0_3.

## Task 0.4: Modules
This task is to implement the core structure of the :class:minitorch.Module class. We ask you to implement a tree data structure that stores named :class:minitorch.Parameter on each node. Such a data structure makes it easy for users to create trees that can be walked to find all of the parameters of interest.

> TODO:
>
> Complete the functions in minitorch/module.py and pass tests marked as tasks0_4.
<h1>NoTorch</h1>

A 'from scratch' implementation of a deep learning library (no pytorch/tensorflow) built only on NumPy.

This is a learning project heavily inspired by and based on Andrej Karpathy's micrograd:
https://github.com/karpathy/micrograd


FEATURES:

- A matrix valued autograd engine, allowing for differentiable matrix operations:
    - element-wise: +, -, *, /, ^, log, exp, relu, sigmoid
    - matrix multiplication (used for efficient forward passes)
    - summation and concatenation across a single dimension

- A neural network library allowing users to build fully customizable neural networks, from basic multi-layer perceptrons to modern Transformers

- Extremely fast performance, speeds similar to PyTorch. Orders of magnitude faster than micrograd (see speed_test.py)

IN PROGRESS:

- GPU support via CuPy


<h1>Implementation Details</h1>

Coming soon



<br>
<h1>How to Use</h1>
Install poetry: https://python-poetry.org/docs/#installation

To run speed_test.py: (OSX/Linux)
```
$ git clone https://github.com/Lucasc-99/NoTorch
$ cd NoTorch
$ poetry install 
$ poetry run python speed_test.py
```

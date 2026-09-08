# godl

Deep learning building blocks implemented from scratch in Go — no ML libraries, just slices and math.

Covers the core mechanics of a feedforward neural network:

- Parameter initialization (shallow and deep/multi-layer)
- Forward propagation with linear + activation caching
- Backward propagation (gradients w.r.t. weights, biases, and activations)
- Activation functions: ReLU, Sigmoid
- Gradient descent parameter updates

The goal was to work through the actual math of training a neural net (forward pass, backprop, gradient updates) without numpy or a framework doing it for me — one layer at a time, in a language that doesn't hide the loops.

## Structure

```
activation/   ReLU, Sigmoid and their derivatives
ml/           forward.go, backward.go — the core prop algorithms
model/        shared cache/parameter types
```

## Run

```
go run .
```

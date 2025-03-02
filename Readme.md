# Linformer: Self-Attention with Linear Complexity

## Introduction

This repository is an implementation of the Linformer paper: *"Linformer: Self-Attention with Linear Complexity"* ([Paper Link](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2006.04768)).

Linformer introduces an efficient attention mechanism that reduces the quadratic complexity of traditional self-attention (O(N²d)) to a linear complexity O(Nd) by approximating the attention matrix as a low-rank matrix. Instead of computing the full NxN attention matrix, Linformer projects the key and value matrices into a lower-dimensional space using learned projection matrices, reducing memory and computational costs.

## Traditional Self-Attention Mechanism

The traditional attention mechanism computes attention scores using the following equation:

```math
A = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} \right) V
```

where:
- **Q** is the query matrix \( (N \times d) \)
- **K** is the key matrix \( (N \times d) \)
- **V** is the value matrix \( (N \times d) \)
- **N** is the sequence length
- **d** is the hidden dimension

The resulting attention matrix \( A \) has a computational complexity of **O(N²d)**, which becomes infeasible for large sequences.

## Linformer Attention Mechanism

Linformer reduces this complexity by introducing learned projection matrices \(E\) and \(F\) that project keys and values into a lower-dimensional space of size \( k \), where \( k << N \). The attention mechanism is then modified as:

```math
A = \text{softmax}\left( \frac{Q (EK)^T}{\sqrt{d}} \right) (FV)
```

where:
- **E** is the projection matrix for keys \( (k \times N) \)
- **F** is the projection matrix for values \( (k \times N) \)

Since \( k \) is a fixed, much smaller dimension, the computational complexity is reduced from **O(N²d) to O(Nd)**, making Linformer significantly more efficient while maintaining performance comparable to traditional transformers.

## Validation Loss Plot

Below is the validation loss plot obtained during training:

![Validation Loss](./plots/training_analysis.png)

![Learning Curve](./plots/learning_curve.png)

## Conclusion

Linformer significantly reduces the memory and computational cost of self-attention while maintaining competitive performance, making it suitable for handling long sequences efficiently. This implementation provides a hands-on approach to understanding and leveraging the power of Linformer.

For further details, refer to the original [paper](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2006.04768).


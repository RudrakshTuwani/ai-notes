# Plan

## Mathematical Foundations

- [ ] Linear Algebra: Rank, Eigenvalues, SVD
- [ ] Probability: Maximum Likelihood, Bayesian Inference

## Transformer Architecture

- [x] [Multihead Self-Attention](cheatsheets/attention.md) <sup>[1](#ref-1)</sup>
- [x] [LayerNorm / RMSNorm](cheatsheets/normalization.md) <sup>[2](#ref-2)</sup>
- [ ] [Transformer Layer](cheatsheets/transformer-layer.md) <sup>[3](#ref-3)</sup>
- [ ] Attention Variants: GQA, MLA <sup>[4](#ref-4)</sup>
- [ ] Positional Info: Rotary Embeddings

## Neural Net Optimization

- [ ] Optimizers: SGD, Adam, AdamW
- [ ] Automatic Differentiation: Forward and Reverse Mode

## Performance Engineering

- [ ] Transformer Compute and Memory Costs <sup>[5](#ref-5)</sup>
- [ ] Roofline Analysis <sup>[6](#ref-6)</sup>
- [ ] Sharded Matrix Multiplication <sup>[7](#ref-7)</sup>
- [ ] FlashAttention
- [ ] TPU Architecture <sup>[8](#ref-8)</sup>
- [ ] Distributed Training: Data, Tensor, and Pipeline Parallelism
- [ ] KV Caching
- [ ] Quantization: INT8, FP8
- [ ] Speculative Decoding

## Pre-training

- [ ] Scaling Laws <sup>[9](#ref-9), [10](#ref-10)</sup>

## Post-training

- [ ] Low-Rank Adaptation: LoRA
- [ ] Reinforcement Learning from Human Feedback: RLHF
- [ ] Constitutional AI

---

## References

1. <span id="ref-1"></span>[AI Notes — Multihead Self-Attention](cheatsheets/attention.md)
2. <span id="ref-2"></span>[AI Notes — LayerNorm and RMSNorm](cheatsheets/normalization.md)
3. <span id="ref-3"></span>[AI Notes — Transformer Layer](cheatsheets/transformer-layer.md)
4. <span id="ref-4"></span>[Visual Guide to Attention Variants](https://magazine.sebastianraschka.com/p/visual-attention-variants)
5. <span id="ref-5"></span>[Scaling Book, Chapter 4 — All the Transformer Math You Need to Know](https://jax-ml.github.io/scaling-book/transformers)
6. <span id="ref-6"></span>[Scaling Book, Chapter 1 — A Brief Intro to Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline)
7. <span id="ref-7"></span>[Scaling Book, Chapter 3 — Sharded Matrices and How to Multiply Them](https://jax-ml.github.io/scaling-book/sharding)
8. <span id="ref-8"></span>[Scaling Book, Chapter 2 — How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus)
9. <span id="ref-9"></span>[Stanford CS336 Spring 2026 — Lecture 9: Scaling Laws](https://www.youtube.com/watch?v=Q15rhEWZPQ4)
10. <span id="ref-10"></span>[Stanford CS336 Spring 2026 — Lecture 11: Scaling Laws](https://www.youtube.com/watch?v=vTfEyOyzV9E)

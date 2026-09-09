# Attention

**Mix values → learn weights → scale scores → repeat across heads.**

!!! note "Notation · tokens are columns"

    $\mathbf{X}=[\mathbf{x}_1,\ldots,\mathbf{x}_N]\in\mathbb{R}^{D\times N}$: $N$ tokens of width $D$. Output $n$ reads from source $m$. Softmax normalizes over **sources $m$**, separately for each query $n$.

## 1. Self-attention

**Create values.** A learned projection determines the content each token contributes:

$$
\mathbf{v}_m=\boldsymbol{\beta}_v+\boldsymbol{\Omega}_v\mathbf{x}_m
\in\mathbb{R}^{D}.
$$

**Mix values.** An attention function assigns a nonnegative weight to each source:

$$
\begin{aligned}
\mathbf{y}_n
  &=\sum_{m=1}^{N}a[\mathbf{x}_m,\mathbf{x}_n]\,\mathbf{v}_m,\\
\sum_{m=1}^{N}a[\mathbf{x}_m,\mathbf{x}_n]
  &=1.
\end{aligned}
$$

Collect the values into $\mathbf{V}$ and set $A_{mn}=a[\mathbf{x}_m,\mathbf{x}_n]$. The same mixture becomes:

$$
\begin{aligned}
\mathbf{V}&=\boldsymbol{\beta}_v\mathbf{1}^{T}+\boldsymbol{\Omega}_v\mathbf{X},\\
\operatorname{SA}[\mathbf{X}]&=\mathbf{V}\mathbf{A}.
\end{aligned}
$$

Here $\mathbf{1}\in\mathbb{R}^{N}$ broadcasts the bias. Value width $D$ makes the output fit a residual connection; attention itself permits other value widths.

## 2. Dot-product self-attention

**Learn the weights.** Add queries and keys as projections of the same input:

$$
\begin{aligned}
\mathbf{Q}[\mathbf{X}] &= \boldsymbol{\beta}_q\mathbf{1}^{T}+\boldsymbol{\Omega}_q\mathbf{X} &&\in\mathbb{R}^{D_q\times N},\\
\mathbf{K}[\mathbf{X}] &= \boldsymbol{\beta}_k\mathbf{1}^{T}+\boldsymbol{\Omega}_k\mathbf{X} &&\in\mathbb{R}^{D_q\times N}.
\end{aligned}
$$

Keys and queries have equal width $D_q$. Their dot product gives a score; softmax turns scores into weights:

$$
\begin{aligned}
a[\mathbf{x}_m,\mathbf{x}_n]
  &=\operatorname{softmax}_{m}(\mathbf{k}_m^{T}\mathbf{q}_n)\\
  &=\frac{\exp(\mathbf{k}_m^{T}\mathbf{q}_n)}
  {\sum_{m'=1}^{N}\exp(\mathbf{k}_{m'}^{T}\mathbf{q}_n)}.
\end{aligned}
$$

**Substitute into the same mixture** $\mathbf{V}\mathbf{A}$:

$$
\operatorname{SA}[\mathbf{X}]
=\mathbf{V}\cdot\operatorname{softmax}_{m}(\mathbf{K}^{T}\mathbf{Q}).
$$

!!! tip "Shape check"

    Scores and weights are $N\times N$. Multiplying by values gives $(D\times N)(N\times N)\rightarrow D\times N$.

## 3. Scaled dot-product self-attention

**Change only the score:** divide each query–key dot product by $\sqrt{D_q}$.

$$
\operatorname{SA}[\mathbf{X}]
=\mathbf{V}\cdot\operatorname{softmax}_{m}\!\left(
\frac{\mathbf{K}^{T}\mathbf{Q}}{\sqrt{D_q}}
\right).
$$

With independent, unit-variance components, dot-product variance grows with $D_q$; this scaling keeps it stable and helps prevent softmax saturation.

## 4. Multi-head self-attention

**Repeat scaled attention $H$ times.** Typically $H$ divides $D$, giving each head width $d_h=D/H$. Each head projects the full input independently:

$$
\begin{aligned}
\mathbf{V}_h &= \boldsymbol{\beta}_{vh}\mathbf{1}^{T}+\boldsymbol{\Omega}_{vh}\mathbf{X},\\
\mathbf{Q}_h &= \boldsymbol{\beta}_{qh}\mathbf{1}^{T}+\boldsymbol{\Omega}_{qh}\mathbf{X},\\
\mathbf{K}_h &= \boldsymbol{\beta}_{kh}\mathbf{1}^{T}+\boldsymbol{\Omega}_{kh}\mathbf{X}.
\end{aligned}
$$

All three matrices are $d_h\times N$. Apply the previous step per head:

$$
\begin{aligned}
\mathbf{Y}_h &= \mathbf{V}_h\cdot\operatorname{softmax}_{m}\!\left(
\frac{\mathbf{K}_h^{T}\mathbf{Q}_h}{\sqrt{d_h}}
\right) \in \mathbb{R}^{d_h\times N}.
\end{aligned}
$$

**Concatenate, then mix heads.** Stack the outputs along the feature dimension and apply a final projection:

$$
\operatorname{MHSA}[\mathbf{X}]
=\boldsymbol{\beta}_o\mathbf{1}^{T}
+\boldsymbol{\Omega}_o
\begin{bmatrix}
\mathbf{Y}_1\\ \vdots\\ \mathbf{Y}_H
\end{bmatrix}
\in\mathbb{R}^{D\times N}.
$$

Different projections let the $H$ heads learn different relationships; the output projection recombines their $D/H$-wide representations.

_Notation adapted from Simon J. D. Prince's [Understanding Deep Learning](https://udlbook.github.io/udlbook/)._

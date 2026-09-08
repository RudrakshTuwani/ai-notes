# Attention

Let a sequence be $\mathbf{X}=[\mathbf{x}_1,\ldots,\mathbf{x}_N]\in\mathbb{R}^{D\times N}$, with one token embedding per **column**. Softmax is taken over the source index $m$, so every output position $n$ receives weights that sum to one.

## 1. Self-attention

Create one value vector for every input, then mix those values using an attention function over pairs of positions:

$$
\mathbf{v}_m=\mathbf{v}[\mathbf{x}_m]\in\mathbb{R}^{D},
\qquad
\mathbf{y}_n=\sum_{m=1}^{N}a[\mathbf{x}_m,\mathbf{x}_n]\,\mathbf{v}_m,
\qquad
\sum_{m=1}^{N}a[\mathbf{x}_m,\mathbf{x}_n]=1.
$$

$\mathbf{v}_m$ carries the content; $a[\mathbf{x}_m,\mathbf{x}_n]$ says how much position $n$ reads from position $m$. Using $\mathbf{v}_m\in\mathbb{R}^{D}$ keeps each output $\mathbf{y}_n$ the same shape as its input embedding, as required for a residual connection.

## 2. Dot-product self-attention

Learn values, queries, and keys as affine projections of the same input:

$$
\begin{aligned}
\mathbf{V}[\mathbf{X}] &= \boldsymbol{\beta}_v\mathbf{1}^{T}+\boldsymbol{\Omega}_v\mathbf{X} &&\in\mathbb{R}^{D\times N},\\
\mathbf{Q}[\mathbf{X}] &= \boldsymbol{\beta}_q\mathbf{1}^{T}+\boldsymbol{\Omega}_q\mathbf{X} &&\in\mathbb{R}^{D_q\times N},\\
\mathbf{K}[\mathbf{X}] &= \boldsymbol{\beta}_k\mathbf{1}^{T}+\boldsymbol{\Omega}_k\mathbf{X} &&\in\mathbb{R}^{D_q\times N}.
\end{aligned}
$$

Queries and keys must have the same dimension $D_q$ so their dot product is defined. It becomes the score whose column-wise softmax gives the attention weight:

$$
a[\mathbf{x}_m,\mathbf{x}_n]
=\operatorname{softmax}_{m}(\mathbf{k}_m^{T}\mathbf{q}_n)
=\frac{\exp(\mathbf{k}_m^{T}\mathbf{q}_n)}
{\sum_{m'=1}^{N}\exp(\mathbf{k}_{m'}^{T}\mathbf{q}_n)}.
$$

Therefore, in matrix form:

$$
\operatorname{SA}[\mathbf{X}]
=\mathbf{V}\cdot\operatorname{softmax}_{m}(\mathbf{K}^{T}\mathbf{Q}).
$$

## 3. Scaled dot-product self-attention

$$
\operatorname{SA}[\mathbf{X}]
=\mathbf{V}\cdot\operatorname{softmax}_{m}\!\left(
\frac{\mathbf{K}^{T}\mathbf{Q}}{\sqrt{D_q}}
\right).
$$

The $\sqrt{D_q}$ divisor keeps dot-product magnitudes roughly stable as the query/key width grows, helping prevent softmax from saturating.

## 4. Multi-head self-attention

Use $H$ independently learned attention heads. Typically $H$ divides $D$, giving each head width $d_h=D/H$:

$$
\begin{aligned}
\mathbf{V}_h &= \boldsymbol{\beta}_{vh}\mathbf{1}^{T}+\boldsymbol{\Omega}_{vh}\mathbf{X},\\
\mathbf{Q}_h &= \boldsymbol{\beta}_{qh}\mathbf{1}^{T}+\boldsymbol{\Omega}_{qh}\mathbf{X},\\
\mathbf{K}_h &= \boldsymbol{\beta}_{kh}\mathbf{1}^{T}+\boldsymbol{\Omega}_{kh}\mathbf{X},\\
\mathbf{Y}_h &= \mathbf{V}_h\cdot\operatorname{softmax}_{m}\!\left(
\frac{\mathbf{K}_h^{T}\mathbf{Q}_h}{\sqrt{d_h}}
\right) \in \mathbb{R}^{d_h\times N}.
\end{aligned}
$$

Concatenate all head outputs along the feature dimension, then mix them with one final learned linear transform:

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

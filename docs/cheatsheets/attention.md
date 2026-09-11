# Attention

**Mix values → learn weights → scale scores → repeat across heads.**

!!! note "Notation · tokens are columns"

    - $B$: batch size; $b=1,\ldots,B$ indexes sequences.
    - $L$: sequence length.
    - $D$: input embedding dimension.
    - $l=1,\ldots,L$: output (query) position.
    - $m=1,\ldots,L$: source (key/value) position.
    - $\mathbf{x}_l\in\mathbb{R}^{D}$: token embedding at position $l$.
    - $\mathbf{X}=[\mathbf{x}_1,\ldots,\mathbf{x}_L]\in\mathbb{R}^{D\times L}$: one sequence.
    - Batched input: shape $B\times D\times L$; equations below apply independently to each sequence, omitting $b$.

## 1. Self-attention

**Create values.** A learned projection determines the content each token contributes:

$$
\mathbf{v}_l=\boldsymbol{\beta}_v+\boldsymbol{\Omega}_v\mathbf{x}_l
\in\mathbb{R}^{D}.
$$

**Mix values.** An attention function assigns a nonnegative weight to each source:

$$
\begin{aligned}
\mathbf{y}_l
  &=\sum_{m=1}^{L}a[\mathbf{x}_m,\mathbf{x}_l]\,\mathbf{v}_m,\\
\sum_{m=1}^{L}a[\mathbf{x}_m,\mathbf{x}_l]
  &=1.
\end{aligned}
$$

Collect the values into $\mathbf{V}$ and set $A_{ml}=a[\mathbf{x}_m,\mathbf{x}_l]$. The same mixture becomes:

$$
\begin{aligned}
\mathbf{V}&=\boldsymbol{\beta}_v\mathbf{1}^{T}+\boldsymbol{\Omega}_v\mathbf{X},\\
\operatorname{SA}[\mathbf{X}]&=\mathbf{V}\mathbf{A}.
\end{aligned}
$$

Here $\mathbf{1}\in\mathbb{R}^{L}$ broadcasts the bias. Value width $D$ makes the output fit a residual connection; attention itself permits other value widths.

## 2. Dot-product self-attention

**Learn the weights.** Add queries and keys as projections of the same input:

$$
\begin{aligned}
\mathbf{Q}[\mathbf{X}] &= \boldsymbol{\beta}_q\mathbf{1}^{T}+\boldsymbol{\Omega}_q\mathbf{X} &&\in\mathbb{R}^{D_q\times L},\\
\mathbf{K}[\mathbf{X}] &= \boldsymbol{\beta}_k\mathbf{1}^{T}+\boldsymbol{\Omega}_k\mathbf{X} &&\in\mathbb{R}^{D_q\times L}.
\end{aligned}
$$

Keys and queries have equal width $D_q$. Their dot product gives a score; softmax normalizes over sources $m$ for each query $l$:

$$
\begin{aligned}
a[\mathbf{x}_m,\mathbf{x}_l]
  &=\operatorname{softmax}_{m}(\mathbf{k}_m^{T}\mathbf{q}_l)\\
  &=\frac{\exp(\mathbf{k}_m^{T}\mathbf{q}_l)}
  {\sum_{m'=1}^{L}\exp(\mathbf{k}_{m'}^{T}\mathbf{q}_l)}.
\end{aligned}
$$

**Substitute into the same mixture** $\mathbf{V}\mathbf{A}$:

$$
\operatorname{SA}[\mathbf{X}]
=\mathbf{V}\cdot\operatorname{softmax}_{m}(\mathbf{K}^{T}\mathbf{Q}).
$$

!!! tip "Shape check"

    Scores and weights are $L\times L$. Multiplying by values gives $(D\times L)(L\times L)\rightarrow D\times L$.

## 3. Scaled dot-product self-attention

**Change only the score:** divide each query–key dot product by $\sqrt{D_q}$.

$$
\operatorname{SA}[\mathbf{X}]
=\mathbf{V}\cdot\operatorname{softmax}_{m}\!\left(
\frac{\mathbf{K}^{T}\mathbf{Q}}{\sqrt{D_q}}
\right).
$$

With independent, zero-mean, unit-variance components, dot-product variance grows with $D_q$; this scaling keeps it stable and helps prevent softmax saturation.

??? note "Proof · why divide by √D_q?"

    For a fixed source $m$ and query $l$, write $s=\mathbf{k}_m^T\mathbf{q}_l=\sum_{i=1}^{D_q}k_iq_i$. Assume all $k_i,q_i$ are mutually independent, with mean $0$ and variance $1$.

    Each product has mean $0$ and variance $1$:

    $$
    \begin{aligned}
    \mathbb{E}[k_iq_i]&=\mathbb{E}[k_i]\mathbb{E}[q_i]=0,\\
    \operatorname{Var}(k_iq_i)
    &=\mathbb{E}[k_i^2]\mathbb{E}[q_i^2]-0^2=1.
    \end{aligned}
    $$

    Independence removes cross-covariances, so:

    $$
    \begin{aligned}
    \operatorname{Var}(s)
    &=\sum_{i=1}^{D_q}\operatorname{Var}(k_iq_i)=D_q,\\
    \operatorname{Var}\!\left(\frac{s}{\sqrt{D_q}}\right)
    &=\frac{\operatorname{Var}(s)}{D_q}=1.
    \end{aligned}
    $$

    **Connection to softmax.** Softmax exponentiates scores, amplifying their differences:

    - Scores $[0,1]$ → probabilities approximately $[0.27,0.73]$.
    - Scores $[0,10]$ → probabilities approximately $[0.00005,0.99995]$.

    The second case is **saturation**: almost all attention goes to one token. Small score changes barely change the probabilities, so gradients through softmax become small.

    Wider query/key vectors tend to produce more spread-out scores under the assumptions above. Dividing by $\sqrt{D_q}$ keeps that spread roughly constant: attention should not become overly confident **just because the vectors are wider**. Learned scores can still saturate; scaling only removes this width-driven effect.

## 4. Multi-head self-attention

**Repeat scaled attention $H$ times.** Typically $H$ divides $D$, giving each head width $d_h=D/H$. Each head projects the full input independently:

$$
\begin{aligned}
\mathbf{V}_h &= \boldsymbol{\beta}_{vh}\mathbf{1}^{T}+\boldsymbol{\Omega}_{vh}\mathbf{X},\\
\mathbf{Q}_h &= \boldsymbol{\beta}_{qh}\mathbf{1}^{T}+\boldsymbol{\Omega}_{qh}\mathbf{X},\\
\mathbf{K}_h &= \boldsymbol{\beta}_{kh}\mathbf{1}^{T}+\boldsymbol{\Omega}_{kh}\mathbf{X}.
\end{aligned}
$$

All three matrices are $d_h\times L$. Apply the previous step per head:

$$
\begin{aligned}
\mathbf{Y}_h &= \mathbf{V}_h\cdot\operatorname{softmax}_{m}\!\left(
\frac{\mathbf{K}_h^{T}\mathbf{Q}_h}{\sqrt{d_h}}
\right) \in \mathbb{R}^{d_h\times L}.
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
\in\mathbb{R}^{D\times L}.
$$

Different projections let the $H$ heads learn different relationships; the output projection recombines their $D/H$-wide representations.

_Notation adapted from Simon J. D. Prince's [Understanding Deep Learning](https://udlbook.github.io/udlbook/)._

## Visual explanations

- [3Blue1Brown — Attention in transformers, step by step](https://www.3blue1brown.com/lessons/attention/): visual intuition for queries, keys, attention weights, and value mixing. [Watch the video](https://www.youtube.com/watch?v=eMlx5fFNoYc).
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): illustrated calculations and a clear walkthrough of multi-head concatenation and output projection. Its matrix diagrams use tokens as rows; this cheatsheet uses columns.
- [3Blue1Brown — Softmax](https://www.3blue1brown.com/lessons/gpt/#softmax): how scores become probabilities and how scaling changes their concentration.

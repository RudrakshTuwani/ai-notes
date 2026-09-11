# Normalization

**LayerNorm: center → rescale → learn scale and bias.**

**RMSNorm: rescale → learn scale.**

**Why normalize?** Hidden-state magnitudes can change as representations pass through layers and parameters change during training. Normalization controls the scale presented to the next computation, helping stabilize training. Learned channel scales let the model adjust that normalized representation.

!!! note "Notation · tokens are columns"

    - $B$: batch size; $L$: sequence length; $D$: channel / embedding width.
    - $\mathbf{x}_l\in\mathbb{R}^{D}$: token's hidden representation at position $l$; $x_{il}$: its channel $i$.
    - Tokens are columns: $\mathbf{X}\in\mathbb{R}^{D\times L}$; batched shape $B\times D\times L$.
    - $\boldsymbol{\gamma},\boldsymbol{\beta}\in\mathbb{R}^{D}$: learned scale and bias; $\odot$: elementwise multiplication.
    - $\epsilon>0$: numerical stability; $\mathbf{1}\in\mathbb{R}^{D}$: all ones. Batch index omitted below.

Both operations map $\mathbb{R}^{D}\to\mathbb{R}^{D}$ independently per token, preserving the full $B\times D\times L$ shape. See [Attention](attention.md) for the same notation.

## 1. LayerNorm

**One mean and variance per token, computed across its $D$ channels.**

$$
\begin{aligned}
\mu_l&=\frac{1}{D}\sum_{i=1}^{D}x_{il},\\
\sigma_l^2&=\frac{1}{D}\sum_{i=1}^{D}(x_{il}-\mu_l)^2.
\end{aligned}
$$

$$
\boxed{
\operatorname{LN}[\mathbf{x}_l]
=\boldsymbol{\gamma}\odot
\frac{\mathbf{x}_l-\mu_l\mathbf{1}}{\sqrt{\sigma_l^2+\epsilon}}
+\boldsymbol{\beta}
}
$$

- **Center:** subtract the token's mean across channels.
- **Rescale:** give those channels approximately unit variance when $\sigma_l^2\gg\epsilon$.
- **Learn:** $\boldsymbol{\gamma},\boldsymbol{\beta}$ let each channel recover a useful scale and offset.

The zero mean and approximately unit variance describe the vector **before the learned scale and bias**. The final output need not have either property. Variance divides by $D$ because it measures this vector's spread, rather than estimating a population variance from a sample.

### Why across channels?

- **The unit is one token.** Its $D$ channels jointly describe its state. Normalizing them controls the scale entering the next computation.
- **Keep positions independent.** Normalizing across positions would make statistics depend on other tokens or padding. Including future tokens would break decoder causality.
- **Channels can still differ.** Each has its own learned scale and bias. Attention handles interactions between tokens.

### Why doesn't batch composition matter?

**Only the channel index is summed:**

$$
\mu_{bl}=\frac{1}{D}\sum_{i=1}^{D}x_{bil}.
$$

- **No other examples enter the statistics.** The variance also uses only this token's channels.
- **Same vector + same parameters → same output**, whether processed alone or beside unrelated sequences, up to numerical rounding.
- **Same rule in training and evaluation.** No running averages are needed. BatchNorm instead pools statistics across examples during training.

!!! note "Shared parameters ≠ shared statistics"

    $\boldsymbol{\gamma},\boldsymbol{\beta}$ are shared across tokens and sequences within a layer. Batch composition affects gradients and later parameter updates, but not this normalization of a fixed input with fixed parameters.

[LayerNorm paper](https://arxiv.org/abs/1607.06450)

## 2. RMSNorm

**Skip centering. Divide by the root mean square.**

$$
r_l^2=\frac{1}{D}\sum_{i=1}^{D}x_{il}^2,
\qquad r_l=\sqrt{r_l^2}\text{ is the RMS}.
$$

$$
\boxed{
\operatorname{RMSNorm}[\mathbf{x}_l]
=\boldsymbol{\gamma}\odot
\frac{\mathbf{x}_l}{\sqrt{r_l^2+\epsilon}}
}
$$

Same channel axis and batch independence. The standard form learns a scale, with no additive bias.

### Why less computation?

- **LayerNorm:** compute mean and variance, subtract mean, rescale.
- **RMSNorm:** compute mean square, rescale. No mean computation or subtraction.
- **Why it adds up:** normalization repeats across tokens and layers. Fewer statistics simplify arithmetic and reductions across channels.
- **Same complexity:** both take $O(BLD)$ work. RMSNorm reduces constant cost; actual speedup depends on hardware and kernels, including LayerNorm fusion.

### What does it give up?

$$
\underbrace{r_l^2}_{\text{mean square}}
=\underbrace{\sigma_l^2}_{\text{variance}}+\mu_l^2.
$$

- **LayerNorm measures spread around the mean.** Adding the same offset to every channel cancels out.
- **RMSNorm measures magnitude relative to zero.** An offset changes the result; the output need not have zero mean.
- **Both control scale.** A positive input scaling cancels exactly for $\epsilon=0$ and nonzero denominators; approximately when $\epsilon$ is negligible relative to both the original and scaled variance or mean square.

The tradeoff: **retain scale normalization, give up offset invariance, simplify computation.**

[RMSNorm paper](https://arxiv.org/abs/1910.07467)

## 3. Worked example · same spread, different offset

Let $D=2$, $\boldsymbol{\gamma}=\mathbf{1}$, $\boldsymbol{\beta}=\mathbf{0}$, and $\epsilon=0$. Both inputs below have nonzero variance and RMS.

| Input $\mathbf{x}$ | Mean $\mu$ | Variance $\sigma^2$ | RMS $r$ | LayerNorm | RMSNorm |
| --- | --- | --- | --- | --- | --- |
| $[1,3]^T$ | $2$ | $1$ | $\sqrt{5}$ | $[-1,1]^T$ | $[1,3]^T/\sqrt{5}$ |
| $[3,5]^T$ | $4$ | $1$ | $\sqrt{17}$ | $[-1,1]^T$ | $[3,5]^T/\sqrt{17}$ |

For the first row, LayerNorm subtracts $2$ and divides by $\sqrt{1}$; RMSNorm divides by $\sqrt{(1^2+3^2)/2}=\sqrt{5}$.

**Add $2$ to every channel:** LayerNorm gives the same result because centering removes the offset. RMSNorm preserves the vector's direction by dividing every channel by the same scalar; shifting the input changes that direction. A later learned channel scale can change direction again.

## 4. Pre-norm vs. post-norm

The difference is **where normalization sits relative to each residual addition**. A transformer block applies this pattern twice: once for attention, once for the feed-forward network.

```python
# Post-norm: normalize after adding the residual
x = norm1(x + attention(x))
x = norm2(x + feed_forward(x))

# Pre-norm: normalize the input to each sublayer
x = x + attention(norm1(x))
x = x + feed_forward(norm2(x))
```

*Pseudocode: `norm1` and `norm2` have separate learned parameters; dropout is omitted.*

- **Post-norm:** normalization acts on the combined signal—the original `x` plus the sublayer's update.
- **Pre-norm:** only the update branch receives normalized input. The residual path carries `x` directly through the addition.
- **Why this helps:** for pre-norm, $\mathbf{y}=\mathbf{x}+F(\operatorname{norm}(\mathbf{x}))$, where $F$ is the attention or feed-forward sublayer, the derivative contains an identity term: $\partial\mathbf{y}/\partial\mathbf{x}=\mathbf{I}+\cdots$. Here $\mathbf{I}$ is the identity map on the input. This gives gradients a direct route through stacked residual blocks. In post-norm, even that route passes through normalization.
- **Training implication:** pre-norm generally makes deep transformers easier to optimize. It can reduce sensitivity to learning-rate warmup; it does **not** guarantee stable training or make warmup universally unnecessary. [Xiong et al., 2020](https://arxiv.org/abs/2002.04745)

**Placement and normalization type are separate choices:** pre-norm can use LayerNorm or RMSNorm.

## 5. Check your understanding

### Easy

**1. For an input of shape $B\times D\times L$, how many means does LayerNorm compute, and what is the output shape?**

??? note "Answer"

    $BL$ means: one per token per sequence, each using $D$ channels. The output shape remains $B\times D\times L$.

**2. Does LayerNorm's final output always have zero mean? Does RMSNorm center its input?**

??? note "Answer"

    Neither guarantees a zero-mean final output. LayerNorm centers before applying the learned scale and bias, which can change the mean. RMSNorm performs no centering.

### Medium

**3. Let $\mathbf{x}=c\mathbf{1}$ with $c\ne0$. What does each normalization return for $\epsilon>0$?**

??? note "Answer"

    Every channel equals the mean, so $\sigma^2=0$ and LayerNorm returns $\boldsymbol{\beta}$. The positive $\epsilon$ keeps the denominator nonzero.

    For RMSNorm, $r^2=c^2$, giving

    $$
    \operatorname{RMSNorm}[\mathbf{x}]
    =\frac{c}{\sqrt{c^2+\epsilon}}\boldsymbol{\gamma}.
    $$

    When $c^2\gg\epsilon$, this is approximately $\boldsymbol{\gamma}$ for $c>0$, or $-\boldsymbol{\gamma}$ for $c<0$. RMSNorm retains the common offset that LayerNorm removes.

**4. Derive $r^2=\sigma^2+\mu^2$. When do LayerNorm and RMSNorm give the same output if they use the same $\boldsymbol{\gamma}$ and $\epsilon$, and LayerNorm has $\boldsymbol{\beta}=\mathbf{0}$?**

??? note "Answer"

    Expand the variance, using $\frac{1}{D}\sum_i x_i=\mu$:

    $$
    \begin{aligned}
    \sigma^2
    &=\frac{1}{D}\sum_i(x_i-\mu)^2\\
    &=\frac{1}{D}\sum_i x_i^2-2\mu^2+\mu^2\\
    &=r^2-\mu^2.
    \end{aligned}
    $$

    A sufficient condition is $\mu=0$: centering changes nothing, and $r^2=\sigma^2$, so the numerators and denominators match.

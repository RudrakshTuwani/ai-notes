# Transformer layer

**Normalize → mix tokens → add residual → normalize → transform each token → add residual.**

A Transformer layer combines communication between tokens with a nonlinear transformation of each token's features. Both sublayers learn **updates to a shared residual stream**, preserving its shape so layers can be stacked.

For the component definitions, see [multi-head self-attention](attention.md#4-multi-head-self-attention) and [normalization](normalization.md). This page connects them into one **pre-norm** layer.

## 1. The pre-norm layer

![Pre-norm Transformer layer: a D by L input enters LayerNorm, multi-head self-attention, and a residual addition, then a second LayerNorm, a shared feed-forward network applied per token, and a second residual addition. Each residual bypasses both normalization and its sublayer.](../assets/transformer-layer-pre-norm.svg)

*[Open the diagram at full size](../assets/transformer-layer-pre-norm.svg). ×L means shared parameters applied independently to each token.*

!!! note "Notation · tokens are columns"

    - $L$: sequence length; $D$: embedding / residual width.
    - $\mathbf{X},\mathbf{U},\mathbf{Y}\in\mathbb{R}^{D\times L}$: layer input, state after attention's residual addition, and layer output.

Here $\operatorname{MHSA}$ is multi-head self-attention, $\operatorname{FFN}$ is the feed-forward network, and $\operatorname{LN}_1,\operatorname{LN}_2$ are LayerNorms with separate learned parameters. Dropout is omitted.

$$
\boxed{
\begin{aligned}
\mathbf{U} &= \mathbf{X}+\operatorname{MHSA}[\operatorname{LN}_1[\mathbf{X}]],\\
\mathbf{Y} &= \mathbf{U}+\operatorname{FFN}[\operatorname{LN}_2[\mathbf{U}]].
\end{aligned}
}
$$

**Read each residual loop as “keep the current state, then add an update.”** The first skip carries $\mathbf{X}$; the second carries the updated state $\mathbf{U}$. Each update has shape $D\times L$, so both additions are elementwise.

Normalization sits at the **entrance to each update branch**; see [pre-norm vs. post-norm](normalization.md#4-pre-norm-vs-post-norm) for the placement and gradient intuition. A pre-norm stack typically adds a final normalization **after the last layer**, outside the block shown here. [Xiong et al., 2020](https://arxiv.org/html/2002.04745v2)

## 2. Positions vs. channels

**Attention mixes across positions; the FFN mixes across channels.** For $\mathbf{X}\in\mathbb{R}^{D\times L}$, each column is a token and each row is a channel:

- **Attention:** each token gathers information from other token columns, subject to the attention mask.
- **FFN:** each token's channels are combined and transformed within its own column. The same network is applied independently at every position.

Attention's projections also mix channels; the distinguishing role of attention is **communication between positions**. See [multi-head self-attention](attention.md#4-multi-head-self-attention) for the projections and value mixing.

## 3. The FFN expands, then contracts

The usual FFN first projects each token into a wider hidden representation, applies a nonlinearity, then projects back to the residual width:

$$
\underbrace{D\times L}_{\text{input}}
\xrightarrow{\text{expand}}
\underbrace{D_{\mathrm{ff}}\times L}_{\text{hidden}}
\xrightarrow{\text{nonlinearity}}
\underbrace{D_{\mathrm{ff}}\times L}_{\text{activations}}
\xrightarrow{\text{project back}}
\underbrace{D\times L}_{\text{update}}.
$$

Here $D_{\mathrm{ff}}$ is the hidden channel width, usually larger than $D$. The wider intermediate representation gives the network room to form more feature combinations, which it then combines into a $D$-dimensional update for residual addition. The token count $L$ stays fixed.

For example, the original Transformer used **512 → 2048 → 512 channels per token**. The two projections are learned transformations, not inverse compression and decompression steps. [Vaswani et al., §3.3](https://arxiv.org/html/1706.03762v7#S3.SS3)

## 4. Check your understanding

### Easy

**1. What does the second residual connection carry?**

??? note "Answer"

    $\mathbf{U}$, the state after the first residual addition. It bypasses both $\operatorname{LN}_2$ and the FFN.

### Medium

**2. If both sublayers return zero updates for every input, what does the pre-norm layer compute?**

??? note "Answer"

    The identity: $\mathbf{U}=\mathbf{X}$ and $\mathbf{Y}=\mathbf{X}$. Normalization is inside each update branch, so it does not alter the skip path. A final normalization outside the stack would still act on the result.

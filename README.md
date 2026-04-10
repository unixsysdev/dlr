# Decoupled Latent Reasoner (DLR)

**Continuous-Space Mathematical Reasoning via Rectified Flow Over Learned Geometric Trajectories**

---

## Abstract

We present the **Decoupled Latent Reasoner** (DLR), a three-module architecture that decouples mathematical reasoning from token generation by performing inference in continuous latent space. Rather than predicting the next token autoregressively — where each step compounds hallucination risk — DLR first constructs a complete geometric reasoning trajectory through a learned latent manifold, then recovers discrete text only at the final stage.

The architecture consists of:
1. A **Text-JEPA** (Joint Embedding Predictive Architecture) that learns the latent geometry of logical transitions via self-supervised prediction of future proof states.
2. A **Rectified Flow Expert** (DiT backbone) that generates optimal-transport trajectories between premise and conclusion through deterministic ODE integration.
3. A **Scribe Decoder** that translates the continuous trajectory into discrete mathematical text via sliding-window cross-attention.

This repository provides a complete, GPU-validated Proof-of-Concept implementation, including a custom math-native tokenizer, configurable compute optimizations (torch.compile, BF16, Liger kernels), and a four-metric evaluation dashboard. The architecture is designed for validation on a single high-memory GPU (H200/B200) within 24 hours.

---

## Motivation

Autoregressive language models generate mathematical proofs one token at a time. This creates a fundamental architectural limitation: each token prediction is conditioned only on previous tokens, with no mechanism to plan ahead or verify the logical coherence of the full argument before committing to it.

Consider how a human mathematician works. They first grasp the problem structure, visualize the geometric path from premise to conclusion, and only then write the steps. The writing is a *transcription* of an already-completed reasoning process — not the reasoning itself.

DLR replicates this workflow:
- The **Text-JEPA** learns to represent reasoning states as points in a continuous $d$-dimensional manifold, capturing the logical "momentum" of a proof.
- The **Flow Expert** computes the shortest geometric path between the premise and the conclusion using Rectified Flow — an optimal-transport ODE that is mathematically forbidden from wandering.
- The **Decoder** reads this pre-computed trajectory and transcribes it into text.

The critical insight is that **reasoning happens in continuous space, not in token space**. Token generation is a post-hoc decoding step, not a cognitive one.

---

## Architecture

### Module A: Text-JEPA (Semantic Anchor)

A 1D-sequence adaptation of I-JEPA for structured mathematical reasoning.

**Training Objective**: Given a premise and the first $k$ steps of a proof, predict the latent representation of step $k+1$ in continuous space.

$$\mathcal{L}_{\text{JEPA}} = \| P(E_x(\text{context})) - \text{sg}(E_y(\text{target})) \|_2^2$$

| Component | Role | Gradient |
|---|---|---|
| Context Encoder $E_x$ | Encodes premise + previous steps | ✓ Trainable |
| Target Encoder $E_y$ | Encodes the next step (via EMA) | ✗ Frozen (EMA) |
| Predictor $P$ | Narrow MLP with learned "next-step" embedding | ✓ Trainable |

**EMA Schedule**: Cosine annealing from $\tau = 0.996 \to 1.0$, preventing representation collapse while maintaining target stability.

**Collapse Detection**: Per-dimension variance $\text{Var}(z_{\text{target}})$ is monitored continuously; values below $10^{-4}$ trigger early warnings.

### Module B: Rectified Flow Expert (Logic Engine)

A DiT-based (Diffusion Transformer) model that generates continuous reasoning trajectories from noise via Rectified Flow.

**Rectified Flow Formulation**:
$$x_t = t \cdot Z_{\text{true}} + (1 - t) \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$
$$v_{\text{true}} = Z_{\text{true}} - \varepsilon$$
$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, \varepsilon}\left[\| v_\theta(x_t, t, z_0, z_{\text{target}}) - v_{\text{true}} \|^2 \cdot \mathbf{m}\right]$$

where $\mathbf{m}$ is the active waypoint mask (Velocity Zero-Out).

**V3 Unified Trajectory Architecture**: The Flow Expert receives the premise as a compressed $d$-dimensional vector $z_0$ via Adaptive Layer Normalization (AdaLN), not via a separate KV cache. This eliminates the "double-dip" memory redundancy and compresses the entire problem context into the conditioning signal:

$$\text{cond} = \text{MLP}([z_0 \| z_{\text{target}} \| \phi(t)])$$

**ODE Solvers**: Euler (1st order) and Heun (2nd order Runge-Kutta) integration. Heun corrects for curvature in the velocity field at the cost of 2× network evaluations per step.

### Module C: Scribe Decoder (Y-Decoder)

A lightweight causal decoder that translates the $N \times d$ continuous trajectory into discrete text tokens via sliding-window cross-attention.

**Sliding Window Constraint**: Token $i$ attends only to trajectory waypoints $[k-w, k+w]$ where $k = \lfloor i \cdot N / L \rfloor$. This prevents attention smearing and enforces local trajectory-to-token alignment.

**Weight Tying**: The input embedding matrix and lm_head output projection share weights, halving parameter count on the vocabulary dimension.

---

## Key Design Decisions

### Velocity Zero-Out (Variable-Length Proofs)

Mathematical proofs vary in length. For a proof with $K < N_{\max}$ reasoning steps:
- Waypoints $0, \ldots, K-1$ contain the actual encoded states.
- Waypoints $K, \ldots, N_{\max}-1$ are filled with copies of $z_{\text{final}}$.
- The loss mask is $\mathbf{m}_i = \mathbb{1}[i < K]$.

The physical interpretation: once the proof reaches its conclusion, the velocity field drops to zero. The trajectory "parks" at $z_{\text{final}}$. The model learns that zero velocity means "done reasoning."

### Unified Trajectory Cache (No Double-Dip)

Previous iterations passed a full token-level prompt KV cache to both the Flow Expert and Decoder. The V3 architecture eliminates this redundancy:

- The JEPA's cumulative encoding compresses the full problem context into $z_0 = E_y([\text{PREMISE}])$, a single $d$-dimensional vector.
- The Flow Expert receives $z_0$ via AdaLN conditioning — same information, $S \times$ smaller memory footprint (where $S$ is the prompt sequence length).
- The Decoder cross-attends to the generated trajectory, which already contains $z_0$ as its first waypoint.

Memory reduction: from $[S \times d] + [N \times d]$ to $[N \times d]$ only.

### Custom Math-Native Tokenizer

A BPE tokenizer trained directly on the mathematical reasoning corpus, with:
- Digit-boundary pre-tokenization: numbers remain intact as single tokens (`42`, `100`, `256`)
- Structural tokens (`[PREMISE]`, `[STEP]`, `[/STEP]`, `[CONCLUSION]`) as native vocabulary entries
- 8,192 token vocabulary (3.7× smaller than BERT's 30,522), reducing the decoder softmax computational cost

---

## Evaluation Protocol

Four metrics designed to isolate each stage of the information pipeline:

| Metric | What It Measures | Success Criterion |
|---|---|---|
| **A. JEPA Collapse Variance** | $\text{Var}(z_{\text{target}})$ over training | Must stay $> 10^{-4}$ |
| **B. Cosine Monotonicity** | Whether $\cos(z_i, z_{\text{final}})$ increases along the trajectory | High monotonicity rate |
| **C. Flow Endpoint Distance** | $\|z_{\text{generated}}^{\text{final}} - z_{\text{true}}^{\text{final}}\|_2$ | Decreasing with training |
| **D. Token Recovery Rate** | Exact match of numbers and operators in decoded text | Above baseline (>10% at PoC scale) |

---

## Compute Optimization

The implementation employs a layered optimization strategy:

| Layer | Mechanism | Effect |
|---|---|---|
| Graph Compilation | `torch.compile` | Operator fusion, 1.3-2× throughput |
| Mixed Precision | BF16 autocast (forward) + FP32 (EMA, gradients) | 2× memory reduction, 2× throughput |
| Matrix Precision | TF32 matmul on Ampere+ | Free 2-3× on matmuls |
| Fused Kernels | Liger fused cross-entropy (decoder) | Avoids materializing $[B \cdot L, V]$ logits |
| Data Loading | Persistent workers, pinned memory | Eliminates inter-epoch fork overhead |

**Safety invariant**: EMA updates always execute in FP32 on the uncompiled model to prevent silent corruption of the target encoder geometry.

---

## Repository Structure

```
dlr/
├── config.py               # Hyperparameters, compute opts, PoC + production profiles
├── data_pipeline.py         # NuminaMath-CoT loading, step parsing, 3 dataset classes
├── train_tokenizer.py       # Custom math BPE tokenizer training
├── modules/
│   ├── text_jepa.py         # Module A: Context/Target encoder + Predictor + EMA
│   ├── flow_expert.py       # Module B: DiT + AdaLN + Rectified Flow + Heun solver
│   └── decoder.py           # Module C: Causal decoder + sliding-window cross-attn
├── train_jepa.py            # Phase 1: JEPA training with collapse detection
├── extract_trajectories.py  # Phase 1.5: Frozen encoder → Z_true extraction
├── train_flow.py            # Phase 2: Rectified Flow training with masked loss
├── train_decoder.py         # Phase 3: Decoder training with Liger CE
├── visualize.py             # 4-metric evaluation dashboard
├── run_poc.py               # Master orchestrator (--production for H200/B200)
├── test_data_pipeline.py    # Data pipeline validation (no GPU required)
├── RUNBOOK.md               # Production execution guide
└── requirements.txt         # Dependencies
```

---

## Usage

```bash
# Install
pip install -r requirements.txt

# Train math tokenizer
python train_tokenizer.py --vocab-size 8192 --n-samples 5000

# Validate data pipeline
python test_data_pipeline.py --n-samples 500

# PoC run (d=128, ~7h on RTX 4090)
python run_poc.py

# Production run (d=1024, ~14-18h on H200/B200)
python run_poc.py --production

# Resume from specific phase
python run_poc.py --production --skip-to flow
```

---

## Data

Training data is sourced from **NuminaMath-CoT** (AI-MO), a corpus of ~860,000 mathematical problems with structured chain-of-thought solutions spanning arithmetic, algebra, geometry, combinatorics, and number theory.

Solutions are parsed into discrete reasoning steps using a multi-strategy parser (explicit markers → numbered lists → paragraph boundaries → line boundaries), yielding an average of 6.6 steps per problem with a mean context length of 177 words.

---

## Scaling Properties

| Configuration | d | Problems | Parameters | Hardware | Time |
|---|---|---|---|---|---|
| Quick Test | 32 | 50 | ~100K | CPU | ~5 min |
| PoC | 128 | 1,000 | ~400K | RTX 4090 | ~7h |
| Validation | 1024 | 100,000 | ~50M | H200 (141GB) | ~14-18h |
| Production | 1024 | 860,000 | ~50M | 8×H200 | ~10-14 days |

---

## Related Work

- **I-JEPA** (Assran et al., 2023): Image-level joint embedding prediction; DLR adapts this to 1D text sequences with cumulative pooling.
- **Rectified Flow** (Liu et al., 2022): Optimal-transport ODE for straight-line generative flows; DLR applies this to discrete reasoning trajectories rather than images.
- **DiT** (Peebles & Xie, 2023): Diffusion Transformer with AdaLN conditioning; DLR uses the DiT block architecture for velocity field prediction.
- **Latent Diffusion** (Rombach et al., 2022): Compression to latent space before generation; DLR extends this principle from visual to logical domains.

---

## License

MIT

# Review

## 1. The Oracle -> Subgoal Proposer (Distributional Planning)

You recognized that a deterministic Oracle is a fantasy for computationally irreducible problems. Moving to a distribution or a subgoal chain is the correct fix.

### What To Look At

Look into Michael Janner's Diffuser and Decision Diffuser. They don't predict a terminal state and draw a straight line; they diffuse entire trajectories where the "goal" is just a high-reward conditioning region.

### The Implementation

Instead of an MLP predicting `z_final`, train a conditional prior that outputs a Gaussian mixture or a diffusion prior over intermediate subgoals. Your Flow Expert then becomes a Schrödinger Bridge or uses Conditional Flow Matching, bridging between `z_0` and a sampled subgoal, rather than a forced deterministic point.

## 2. The Compression Bottleneck -> Structured Symbolic State

Pooling an entire mathematical sequence into a single `[d]` vector is where your V1 system bleeds out its symbolic fidelity. You cannot transcribe math from a semantic smoothie.

### What To Look At

Look at Slot Attention (Locatello et al.) and Object-Centric Learning. Also look at how Continuous Diffusion for Categorical Data (CDCD) and Diffusion-LM (Li et al.) handle discrete tokens in continuous space.

### The Implementation

You must move your latent representation from `[N, d]` to `[N, S, d]` where `S` is sequence length, or to a bounded set of entity slots `[N, K, d]` where `K` is the number of mathematical entities or operands. This means your Flow Expert (DiT) becomes a 2D Transformer, attending across both the trajectory dimension and the sequence or slot dimension. It's computationally heavier, but it is the only way to preserve the exact variable bindings and operator structures required for lossless transcription.

## 3. The Energy Critic -> The Verifier (MCTS Over Latents)

You correctly identified that isotropic Gaussian noise teaches topological smoothness, not mathematical truth.

### What To Look At

AlphaGeometry (Trinh et al.) is the gold standard here: they pair a neural proposer with a fast symbolic deduction engine. Also look at DeepSeek-Prover or projects integrating Lean 4 for formal verification during generation.

### The Implementation

If you keep the continuous flow, your verifier needs to act as a guidance signal, similar to classifier-free guidance, during the ODE integration. Alternatively, generate a batch of candidate latent trajectories, decode them, and run them through SymPy or a formal verifier to score them. You can then use those verified and rejected trajectories to train a robust reward model, via DPO or PPO, that actually penalizes logical contradictions rather than just off-manifold drift.

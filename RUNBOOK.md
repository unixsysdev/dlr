# DLR Production Run — H200/B200 Runbook

## Quick Start (Copy-Paste Sequence)

```bash
# 1. Clone
git clone git@github.com:unixsysdev/dlr.git && cd dlr

# 2. Install deps
pip install -r requirements.txt

# 3. Train the custom math tokenizer (~2 min)
python train_tokenizer.py --vocab-size 8192 --n-samples 5000

# 4. Validate data pipeline (~1 min)
python test_data_pipeline.py --n-samples 500

# 5. Fire the production run (all V5 modules active)
nohup python run_poc.py --production > dlr_run.log 2>&1 &
echo $! > dlr.pid

# 6. Monitor
tail -f dlr_run.log
```

## What `--production` Does

| Parameter | Value |
|---|---|
| d_model | 1024 |
| max_seq_len | 1024 |
| n_waypoints | 32 |
| n_samples | 100,000 (90/10 train/test split) |
| Encoder layers | 8 |
| Flow layers | 12 |
| Decoder layers | 4 |
| Oracle layers | 6 |
| Pooling | Attention-weighted (learned query) |
| Batch sizes | 256/128/128 |
| Oracle exposure | 20% target replacement during flow training |
| Decoder training | Mixed extracted/generated trajectory curriculum |
| ODE solver | Heun (2nd order) + hard boundary |
| VICReg | λ_inv=25, λ_var=25, λ_cov=1 |
| Eval decoding | Greedy (`temperature=0.0`) |
| Energy Critic | Spectral norm + manifold regularization penalty |
| Energy penalty α | 0.1 |
| Compute | compile + bf16 + tf32 + liger |

## Expected Timeline (~14-18h)

```
Phase 1   (JEPA+Oracle+VICReg):  ~4-6h  → checkpoints/jepa_final.pt
Phase 1.5 (Extract):             ~30min → data/trajectories.pt
Phase 2   (Flow+Energy Critic):  ~6-8h  → checkpoints/flow_final.pt
                                         → checkpoints/energy_critic_final.pt
Phase 3   (Decoder):             ~2-3h  → checkpoints/decoder_final.pt
Eval:                            ~30min → plots/*.png
```

## Monitoring Checkpoints

```bash
# Check GPU utilization
nvidia-smi -l 5

# Check what phase we're on
grep "PHASE\|Epoch\|✓\|⚠" dlr_run.log | tail -20

# Phase 1: VICReg sub-losses (inv should decrease, var should stabilize near 0)
grep "inv=\|var=\|ora=" dlr_run.log | tail -10

# Phase 2: Flow + stop + Energy penalty
grep "flow=\|stop=\|e_pen=\|crit=" dlr_run.log | tail -10

# Check z_var (VICReg should keep this healthy automatically)
grep "z_v=" dlr_run.log | tail -5

# Check flow loss trend
grep "Flow.*Loss\|Epoch.*Loss" dlr_run.log | tail -10
```

## What to Watch For

| Signal | Meaning | Action |
|---|---|---|
| `var_loss > 1.0` sustained | VICReg variance not kicking in | Increase `vicreg_lambda_var` |
| `oracle_loss` not decreasing | Oracle can't predict final goal states | Increase `oracle_layers` (try 6→8) |
| `stop` stays near chance | Flow stop head is not learning endpoint length | Increase `stop_loss_weight` or inspect masks |
| `e_pen` near 0 from start | Energy Critic too weak | Increase `energy_noise_std` |
| `crit_loss` near 0 | Critic perfectly separates (good) | Normal — means negatives are easy |
| `flow_loss` stalls > 1.0 | Flow can't learn velocity | Check trajectory quality, try more JEPA epochs |

## If Something Breaks

```bash
# Resume from Phase 2 (skips JEPA, uses saved checkpoint)
python run_poc.py --production --skip-to flow

# Resume from Phase 3
python run_poc.py --production --skip-to decoder

# Just re-run eval
python run_poc.py --production --skip-to eval

# Disable compile if it crashes (rare edge case)
python run_poc.py --production --no-compile

# Disable BF16 if NaN losses appear
python run_poc.py --production --no-bf16
```

## Output Files

```
checkpoints/
  tokenizer/              # Custom math BPE (8K vocab)
  jepa_final.pt           # Frozen JEPA + Oracle
  flow_final.pt           # Trained Flow Expert
  energy_critic_final.pt  # Trained Energy Critic
  decoder_final.pt        # Trained Scribe Decoder

data/
  trajectories.pt         # Extracted Z_true [P, 32, 1024]
  jepa_history.json       # Phase 1 metrics (VICReg + Oracle losses)
  flow_history.json       # Phase 2 metrics (flow + energy + critic)
  decoder_history.json    # Phase 3 metrics
  full_pipeline_results.json  # Metric E honest eval results + answer EM/symbolic match

plots/
  01_jepa_training.png              # VICReg sub-losses + z_var + EMA schedule
  02_cosine_monotonicity.png        # Do trajectories progress logically?
  03_trajectory_visualization.png   # UMAP/t-SNE of latent paths
  04_flow_training.png              # Flow MSE + energy penalty curves
  05_flow_endpoint.png              # Can the flow land at z_final?
  06_token_recovery.png             # Metric D: decoder on Z_true (diagnostic)
  07_decoder_training.png           # CE loss + perplexity
  08_full_pipeline_recovery.png     # Metric E: HONEST test (Oracle→Flow→Decoder)
```

## The Honest Test (Metric E)

The primary pass/fail is `08_full_pipeline_recovery.png`. This runs **on held-out test data only** (10% of the dataset, never seen during training):

```
premise → JEPA.encode (attn-pool) → z_0
z_0 → Oracle.predict_goal → ĉ_final
noise → Flow.generate(z_0, ĉ_final, heun, hard_boundary) → Z_gen
Z_gen + active_mask → Decoder.generate → text
text vs. ground_truth → recovery rate
```

**No ground-truth leakage at any stage. No training trajectory contamination.** Metric E now encodes each held-out premise directly through the JEPA instead of loading `trajectories.pt`, and reports token recovery, final-answer exact match, and symbolic equivalence when parseable.

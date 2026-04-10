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

# 5. Fire the production run (Heun solver, all compute opts ON)
nohup python run_poc.py --production > dlr_run.log 2>&1 &
echo $! > dlr.pid

# 6. Monitor
tail -f dlr_run.log
```

## What `--production` Does

| Parameter | Value |
|---|---|
| d_model | 1024 |
| n_waypoints | 32 |
| n_samples | 100,000 |
| Encoder layers | 8 |
| Flow layers | 12 |
| Decoder layers | 4 |
| Batch sizes | 256/128/128 |
| ODE solver | Heun (2nd order) |
| Compute | compile + bf16 + tf32 + liger |

## Expected Timeline (~14-18h)

```
Phase 1   (JEPA):      ~4-6h    → checkpoints/jepa_final.pt
Phase 1.5 (Extract):   ~30min   → data/trajectories.pt
Phase 2   (Flow):      ~6-8h    → checkpoints/flow_final.pt
Phase 3   (Decoder):   ~2-3h    → checkpoints/decoder_final.pt
Eval:                  ~30min   → plots/*.png
```

## Monitoring Checkpoints

```bash
# Check GPU utilization
nvidia-smi -l 5

# Check what phase we're on
grep "PHASE\|Epoch\|✓\|⚠" dlr_run.log | tail -20

# Check for JEPA collapse (CRITICAL — if this happens, kill and adjust)
grep "COLLAPSE\|z_var" dlr_run.log | tail -5

# Check flow loss trend
grep "Flow.*Loss" dlr_run.log | tail -10
```

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
  tokenizer/          # Custom math BPE (8K vocab)
  jepa_final.pt       # Frozen JEPA for trajectory extraction
  flow_final.pt       # Trained Flow Expert
  decoder_final.pt    # Trained Scribe Decoder

data/
  trajectories.pt     # Extracted Z_true [P, 32, 1024]
  jepa_history.json   # Phase 1 metrics
  flow_history.json   # Phase 2 metrics
  decoder_history.json # Phase 3 metrics

plots/
  01_jepa_training.png       # Loss + z_var + EMA schedule
  02_cosine_monotonicity.png # Do trajectories progress logically?
  03_trajectory_visualization.png  # UMAP/t-SNE of latent paths
  04_flow_training.png       # Flow MSE loss curve
  05_flow_endpoint.png       # Can the flow land at z_final?
  06_token_recovery.png      # Can the decoder recover math tokens?
  07_decoder_training.png    # CE loss + perplexity
```

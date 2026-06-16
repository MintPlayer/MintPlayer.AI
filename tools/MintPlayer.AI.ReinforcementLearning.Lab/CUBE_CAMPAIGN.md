# Cube DAVI campaign — runbook

The teacher-free Rubik's-Cube value-net campaign (`--game cube-davi`): deep approximate value iteration over
`CubeModel`, no Kociemba/oracle. It is **resumable** (value net + Adam moments + curriculum depth + iteration
count + sampler RNG all checkpoint to the model store), **GPU-resident** when a CUDA device is present, and logs
a **BWAS capability probe** so you can watch true capability climb over the run (the greedy in-loop eval
understates it badly — search reads the net far deeper).

## The 1-billion-state campaign (DeepCubeA scale)

Why this size: the shipped 1024×4 net is already **optimal-length to depth 15** (search-bound, not capacity-bound).
The only training lever that pushes the frontier past d15 toward "any cube" is **more deep-state coverage** — i.e.
a DeepCubeA-scale campaign (~10⁹–10¹⁰ states). 1 B states is the sane first bite.

```bash
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game cube-davi --net residual --width 1024 --blocks 4 \
  --batch 1000 --lr 2e-3 --eps-sync 0.06 \
  --samples 1000000000 --max-depth 26 \
  --probe-depths 12,14,15,16 \
  --seed 1 --data <campaign-data-dir> --hours 12
```

- **`--samples 1000000000`** — hard stop at 1 B total states processed. It is **resumable across sessions**: the
  loop also honours `--hours`, so run it in 12-hour chunks (or any length) — re-running the exact command resumes
  from the last checkpoint and stops once 1 B *total* states are done. Keep `--batch` fixed across sessions so the
  state count stays exact (the counter is `iterations × batch`).
- **`--max-depth 26`** — let the solve-rate curriculum climb toward god's-number depth (QTM). It advances on
  mastery or force-advances after a stall, so it will keep deepening as the value propagates outward from the goal.
- **`--probe-depths 12,14,15,16`** — the in-loop BWAS probe runs a *cheap* 8k-expansion search, so it only shows
  signal near the current frontier. These four depths track whether capability is still climbing. (Deep d16+ is
  measured separately — see below.)
- **`--data <dir>`** — fresh empty dir = clean 1 B-from-scratch run (tests the scale hypothesis cleanly). To
  **warm-start from the shipped net instead**, point it at a dir containing `cube.value-davi-res.ckpt` +
  `…-state.ckpt`; it resumes that net (the `--samples` count is then approximate, since the original ran at a
  different batch).

### Expected wall-clock (single GPU, current 1024×4 net speed ≈ 3,050 states/s)

| States | Wall-clock |
|---|---|
| 1 B | **~3.8 days** (~91 GPU-hours) |
| 5 B | ~19 days |
| 10 B (full DeepCubeA) | ~38 days |

Caveats that push it **up**: a *wider* net is slower per sample (GPU-bound at ~2 TFLOP/s) so going past 1024-wide
multiplies the clock; checkpoint/eval overhead is real; and this is single-GPU — it scales down ~linearly with
more GPUs. **ROI honesty:** the net is already optimal to d15; this chases d16→20, and even DeepCubeA only reaches
"solved, ~60% optimal" there. Treat 1 B as a bounded experiment, not a guaranteed "any cube".

## Measuring deep capability (the real check, not the cheap probe)

The in-loop probe is a trend tracker. To measure true reach at the frontier, run an **eval-only heavy search** on
the checkpoint (no training), with a near-admissible weight and a big expansion budget, reporting length vs Kociemba:

```bash
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game cube-davi --net residual --eval-only --search --batched \
  --weight 1.0 --max-exp 100000 --max-depth 20 --vs-kociemba \
  --data <campaign-data-dir>
```

`weight 1.0` + goal-on-pop is optimal under an admissible heuristic; raise `--max-exp` for deeper reach (slower).
This is the command that produced the "12/12 optimal-length through d15, d16 83%, d17 42%" capability table.

## Outputs

- Checkpoints: `<data>/cube.value-davi-res.ckpt` (+ `…-state.ckpt`) — saved every eval; the web "Solve
  (self-taught AI)" button picks up a newer net automatically when copied into the web `data/` dir.
- CSVs in `<data>/logs/`: `cube-davi-res.csv` (per-depth greedy solve-rate over time) and
  `cube-davi-res-cap.csv` (the BWAS capability probe over time — the honest capability curve).

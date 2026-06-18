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

### The 10-billion-state brute-force campaign — the sample-bound lever (2026-06-16 conclusion)

The d14 accuracy wall was proven **sample/bootstrap-bound, NOT capacity or lr** (see `docs/OPTIMIZATIONS.md`
"Conclusion (2026-06-16)": the loss floor is invariant to lr 2e-3/1e-3/5e-4, width 1024→2048, and 690M samples).
So the *only* lever that deepens the net is DeepCubeA-scale compute (~10¹⁰ states). Recipe corrections vs the M-eff
autopilot, which matter at this scale:

- **Use plain uniform sampling, NOT the curriculum gate / `--frontier-bias` / `--auto-widen`.** The value-accuracy
  gate refuses to sample past the mastered frontier — stuck at d14 it would *never train d15-26*, starving the
  exact states that need to improve. DeepCubeA uses uniform scramble-length `[1, K]` and brute scale: shallow
  states (accurate targets) anchor and the accurate region propagates outward on its own. Pin the curriculum at the
  cap so sampling is uniform `[1, 26]`: `--set-curriculum-depth 26 --max-depth 26` and **omit** `--frontier-bias`
  and `--auto-widen`. (The gate is a small-scale sample-efficiency trick; at 10 B it's counterproductive.)
- **Width 1024, not 2048.** Capacity is ruled out, and 1024 trains ~4× more samples/hour — for a sample-bound
  problem that's 4× faster to 10 B.
- **Robustness for ~weeks unattended:** it's fully resumable (checkpoints every eval to `--data`), so run it in
  chunks via `--hours` and just re-run the same command after any reboot/sleep — it resumes net + Adam + curriculum
  + RNG and stops once 10 B *total* are done. **Disable OS sleep** (a sleeping laptop has killed runs before).

```bash
# Warm-start from the shipped 1024 net, uniform [1,26], 10 B states, chunked (re-run to resume):
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game cube-davi --net residual --width 1024 --blocks 4 \
  --batch 1000 --lr 1e-3 --eps-sync 0.06 \
  --set-curriculum-depth 26 --max-depth 26 \
  --samples 10000000000 --probe-depths 12,14,16,18 \
  --data <campaign-dir> --hours 12
```
**Honest expectation:** ~38 days on one 3060 (chunked over calendar weeks). Even at the full 10 B, DeepCubeA reaches
"solved, ~60% optimal" at the deep end — a big jump from today's ~d15-optimal / d16-partial, but **not** flawless
god's-number optimality. A cloud GPU (A100-class, or multi-GPU) would cut the wall-clock several-fold if weeks of
laptop GPU is disruptive. Promote the result into the web `data/` + `models/` (LFS) only after a heavy-search probe
confirms it actually beats the current net.

## Efficiency levers (M-eff)

New knobs and experiments to reach the same capability in less wall-clock / fewer samples. All default
to the previous behaviour, so the in-flight campaign is unaffected unless you opt in.

### New CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--target-sync-interval N` | `200` | Steps between bootstrap-target syncs (still gated by `--eps-sync`). Lower = the target tracks the online net more tightly early on; raise it late. It's a **step** count — if you change `--batch`, scale this to keep the *samples-per-sync* constant. |
| `--beta2 B` | `0.999` | Adam β₂. DeepCubeA uses `0.9999` for a steadier step once targets stretch to depth ~20+. **Pass the same value on resume** (the resident trainer always starts its moments fresh — see note). |
| `--checkpoint-every N` | `1` | Write a checkpoint only every Nth eval. On slow/HDD storage the per-eval write (weights + Adam moments) is real overhead; `N=5` keeps a recent rolling save without stalling the loop. The final state is always saved on exit. |
| `--frontier-bias` | off | Sample scramble depth with a triangular weighting toward the curriculum frontier (max of two uniform draws) instead of uniform `[1, depth]`. Concentrates samples where the value signal is still moving; easy depths converge early and stop needing a fixed batch share. **Use on a FRESH run** — it changes the sampler's RNG draw pattern, so it can't resume a uniform-sampled checkpoint cleanly. |
| `--grow-to W` + `--grow-at S` | off (0) | **Progressive growing.** Train at the narrow `--width` until `S` total samples, then Net2WiderNet-widen the residual trunk to `W` and continue. The widen is a function-preserving **warm start** (capability carries over — confirmed in `ResidualMlpTests.WidenTo_*`), so you pay the cheap narrow GEMM for the bulk of the run and the wide GEMM only near the frontier. |

> **Note — resident-path Adam moments do not persist across resume.** For a residual GPU campaign the
> actual optimizer is `DeviceResidualTrainer`, which allocates its own (zeroed) moment buffers each launch;
> the checkpointed host-`Adam` moments feed only the CPU autograd path. So a resumed residual run re-warms
> Adam from zero (a few hundred steps). Fixing this means persisting the device M/V buffers — tracked
> separately, out of scope for the M-eff flags.

### Experiment 1 — the 512×4 width validation (highest ROI: potential ~3×)

`OPTIMIZATIONS.md` concludes **width is not the bottleneck through d15** (the net is search-bound there).
GEMM cost is ~quadratic in width, so a **512-wide net trains ~3–4× faster per sample** and *should* reach
the same d15 capability. This decides the per-sample cost of every future campaign — run it before
committing to another wide multi-day run.

```bash
# Same recipe as the 1 B campaign but half-width, into a SEPARATE data dir. Compare the BWAS
# capability curve (logs/cube-davi-res-cap.csv) against the 1024×4 run at equal SAMPLES (not wall-clock).
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game cube-davi --net residual --width 512 --blocks 4 \
  --batch 1000 --lr 2e-3 --eps-sync 0.06 \
  --samples 100000000 --max-depth 26 \
  --probe-depths 12,14,15,16 \
  --seed 1 --data <512x4-data-dir> --hours 12
```

Decision rule: if 512×4 matches 1024×4's d12–15 cap curve at equal samples, adopt 512×4 as the campaign
default (≈3× cheaper). If it falls short at d15+, the width is buying frontier capacity — keep 1024×4 for
the deep push. (Width-doesn't-matter is proven only *through* d15; the 1 B run's whole point is d16–20.)

### Experiment 1b — progressive growing (the dynamic-width campaign)

Instead of betting on one width, start narrow and widen on demand. The early curriculum (d1–~d10) is
search-bound, not capacity-bound, so a 512-wide trunk learns it ~3–4× cheaper per sample than 1024; widen
to 1024 only once the frontier starts needing the capacity.

```bash
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game cube-davi --net residual --width 512 --blocks 4 \
  --batch 1000 --lr 2e-3 --eps-sync 0.06 \
  --grow-to 1024 --grow-at 400000000 \
  --samples 1000000000 --max-depth 26 --probe-depths 12,14,15,16 \
  --seed 1 --data <grow-data-dir> --hours 12
```

Pick `--grow-to` as an **integer multiple** of `--width` (512→1024 = 2×) so LayerNorm statistics are
preserved exactly at the widen (uniform unit replication; see `ResidualMlp.WidenTo`). Set `--grow-at` to the
plateau point from Experiment 2 (where the narrow net stops climbing). Caveats: Adam moments do **not**
transfer through the widen (a brief re-warm), and the widen's new neurons start as jittered copies that need
training to differentiate — so capability is preserved at the widen but the *added* capacity takes samples to
become useful. Net effect: most of the run is cheap, and you only confirm the wide net helps if the frontier
was capacity-bound (if it was search-bound, widening won't break through — that's the bet).

### Experiment 2 — curriculum-plateau analysis (right-size the next run)

The campaign runs a fixed sample budget, but the curriculum stops deepening once it hits `--max-depth` or
the value stops propagating. Samples spent after that only refine loss on static deep states. After a run:

1. Plot `curriculumDepth` (column 3) vs `iterations` (column 2) in `logs/cube-davi-res.csv`.
2. Find where `curriculumDepth` flattens — that's the plateau sample count (`iterations × batch`).
3. Size the next campaign to `plateau + ~100 M` margin instead of a round 1 B.

If the curriculum plateaus at, say, 300 M, the next run is ~40% shorter for the same end capability.

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

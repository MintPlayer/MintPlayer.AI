# Chess self-play training — CPU / single-GPU / multi-GPU runbook

How to run AlphaZero-style chess self-play (`--game chess`) on the GPU path (M42 conv net + M43 resident
forward + M44 resident trainer + M45 multi-GPU). All commands are run from the repo root.

> **Key point:** you do **not** pick a GPU count. `--gpu` auto-detects **every** CUDA GPU on the machine and shards
> self-play generation across all of them. So the **single-GPU and multi-GPU commands are identical** — the process
> adapts to the hardware it finds. `--gpus` (below) is an *optional override* only when you want to constrain it.
> With no CUDA device, the same command runs on the CPU automatically.

## Prerequisites

- An NVIDIA GPU + the CUDA toolkit for the GPU path (any number of GPUs). No GPU → it falls back to the CPU
  accelerator with no flag change.
- Build once: `dotnet build -c Release tools/MintPlayer.AI.ReinforcementLearning.Lab`.

## Single-GPU run (also the default multi-GPU run)

`--gpu` uses the one GPU it finds:

```bash
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game chess --arch conv --gpu \
  --filters 128 --blocks 10 \
  --leaf-batch 256 --parallel \
  --games 32 --sims 128 --max-plies 200 \
  --window 200000 --batch 512 --epochs 1 \
  --lr 1e-3 --seed 1 \
  --data data/chess-gpu --hours 8
```

## Multi-GPU run

**Exactly the same command** — `--gpu` auto-detects and uses *all* CUDA GPUs (one device-resident forward per GPU;
generation is sharded by game index; training runs on GPU 0 and the trained weights fan out to every GPU each chunk):

```bash
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game chess --arch conv --gpu \
  --filters 128 --blocks 10 \
  --leaf-batch 256 --parallel \
  --games 64 --sims 128 --max-plies 200 \
  --window 500000 --batch 1024 --epochs 1 \
  --lr 1e-3 --seed 1 \
  --data data/chess-gpu --hours 48
```

On a multi-GPU box, raise `--games` (more concurrent games spread across the devices) and `--window`/`--batch` to keep
each GPU fed. To **constrain** which GPUs are used, add `--gpus`:

| `--gpus` value | effect |
|---|---|
| *(omitted)* / `all` | use every detected CUDA GPU (default) |
| `2` | use the first 2 GPUs |
| `0,2` | use GPUs with those ordinals |
| `1` | force a single GPU on a multi-GPU box |

```bash
# e.g. use only GPUs 0 and 1 of a larger box
… --game chess --arch conv --gpu --gpus 0,1 …
```

## CPU-only run

Omit `--gpu` (this is also the only bitwise-reproducible mode — the DOP-invariant checkpoint is CPU-only):

```bash
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game chess --arch conv --parallel \
  --filters 64 --blocks 6 --games 16 --sims 64 --max-plies 200 \
  --seed 1 --data data/chess-cpu --hours 4
```

## Quick smoke test (a few minutes, verifies the pipeline)

```bash
dotnet run -c Release --project tools/MintPlayer.AI.ReinforcementLearning.Lab -- \
  --game chess --arch conv --gpu --parallel \
  --filters 64 --blocks 6 --leaf-batch 128 --games 6 --sims 48 --max-plies 60 \
  --data data/chess-smoke --hours 0.1
```

## Resume

Point `--data` at the same directory — the net (`chess.az.ckpt`) and optimizer (`chess.az-adam.ckpt`) resume
automatically. Note: on a `--gpu` resume the Adam moments re-warm from zero (weights resume fine) — see
[`GPU_RESIDENT_CONV_TRAINER_PRD.md`](../../docs/prd/GPU_RESIDENT_CONV_TRAINER_PRD.md) P.2.

## Growing the website's difficulty ladder

Add `--ladder` to promote increasingly-strong tiers straight into the web app's models dir as they beat the last
champion:

```bash
… --game chess --arch conv --gpu --ladder --difficulty-dir src/RLDemo.Web/wwwroot/models …
```

## Flags that matter for a run

| Flag | Default | Notes |
|---|---|---|
| `--arch conv` | `mlp` | AlphaZero conv-residual tower (the M42 plateau fix). Use `conv` for GPU runs. |
| `--gpu` | off | Opt into the GPU path; **auto-uses all detected CUDA GPUs**. Omit for CPU. |
| `--gpus <spec>` | `all` | Optional override: `all` / a count / ordinals (`0,2`). Only with `--gpu`. |
| `--filters` / `--blocks` | `64` / `6` | Conv tower size = net **capacity** (the real strength lever). Bigger also raises GPU utilization and uses idle VRAM. |
| `--leaf-batch` | `1` | Leaves per net forward (virtual-loss batched MCTS). **>1 is required to use a GPU**; 128–512 is typical. |
| `--parallel` / `--dop` | off / cores−2 | Fan self-play generation across CPU cores. |
| `--games` | `8` | Games per chunk (raise on multi-GPU / many cores). |
| `--sims` | `64` | MCTS simulations per move. |
| `--max-plies` | `200` | Ply cap per game (a chunk's wall-time is bounded by its slowest game). |
| `--window` | `40000` | Replay capacity. Large windows → the training step dominates each chunk (why M44's resident trainer matters). |
| `--batch` / `--epochs` | `128` / `1` | Training minibatch and passes over the window per chunk. |
| `--lr` / `--seed` / `--hours` | `1e-3` / `1` / `1` | Learning rate, RNG seed, wall-clock budget. |
| `--material-weight` / `--value-weight` / `--clip` | `0.5` / `1` / `5` | Value shaping / value-loss weight / grad-norm clip. |
| `--data` | `data` | Model store + logs directory (also the resume source). |

## Verifying GPU usage

- **Real compute utilization:** `nvidia-smi` — its `GPU-Util %` is the authoritative meter. In Windows **Task Manager**,
  select the **Compute_0** engine in a GPU graph dropdown, **not "3D"** (CUDA work doesn't show on the 3D engine).
- **Generation-vs-training split per chunk:** set `CHESS_CHUNK_TIMING=1` before the command — each chunk logs
  `gen … | train …` wall-time. (A chunk is normally generation-bound; the resident trainer keeps the train step ~120 ms
  per 128-batch on an RTX 3060.)
- Utilization is **spiky and well under 100%** by nature: single-process MCTS interleaves GPU leaf-batch forwards with
  CPU tree search, so the GPU idles between waves. Raise it with a bigger net (`--filters/--blocks`), a bigger
  `--leaf-batch`, and more `--games`/`--parallel`.

## Scope & honest caveats

- **Single machine only.** Multi-GPU means all GPUs *on this box*. Cross-machine / cluster training (actor–learner over a
  network) is **not** built — the model store is local-filesystem and one campaign runs per process. See
  [`MULTI_GPU_SELFPLAY_PRD.md`](../../docs/prd/MULTI_GPU_SELFPLAY_PRD.md) §6.
- **Speed ≠ strength.** The GPU path makes the pipeline fast and scale-ready; it does not make a laptop-scale run reach
  engine strength (from-scratch RL chess needs orders of magnitude more compute). The value here is a working,
  GPU-accelerated, cluster-ready showcase of the MintPlayer.AI SDK.

See also: [`docs/prd/RESIDUAL_CONV_NET_PRD.md`](../../docs/prd/RESIDUAL_CONV_NET_PRD.md) (M42),
[`GPU_RESIDENT_CONV_PRD.md`](../../docs/prd/GPU_RESIDENT_CONV_PRD.md) (M43),
[`GPU_RESIDENT_CONV_TRAINER_PRD.md`](../../docs/prd/GPU_RESIDENT_CONV_TRAINER_PRD.md) (M44),
[`MULTI_GPU_SELFPLAY_PRD.md`](../../docs/prd/MULTI_GPU_SELFPLAY_PRD.md) (M45), and the cube runbook
[`CUBE_CAMPAIGN.md`](CUBE_CAMPAIGN.md).

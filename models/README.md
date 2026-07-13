# Pre-trained models (seed checkpoints)

Committed so the trained AI is part of the repository and never lost — every fresh
clone, data directory or Docker volume seeds itself from here on first start
(`SeedModelsDirectory` in configuration) instead of training from scratch.
The running application's live store stays in `data/` (gitignored); these are the
shipped snapshots.

| File | Model | Provenance |
|---|---|---|
| `rushhour.policy.ckpt` | Imitation policy/value net (the strongest Rush Hour solver) | Trained 2026-06-10→11 by the Lab: 405 M oracle-labeled states from ~745k generated puzzles, policy accuracy 92.3%. With policy-guided A* it solves every official ThinkFun card tested **optimally**, incl. expert card 40 (81 moves, ~2.6k node expansions). |
| `rushhour.policy-adam.ckpt` | Adam optimizer state for the net above | Lets the Lab resume the training campaign mid-stride. Not used for inference. |
| `rushhour.dqn.ckpt` | Masked Double DQN (fallback solver) | Trained on 3,000 generated puzzles (optimal 2–20), eval return 92.85 at 480k steps. |
| `2048.ntuple.ckpt` | Afterstate TD(0) n-tuple network | 100k self-play games (~3 min); reaches the 2048 tile in ~84% of games. |
| `cube.dqn.ckpt` | Rubik's Cube masked Double DQN (M14, fallback) | Trained 2026-06-12: 600k steps (~65 min) on quarter-turn scrambles d ~ U[1..6] with the no-undo action mask, eval return ~70. Gate: **600/600** depth-1–6 scrambles solved within 20 moves (greedy alone 77.8%; the rest via Q-guided lookahead, `aiMode: search`). |
| `cube.policy.ckpt` | Kociemba-imitation policy/value net (M16 — the preferred cube AI) | Trained 2026-06-12 by `Lab --game cube`, extended overnight 2026-06-12→13 by two more resumed stints (~236 M cumulative labeled states, action accuracy 79.8%). Gate: **97/100** depth-1–10 scrambles solved within 40 quarter-turns; greedy alone improved 54% → **64%** over the overnight run (rest via value-guided A*, `aiMode: search`). |
| `cube.policy-adam.ckpt` | Adam optimizer state for the net above | Lets `Lab --game cube` resume the campaign mid-stride. Not used for inference. |

To retrain or continue training the imitation nets (both campaigns resume from the
checkpoints in `--data`; point it here to refresh the shipped seeds directly, or at
`src/RLDemo.Web/data` so the running playground picks improvements up live):

```
dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release -- --hours N --data models                 # Rush Hour
dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release -- --game cube --hours N --data models     # Rubik's Cube
```

The DQN fallbacks retrain from the console:
`dotnet run --project src/RLDemo.Console -c Release -- rushhour|cube --save --data models`.

## Chess — AlphaZero self-play, conv residual net (M42.3, experimental — not yet a shipped seed)

The shipped browser chess net (`src/RLDemo.Web/wwwroot/models/chess.az.d1.ckpt`) is still the flat-MLP
baseline. M42 replaces it with an AlphaZero-style **convolutional residual tower** over the `[18,8,8]`
board (`--arch conv`); see `docs/prd/RESIDUAL_CONV_NET_PRD.md`. The conv net is being trained offline and
is **not yet promoted to the browser** — that's gated on M42.4 (a conv forward in the `.pg` twin), which is
itself gated on this training beating the MLP baseline.

Run the conv training (writes to a scratch `--data` dir and a scratch ladder `--difficulty-dir`, **not**
`wwwroot/models`, so it can't clobber the shipped MLP tiers the `/chess` page serves):

```
dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release -- \
  --game chess --arch conv --filters 64 --blocks 6 \
  --sims 64 --games 16 --max-plies 100 --eval-games 8 --arena-games 12 \
  --parallel --ladder --material-weight 0.5 \
  --data data/chess-conv --difficulty-dir data/chess-conv-ladder --hours N
```

**Why these values (learned the hard way — the conv net is far heavier per MCTS node than the MLP):**
- **Everything must be parallel.** `--parallel` fans self-play across cores; and as of commit `71fe44c` the
  **eval + ladder arena** are parallel too. They run on the owner thread *between* training chunks, so at
  conv cost a sequential arena doesn't just report slowly — it **stalls training** (we saw ~0.8 cores for
  ~24 min per cycle before the fix). All of self-play/eval/arena are inference-only and DOP-invariant, so
  trained checkpoints stay bitwise-identical at any core count.
- **`--max-plies` and `--sims` set throughput.** A chunk's wall time is bounded by its *slowest* game, and a
  weak net rarely mates so games otherwise run to the ply cap. 200 plies × 256 sims (the MLP-era defaults)
  made one chunk+eval cycle take 20–30 min; `--max-plies 100 --sims 64` cuts that to ~4–5 min/chunk.
- **`--eval-games` / `--arena-games`** trade eval signal quality for speed; 8 / 12 is a reasonable balance now
  that both loops are parallel.
- **The gate is a *merit* ladder promotion** (Level 2+ on material margin ≥ +0.75 pawns or head-to-head ≥ 60%).
  Level 1 is always an automatic baseline — not evidence of anything. Early signal is healthy: policy loss
  falls off uniform and material margin climbs, unlike the MLP which plateaued at ~random for 500 games.

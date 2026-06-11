# Pre-trained models (seed checkpoints)

Committed so the trained AI is part of the repository and never lost — every fresh
clone, data directory or Docker volume seeds itself from here on first start
(`SeedModelsDirectory` in configuration) instead of training from scratch.
The running application's live store stays in `data/` (gitignored); these are the
shipped snapshots.

| File | Model | Provenance |
|---|---|---|
| `rushhour.policy.ckpt` | Imitation policy/value net (the strongest Rush Hour solver) | Trained 2026-06-10→11 by `tools/RL.NET.Lab`: 405 M oracle-labeled states from ~745k generated puzzles, policy accuracy 92.3%. With policy-guided A* it solves every official ThinkFun card tested **optimally**, incl. expert card 40 (81 moves, ~2.6k node expansions). |
| `rushhour.policy-adam.ckpt` | Adam optimizer state for the net above | Lets `RL.NET.Lab` resume the training campaign mid-stride. Not used for inference. |
| `rushhour.dqn.ckpt` | Masked Double DQN (fallback solver) | Trained on 3,000 generated puzzles (optimal 2–20), eval return 92.85 at 480k steps. |
| `2048.ntuple.ckpt` | Afterstate TD(0) n-tuple network | 100k self-play games (~3 min); reaches the 2048 tile in ~84% of games. |

To retrain or continue training Rush Hour:
`dotnet run --project tools/RL.NET.Lab -c Release -- --hours N --data src/RL.NET.Web/data`
(then copy the improved `.ckpt` files here to ship them).

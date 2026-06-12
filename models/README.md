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
| `cube.policy.ckpt` | Kociemba-imitation policy/value net (M16 — the preferred cube AI) | Trained 2026-06-12 by `Lab --game cube`: 2 h (two resumed 1 h runs), 7.7 M states labeled by ~370k Kociemba solves, action accuracy 73.5%. Gate: **96/100** depth-1–10 scrambles solved within 40 quarter-turns (greedy alone 54%; rest via value-guided A*, `aiMode: search`). |
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

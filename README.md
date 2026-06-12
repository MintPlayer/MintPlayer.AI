# MintPlayer.AI.ReinforcementLearning

A reinforcement-learning library written from scratch in C#/.NET — no Python, no libtorch,
no native dependencies. See [docs/PRD.md](docs/PRD.md) for the why and what, and
[docs/PLAN.md](docs/PLAN.md) for the milestone roadmap.

## Layout

| Project | Contents |
|---|---|
| `src/MintPlayer.AI.ReinforcementLearning.Core` | Environment API (Gymnasium-faithful), spaces, seeded RNG, agents, trainers, solvers, checkpoints + model store |
| `src/MintPlayer.AI.ReinforcementLearning.Environments` | GridWorld, FrozenLake, CartPole, 2048, Rush Hour (incl. BFS solver + puzzle generator), Rubik's Cube (incl. a C# port of Kociemba's two-phase solver) |
| `src/RLDemo.Console` | Console demo — watch agents learn and play (`--save`/`--load` persist trained models) |
| `src/RLDemo.Web` | **MintPlayer.AI.ReinforcementLearning Playground** — ASP.NET Core + Angular web app: three games (Rush Hour, classic-feel 2048, a 3D Rubik's Cube), each playable yourself and solvable by the trained AI with step-through playback, plus a public gallery of every submitted board |
| `tests/MintPlayer.AI.ReinforcementLearning.Tests` | xUnit suite incl. solved-threshold gates, determinism tests and web API integration tests |
| `tools/MintPlayer.AI.ReinforcementLearning.Lab` | Long-running imitation-learning campaigns (Rush Hour from the BFS oracle, Rubik's Cube from Kociemba) — resumable, checkpointing into the model store |

## Run the playground

```
dotnet run --project src/RLDemo.Web
```

Open the printed URL (default `http://localhost:5210`). In Development the host spawns and
proxies the Angular dev server itself — don't run `ng serve` separately. A fresh `data/`
directory seeds itself from the shipped checkpoints in `models/`, so all three games'
AIs are ready immediately; with an empty seed the host trains at startup instead (page
banners show live progress). The strongest Rush Hour and Rubik's Cube solvers (the
imitation policy nets) are trained with `tools/MintPlayer.AI.ReinforcementLearning.Lab`
(see "Train the models" below).

### Docker

```
docker compose -f docker-compose.local.yml up
```

Open `http://localhost:8080`. Models and the public gallery persist on the `rlnet-data`
volume across restarts and upgrades; a fresh volume seeds itself from the shipped
pre-trained checkpoints in `models/`, so the playground is instantly ready.

The root `docker-compose.yml` is the **deployment** variant (Traefik VPS convention):
it pulls the GHCR image and routes `ai.mintplayer.com` through the external `web`
network with Let's Encrypt TLS.

Every push to `master` also publishes the image to GHCR
(`ghcr.io/mintplayer/mintplayer.ai/playground:master`), so running
it without cloning is:

```
docker run -p 8080:8080 -v rlnet-data:/data ghcr.io/mintplayer/mintplayer.ai/playground:master
```

## Run the demo

```
dotnet run --project src/RLDemo.Console -c Release                 # everything, seed 42
dotnet run --project src/RLDemo.Console -c Release -- cartpole     # just the DQN flagship
dotnet run --project src/RLDemo.Console -c Release -- 2048         # n-tuple TD plays 2048
dotnet run --project src/RLDemo.Console -c Release -- grid lake 7  # tabular envs, seed 7
```

Demos (each ends with animated console playback):

- **GridWorld / FrozenLake** — tabular Q-learning, verified exactly against value
  iteration; FrozenLake is Gymnasium-comparable (≥70% success).
- **CartPole-v1** (`cartpole` = Double DQN, `ppo` = PPO over 8 vectorized envs) —
  faithful port, bit-for-bit match against recorded Gymnasium trajectories;
  solved = mean return ≥ 475/500. Both solve in seconds.
- **2048** (`2048` = afterstate TD(0) n-tuple network, `2048dqn` = generic masked
  Double DQN) — reaches the 2048 tile in ~84% of games after ~3 minutes of self-play.
- **Rush Hour** (`rushhour`) — masked Double DQN on a generated 30-puzzle easy set with
  a BFS oracle; solves 100% within 2× optimal after ~1 minute of training.
- **Rubik's Cube** (`cube`) — masked Double DQN on shallow quarter-turn scrambles
  (depths 1–6); with Q-guided lookahead it solves the whole band (600/600) after
  ~65 minutes of training. The Kociemba port doubles as the always-available
  algorithmic solver and the imitation oracle.

The playground's strongest solvers go further with **imitation learning + net-guided
search** (`tools/MintPlayer.AI.ReinforcementLearning.Lab`):

- **Rush Hour** — imitation from the BFS oracle + policy-guided A\*: after an overnight
  run (224M labeled states, pure managed .NET) it solves every official ThinkFun card we
  tested **optimally**, including expert card 40 (81 moves) in ~2,500 node expansions:

![Card 40 solved optimally by the AI](docs/screenshots/card40-ai-solved.png)

- **Rubik's Cube** — imitation from Kociemba solutions + value-guided A\*: after a 2-hour
  campaign (7.7M labeled states) it solves 96% of depth-1–10 scrambles within 40
  quarter-turns; deeper scrambles fail honestly and the Kociemba button always answers.

## Train the models

The web host trains its fallback models automatically when the store is empty. The
imitation policy nets are trained (and resumed — net + Adam state checkpoint to the
model store every eval) by Lab campaigns:

```
dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release -- --hours N --data src/RLDemo.Web/data           # Rush Hour
dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release -- --game cube --hours N --data models            # Rubik's Cube
dotnet run --project tools/MintPlayer.AI.ReinforcementLearning.Lab -c Release -- --game cube --eval-only --data models          # cube gate report
```

Point `--data` at `src/RLDemo.Web/data` to have the running playground pick up improving
checkpoints live (it re-reads them every few minutes), or at `models/` to refresh the
committed seeds. The DQN fallbacks can be retrained from the console:
`dotnet run --project src/RLDemo.Console -c Release -- rushhour|cube --save --data models`.

## Run the tests

```
dotnet test
```

Statistical solve-threshold tests carry `[Trait("Category", "Slow")]`; filter with
`dotnet test --filter "Category!=Slow"` for the fast loop.

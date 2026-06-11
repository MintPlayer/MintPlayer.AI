# MintPlayer.AI.ReinforcementLearning

A reinforcement-learning library written from scratch in C#/.NET — no Python, no libtorch,
no native dependencies. See [docs/PRD.md](docs/PRD.md) for the why and what, and
[docs/PLAN.md](docs/PLAN.md) for the milestone roadmap.

## Layout

| Project | Contents |
|---|---|
| `src/MintPlayer.AI.ReinforcementLearning.Core` | Environment API (Gymnasium-faithful), spaces, seeded RNG, agents, trainers, solvers, checkpoints + model store |
| `src/MintPlayer.AI.ReinforcementLearning.Environments` | GridWorld, FrozenLake, CartPole, 2048, Rush Hour (incl. BFS solver + puzzle generator) |
| `src/RLDemo.Console` | Console demo — watch agents learn and play (`--save`/`--load` persist trained models) |
| `src/RLDemo.Web` | **MintPlayer.AI.ReinforcementLearning Playground** — ASP.NET Core + Angular web app: draw a Rush Hour puzzle on a canvas, play it yourself, then watch the trained DQN solve it with back/forward playback |
| `tests/MintPlayer.AI.ReinforcementLearning.Tests` | xUnit suite incl. solved-threshold gates, determinism tests and web API integration tests |

## Run the playground

```
dotnet run --project src/RLDemo.Web
```

Open the printed URL (default `http://localhost:5210`). In Development the host spawns and
proxies the Angular dev server itself — don't run `ng serve` separately. On first start it
trains its models (2048 ≈ 3 min; Rush Hour DQN fallback ≈ 30 min — page banners show live
progress) and saves them under `data/`; later starts load instantly. The strongest Rush
Hour solver (the imitation policy net) is trained with `tools/MintPlayer.AI.ReinforcementLearning.Lab`.

### Docker

```
docker compose -f docker-compose.local.yml up
```

Open `http://localhost:8080`. Models and the public gallery persist on the `rlnet-data`
volume across restarts and upgrades; a fresh volume seeds itself from the shipped
pre-trained checkpoints in `models/`, so the playground is instantly ready.

The root `docker-compose.yml` is the **deployment** variant (Traefik VPS convention):
it pulls the GHCR image and routes `rl.mintplayer.com` through the external `web`
network with Let's Encrypt TLS.

Every push to `master` also publishes the image to GHCR
(`ghcr.io/mintplayer/mintplayer.ai.reinforcementlearning/playground:master`), so running
it without cloning is:

```
docker run -p 8080:8080 -v rlnet-data:/data ghcr.io/mintplayer/mintplayer.ai.reinforcementlearning/playground:master
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

The playground's strongest Rush Hour solver goes further: **imitation learning from the
BFS oracle + policy-guided A\*** (`tools/MintPlayer.AI.ReinforcementLearning.Lab`) — after an overnight self-supervised
run (224M labeled states, pure managed .NET) it solves every official ThinkFun card we
tested **optimally**, including expert card 40 (81 moves) in ~2,500 node expansions:

![Card 40 solved optimally by the AI](docs/screenshots/card40-ai-solved.png)

## Run the tests

```
dotnet test
```

Statistical solve-threshold tests carry `[Trait("Category", "Slow")]`; filter with
`dotnet test --filter "Category!=Slow"` for the fast loop.

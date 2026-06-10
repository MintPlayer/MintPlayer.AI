# RL.NET

A reinforcement-learning library written from scratch in C#/.NET — no Python, no libtorch,
no native dependencies. See [docs/PRD.md](docs/PRD.md) for the why and what, and
[docs/PLAN.md](docs/PLAN.md) for the milestone roadmap.

## Layout

| Project | Contents |
|---|---|
| `src/RL.NET.Core` | Environment API (Gymnasium-faithful), spaces, seeded RNG, agents, trainers, solvers |
| `src/RL.NET.Environments` | GridWorld, FrozenLake (more coming: CartPole, 2048, Rush Hour) |
| `src/RL.NET.Demo` | Console demo — watch agents learn and play |
| `tests/RL.NET.Tests` | xUnit suite incl. solved-threshold gates and determinism tests |

## Run the demo

```
dotnet run --project src/RL.NET.Demo -c Release                 # everything, seed 42
dotnet run --project src/RL.NET.Demo -c Release -- cartpole     # just the DQN flagship
dotnet run --project src/RL.NET.Demo -c Release -- 2048         # n-tuple TD plays 2048
dotnet run --project src/RL.NET.Demo -c Release -- grid lake 7  # tabular envs, seed 7
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

## Run the tests

```
dotnet test
```

Statistical solve-threshold tests carry `[Trait("Category", "Slow")]`; filter with
`dotnet test --filter "Category!=Slow"` for the fast loop.

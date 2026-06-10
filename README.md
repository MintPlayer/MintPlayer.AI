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
dotnet run --project src/RL.NET.Demo -c Release -- grid lake 7  # tabular envs, seed 7
```

Demos: tabular Q-learning on GridWorld (verified exactly against value iteration) and
slippery FrozenLake (Gymnasium-comparable, ≥70% success), and a from-scratch Double DQN
solving a faithful CartPole-v1 port (bit-for-bit match against recorded Gymnasium
trajectories; solved = mean return ≥ 475/500). Each ends with animated console playback.

## Run the tests

```
dotnet test
```

Statistical solve-threshold tests carry `[Trait("Category", "Slow")]`; filter with
`dotnet test --filter "Category!=Slow"` for the fast loop.

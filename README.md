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
dotnet run --project src/RL.NET.Demo            # default seed 42
dotnet run --project src/RL.NET.Demo -- 1234    # custom master seed
```

Trains tabular Q-learning on GridWorld (verified exactly against value iteration) and on
slippery FrozenLake (Gymnasium-comparable, solved threshold 70% success), then prints
policy maps and animates greedy playback in the console.

## Run the tests

```
dotnet test
```

Statistical solve-threshold tests carry `[Trait("Category", "Slow")]`; filter with
`dotnet test --filter "Category!=Slow"` for the fast loop.

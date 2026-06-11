# MintPlayer.AI.ReinforcementLearning.Core

Reinforcement learning for .NET, written **from scratch in C#** — no Python, no
libtorch, no native dependencies. Tensors, SIMD matmul, tape-based autograd, Adam,
Gymnasium-faithful environment contracts, and an algorithm ladder from tabular
Q-learning through Double DQN and PPO — all readable managed code, verified against
golden trajectories from Python Gymnasium and finite-difference gradient checks.

```csharp
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

// Implement IEnvironment<TObs, TAct> for your own problem (Gymnasium-style
// Reset/Step with the terminated/truncated split), then:
var result = DqnTrainer.Train(env, new DqnOptions { SolveThreshold = 475 }, new SeedSequence(42));
Console.WriteLine($"solved after {result.StepsTrained} steps, eval {result.FinalEvalReturn:F1}");
```

Highlights:

- `IEnvironment<TObs, TAct>` with separate `Terminated`/`Truncated` (the most common
  silent RL bug, made impossible to conflate), spaces, action masking, vectorized envs
  whose parallel mode is **bitwise identical** to sequential.
- Trainers: tabular Q-learning/SARSA, REINFORCE, Double DQN (masked, with full-resume
  checkpointing — an interrupted run resumes *bitwise identically*), PPO with GAE.
- Reproducibility as a feature: own xoshiro256** RNG + master-seed fan-out; one seed
  reproduces an entire training run.
- Versioned binary checkpoints + a file-based model store (atomic writes).

Pair with **MintPlayer.AI.ReinforcementLearning.Environments** for ready-made
environments (CartPole, GridWorld, FrozenLake, 2048, Rush Hour) and the imitation
learning + policy-guided A\* toolkit that solves expert Rush Hour boards optimally.

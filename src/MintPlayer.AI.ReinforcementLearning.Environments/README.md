# MintPlayer.AI.ReinforcementLearning.Environments

Ready-made environments for **MintPlayer.AI.ReinforcementLearning** — pure managed C#:

- **CartPole-v1** — a faithful Gymnasium port, validated bit-for-bit against recorded
  golden trajectories. Double DQN solves it in seconds.
- **GridWorld / FrozenLake** — tabular classics with a value-iteration oracle.
- **2048** — full board mechanics + an afterstate TD(0) n-tuple learner that reaches
  the 2048 tile in ~84% of games after ~3 minutes of self-play.
- **Rush Hour** — 6×6 sliding-block puzzle with masked 32-action space, a BFS optimal
  solver, a seeded puzzle generator, and an **imitation-learning toolkit**:
  `RushHourOracle` labels every reachable state of a configuration with its exact
  distance-to-goal, `RushHourPolicyNet` trains on those labels, and
  `RushHourPolicySearch` runs policy-guided A\* — solving official expert boards
  (ThinkFun card 40: 81 moves) **optimally** in ~2,500 node expansions.

```csharp
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments;

var env = new CartPoleEnv();
var result = DqnTrainer.Train(env, new DqnOptions { SolveThreshold = 475 }, new SeedSequence(42));
Console.WriteLine($"solved after {result.StepsTrained} steps, eval {result.FinalEvalReturn:F1}");
```

using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RushHour;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class RushHourGateTests
{
    [Fact]
    [Trait("Category", "Slow")]
    public void Gate_MaskedDqn_Solves90PercentOfEasySet_Within2xOptimal()
    {
        // PRD §6 / PLAN M6 gate. Reference run: 100% in 40k steps with the sparse
        // −1/+100 reward (no shaping); budget below leaves generous margin.
        var puzzles = RushHourGenerator.Generate(seed: 99, count: 30, minOptimal: 4, maxOptimal: 10);
        var seeds = new SeedSequence(42);
        var env = new RushHourEnv(puzzles, maxMoves: 60);

        var result = DqnTrainer.Train(env, new DqnOptions
        {
            Hidden = [128, 128],
            Gamma = 0.98,
            LearningRate = 5e-4f,
            MaxSteps = 150_000,
            BufferCapacity = 100_000,
            Epsilon = new LinearSchedule(1.0, 0.05, 60_000),
            EvalEvery = 10_000,
            EvalEpisodes = 20,
            SolveThreshold = 88,
        }, seeds);

        int solvedInBudget = 0;
        for (int i = 0; i < puzzles.Count; i++)
        {
            env.FixedPuzzleIndex = i;
            env.Reset(1);
            var obs = env.CurrentObservation();
            while (true)
            {
                var step = env.Step(result.Agent.Act(obs, env.CurrentActionMask(), greedy: true));
                obs = step.Observation;
                if (step.Terminated)
                {
                    if (env.MovesUsed <= 2 * puzzles[i].OptimalMoves) solvedInBudget++;
                    break;
                }
                if (step.Truncated) break;
            }
        }

        Assert.True(solvedInBudget >= 27,
            $"solved {solvedInBudget}/30 within 2x optimal (need >= 27 for the 90% gate)");
    }
}

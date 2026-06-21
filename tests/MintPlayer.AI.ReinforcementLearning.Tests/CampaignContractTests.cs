extern alias Lab; // the Lab exe's campaigns; aliased so its generated `Program` doesn't clash with RLDemo.Web's
using SnakeDqnCampaign = Lab::SnakeDqnCampaign;
using RushHourImitationCampaign = Lab::RushHourImitationCampaign;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Contract tests for the real <c>ITrainingCampaign</c> implementations (PLAN M25): that a campaign trains,
/// checkpoints to the expected store ids, and — the property the whole harness exists for — <b>resumes</b> in a
/// fresh instance rather than restarting. The runner's loop/cadence is covered separately by
/// <see cref="CampaignRunnerTests"/> with a fake campaign; these exercise the live campaigns end-to-end.
/// Marked Slow (they train a little + touch disk). cube-davi / cube-policy aren't here — they stand up the GPU
/// stack and are covered by the Lab's `--eval-only` smokes; cube-imitation needs the multi-second Kociemba warmup.
/// </summary>
public class CampaignContractTests
{
    [Fact]
    [Trait("Category", "Slow")]
    public void SnakeCampaign_Trains_Checkpoints_AndResumesFromState()
    {
        var dir = Directory.CreateTempSubdirectory("snake-campaign-contract");
        try
        {
            var store = new FileModelStore(dir.FullName);
            // Small grid + tiny step budget: this asserts the CONTRACT (advance/checkpoint/resume), not learning.
            // chunk 1500 → first chunk lands at 1500 (< target), the second continues to the 3000 cap (IsComplete).
            SnakeDqnCampaign Fresh() => new(seed: 1, trainGrid: 5, evalGrid: 6, chunkSteps: 1500, targetSteps: 3000, evalEpisodes: 3, learningRate: 5e-4f, epsilonStart: 1.0f);

            var c1 = Fresh();
            Assert.False(c1.Resume(store));            // nothing in the store yet → fresh
            long afterChunk1 = c1.TrainChunk();
            Assert.Equal(1500, afterChunk1);            // advanced by exactly one chunk
            Assert.False(c1.IsComplete);                // 1500 < 3000

            var eval = c1.Evaluate();
            Assert.Contains(eval.Metrics, m => m.Name == "food6");          // eval reports food on the DEPLOY grid
            Assert.All(eval.Metrics, m => Assert.False(double.IsNaN(m.Value)));

            c1.Checkpoint(store);
            c1.Dispose();
            using (var net = store.TryOpenRead("snake", "dqn")) Assert.NotNull(net);          // deployable net (web id)
            using (var state = store.TryOpenRead("snake", "dqn-state")) Assert.NotNull(state); // full resume state

            // A brand-new campaign instance must continue from the checkpoint, not start over.
            var c2 = Fresh();
            Assert.True(c2.Resume(store));              // resumed
            long afterChunk2 = c2.TrainChunk();
            Assert.True(afterChunk2 > afterChunk1, $"resume continued to {afterChunk2}, expected past {afterChunk1}");
            Assert.Equal(3000, afterChunk2);            // reached the cap
            Assert.True(c2.IsComplete);
            c2.Dispose();
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    [Trait("Category", "Slow")]
    public void RushHourCampaign_Checkpoints_AndResumes()
    {
        var dir = Directory.CreateTempSubdirectory("rushhour-campaign-contract");
        try
        {
            var store = new FileModelStore(dir.FullName);
            var c1 = new RushHourImitationCampaign(seed: 1, learningRate: 3e-4f);
            Assert.False(c1.Resume(store));            // fresh
            long samples = c1.TrainChunk();            // one BFS-oracle config + supervised batches
            Assert.True(samples > 0, "a training chunk should process samples");
            c1.Checkpoint(store);                      // (Evaluate is heavy — its parity is the Lab eval-only smoke)
            c1.Dispose();
            using (var net = store.TryOpenRead("rushhour", "policy")) Assert.NotNull(net);
            using (var adam = store.TryOpenRead("rushhour", "policy-adam")) Assert.NotNull(adam);

            var c2 = new RushHourImitationCampaign(seed: 1, learningRate: 3e-4f);
            Assert.True(c2.Resume(store));             // the saved net is picked up
            c2.Dispose();
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }
}

using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M69/B6 — what a campaign does when the bytes under its net id are NOT a net it can load.
/// <para>
/// The rule these pin, in one line: <b>an unreadable checkpoint must degrade to "start fresh", never crash a
/// long run.</b> A checkpoint is written every few minutes of an overnight run, so it is exactly the file most
/// likely to be found truncated after a kill, a full disk, or a shape change that was shipped between two runs.
/// The campaigns handle this with a <c>catch (InvalidDataException)</c> that logs and carries on — and a
/// swallowed exception is untestable by inspection: it either runs or it does not, and nothing in the log of a
/// healthy run tells you which. Losing it turns a recoverable bad file into a crash at the top of the run.
/// </para>
/// <para>
/// The garbage written here is 256 bytes, deliberately far more than a header: <c>CheckpointFormat.ReadHeader</c>
/// raises <c>InvalidDataException</c> on a bad magic number, whereas a file too short to hold one would raise
/// <c>EndOfStreamException</c> — a different, UNCAUGHT failure. These tests are about the caught arm, so they
/// feed it a file the reader gets far enough into to reject properly.
/// </para>
/// <para>
/// The last test covers the other shape of "incomplete store": a perfectly good net with no progress sidecar
/// beside it, which is every Rush Hour checkpoint written before M58. It must resume the net and restart the
/// counters, not refuse the run. (Unreadable <i>sidecars</i> are covered a level down, on the state readers
/// themselves, by <see cref="CampaignStateSerializationTests"/> and the expert-iteration lifecycle tests.)
/// </para>
/// </summary>
public class CampaignResumeErrorTests
{
    /// <summary>Writes bytes that are long enough to be parsed as a checkpoint and are not one.</summary>
    private static void WriteGarbage(IModelStore store, string environmentId, string algorithmId)
    {
        var bytes = new byte[256];
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)(i * 7 + 13);
        store.Save(environmentId, algorithmId, s => s.Write(bytes, 0, bytes.Length));
    }

    private static readonly BlockDudeIds.NetIds Phase1 = BlockDudeIds.ForPhase(1);
    private static readonly BlockDudeIds.NetIds Phase2 = BlockDudeIds.ForPhase(2);

    [Fact]
    public void Expert_iteration_treats_an_unreadable_phase_two_net_as_no_net_and_repairs_the_store()
    {
        var dir = Directory.CreateTempSubdirectory("m69-xit-corrupt");
        try
        {
            var store = new FileModelStore(dir.FullName);
            WriteGarbage(store, BlockDudeIds.Environment, Phase2.Policy);

            using (var c = new BlockDudeExpertIterationCampaign(new BlockDudeExpertIterationOptions
            {
                Seed = 1,
                WarmStart = false,   // isolate the stale-net arm from the warm-start arm below
            }))
            {
                // False, not an exception: the file was there but unusable, which is the same situation as no
                // file at all. The campaign is fully initialised afterwards...
                Assert.False(c.Resume(store));

                // ...and its next checkpoint must OVERWRITE the bad file rather than leave the run permanently
                // unable to resume itself.
                c.Checkpoint(store);
            }

            using var second = new BlockDudeExpertIterationCampaign(new BlockDudeExpertIterationOptions
            {
                Seed = 1,
                WarmStart = false,
            });
            Assert.True(second.Resume(store), "the checkpoint written after the corrupt one must be loadable");
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Expert_iteration_survives_a_phase_one_net_that_cannot_be_warm_started_from()
    {
        // The second catch arm, on a different file: no phase-2 net at all, and the phase-1 BEST net that warm
        // start reaches for is unreadable. This is the realistic shape — phase 1 shipped a net, the observation
        // encoding changed, phase 2 starts anyway — and it must fall through to random weights, not throw.
        var dir = Directory.CreateTempSubdirectory("m69-xit-warmstart");
        try
        {
            var store = new FileModelStore(dir.FullName);
            WriteGarbage(store, BlockDudeIds.Environment, Phase1.PolicyBest);

            using var c = new BlockDudeExpertIterationCampaign(new BlockDudeExpertIterationOptions
            {
                Seed = 1,
                WarmStart = true,
            });

            Assert.False(c.Resume(store));

            // Still usable: a warm start that failed is a slower run, not a broken one.
            c.Checkpoint(store);
            using var saved = store.TryOpenRead(BlockDudeIds.Environment, Phase2.Policy);
            Assert.NotNull(saved);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Imitation_treats_an_unreadable_net_as_no_net_and_repairs_the_store()
    {
        var dir = Directory.CreateTempSubdirectory("m69-bd-corrupt");
        try
        {
            var store = new FileModelStore(dir.FullName);
            WriteGarbage(store, BlockDudeIds.Environment, Phase1.Policy);

            BlockDudeImitationOptions Options() => new()
            {
                Seed = 1,
                PinStage = 0,
                GateEverySamples = long.MaxValue,   // never gate: gating would solve boards
                BoardsPerRound = 1,
                SamplesPerBoard = 1,
            };

            using (var c = new BlockDudeImitationCampaign(Options()))
            {
                Assert.False(c.Resume(store));
                c.Checkpoint(store);
            }

            using var second = new BlockDudeImitationCampaign(Options());
            Assert.True(second.Resume(store), "the checkpoint written after the corrupt one must be loadable");
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void RushHour_resumes_a_net_that_has_no_progress_sidecar_beside_it()
    {
        // Every Rush Hour checkpoint written before M58 looks like this. The counters restart at zero, which is
        // the pre-M58 behaviour and is documented as such; what must NOT happen is the older store failing to
        // resume, which would silently throw away whatever it had trained.
        var dir = Directory.CreateTempSubdirectory("m69-rh-nosidecar");
        try
        {
            var store = new FileModelStore(dir.FullName);

            using (var first = new RushHourImitationCampaign(new RushHourImitationOptions { Seed = 1 }))
            {
                Assert.False(first.Resume(store));
                first.Checkpoint(store);          // net + adam + sidecar
            }

            Assert.True(store.Delete("rushhour", "policy-progress"));   // make it a pre-M58 store
            Assert.True(store.Exists("rushhour", "policy"));

            using var second = new RushHourImitationCampaign(new RushHourImitationOptions { Seed = 1 });
            Assert.True(second.Resume(store));
        }
        finally { dir.Delete(recursive: true); }
    }
}

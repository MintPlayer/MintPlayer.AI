using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M70 — the growth rung survives a round trip, and a file written before the field existed still loads.
/// </summary>
/// <remarks>
/// The rung was previously recovered by matching the live trunk against the ladder, which cannot tell "rung 0"
/// apart from "an architecture this ladder never described". A net in the second category read as the first and,
/// at a high sample count, was walked to the top of the ladder in one call — silently, because every step is
/// function-preserving, so the run kept training and only the architecture was wrong.
/// <para>
/// Recording it only helps if it round-trips, and the compatibility half matters as much as the write: both
/// formats append the field, so a store written before M70 must still load rather than throwing. <b>-1 and 0 are
/// deliberately different values</b> — -1 means "not recorded" and licenses the legacy shape-matching, 0 means
/// "known to be on the bottom rung" and does not.
/// </para>
/// </remarks>
public class GrowthRungPersistenceTests
{
    // ── the campaign progress sidecar (v2 → v3) ──────────────────────────────────────────────────

    [Fact]
    public void The_rung_survives_a_sidecar_round_trip()
    {
        InStore(store =>
        {
            CampaignProgressState.Save(store, "env", "progress", "test-progress",
                samples: 10, units: 2, lastMetric: 0.5, rung: 3, new Xoshiro256StarStar(1));

            var loaded = CampaignProgressState.TryLoad(store, "env", "progress", "test-progress", rngCount: 1);

            Assert.NotNull(loaded);
            Assert.Equal(3, loaded!.Rung);
            Assert.Equal(10, loaded.Samples);   // the appended field must not disturb what preceded it
            Assert.Equal(2, loaded.Units);
            Assert.Equal(0.5, loaded.LastMetric);
        });
    }

    [Fact]
    public void Rung_zero_round_trips_as_zero_and_not_as_not_recorded()
    {
        // The distinction the whole fix rests on. If 0 were flattened to -1 anywhere, a net genuinely on the
        // bottom rung would re-enter the shape-matching path it was recorded to avoid.
        InStore(store =>
        {
            CampaignProgressState.Save(store, "env", "progress", "test-progress",
                samples: 1, units: 1, lastMetric: 0, rung: 0, new Xoshiro256StarStar(1));

            Assert.Equal(0, CampaignProgressState.TryLoad(store, "env", "progress", "test-progress", rngCount: 1)!.Rung);
        });
    }

    [Fact]
    public void The_overload_without_a_rung_records_not_recorded()
    {
        // Campaigns that do not grow keep calling the short overload; they must not claim rung 0.
        InStore(store =>
        {
            CampaignProgressState.Save(store, "env", "progress", "test-progress",
                samples: 1, units: 1, lastMetric: 0, new Xoshiro256StarStar(1));

            Assert.Equal(-1, CampaignProgressState.TryLoad(store, "env", "progress", "test-progress", rngCount: 1)!.Rung);
        });
    }

    // ── the DQN training state (v4 → v5) ─────────────────────────────────────────────────────────

    [Fact]
    public void The_growth_stage_survives_a_DqnTrainingState_round_trip()
    {
        var state = NewState();
        state.GrowthStage = 4;

        var stream = new MemoryStream();
        state.Save(stream);
        stream.Position = 0;

        Assert.Equal(4, DqnTrainingState.Load(stream).GrowthStage);
    }

    [Fact]
    public void A_state_that_never_grew_round_trips_as_not_recorded()
    {
        var stream = new MemoryStream();
        NewState().Save(stream);
        stream.Position = 0;

        Assert.Equal(-1, DqnTrainingState.Load(stream).GrowthStage);
    }

    [Fact]
    public void The_stage_is_carried_across_WithNetwork()
    {
        // WithNetwork rebuilds the state around a new net, which is exactly what growth does. Dropping the stage
        // there would reset it to -1 on every growth — the one moment it is guaranteed to be known.
        var state = NewState();
        state.GrowthStage = 2;

        var online = (DuelingQNet)state.Online;
        var swapped = state.WithNetwork(online, (DuelingQNet)online.CloneStructure(), state.Optimizer);

        Assert.Equal(2, swapped.GrowthStage);
    }

    private static void InStore(Action<IModelStore> body)
    {
        var dir = Directory.CreateTempSubdirectory("m70-rung");
        try { body(new FileModelStore(dir.FullName)); }
        finally { dir.Delete(recursive: true); }
    }

    private static DqnTrainingState NewState()
    {
        var rng = new Xoshiro256StarStar(3);
        var online = new DuelingQNet(4, [8], 2, rng);
        var target = (DuelingQNet)online.CloneStructure();
        target.CopyFrom(online);
        return new DqnTrainingState
        {
            Online = online,
            Target = target,
            Optimizer = new Adam(online.Parameters(), 1e-3f),
            Buffer = new ReplayBuffer(8, 4, 2),
            PolicyRng = new Xoshiro256StarStar(1),
            BufferRng = new Xoshiro256StarStar(2),
        };
    }
}

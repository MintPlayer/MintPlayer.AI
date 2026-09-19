using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The phase-2 <c>TrainChunk</c> (M69/B2) — the search→collect→train round, and the two fallback tiers beneath
/// A*.
/// </summary>
/// <remarks>
/// <see cref="BlockDudeExpertIterationLifecycleTests"/> deliberately never calls <c>TrainChunk</c>, because at
/// the shipped <c>InitialFrontier = 20</c> it runs an A*/beam oracle over every shipped level. At
/// <c>InitialFrontier = 1</c> the start state is one move from the door, so the same code path terminates in a
/// handful of expansions — <b>no production change was needed</b>, the options record already exposed every
/// knob.
/// <para>
/// What this guards: the fallback ladder is invisible when it breaks. A* failing silently falls through to
/// beam, and beam failing falls through to landmark salvage; if a tier stopped working, the run would simply
/// collect fewer samples and the frontier would stall, which reads exactly like the net having hit its limit.
/// The counters asserted here (<c>beam_hits</c>, <c>landmark_hits</c>) are the only outward sign of which tier
/// actually did the work.
/// </para>
/// </remarks>
public class BlockDudeExpertIterationChunkTests
{
    /// <summary>One attempt per level, one move from the door, no phase-1 net to warm-start from.</summary>
    private static BlockDudeExpertIterationOptions AtFrontierOne() => new()
    {
        Seed = 1,
        WarmStart = false,
        InitialFrontier = 1,
        AttemptsPerLevel = 1,
        SearchSeconds = 2,
        BatchSize = 8,
    };

    private static double Metric(CampaignEval eval, string name) =>
        eval.Metrics.Single(m => m.Name == name).Value;

    [Fact]
    public void A_chunk_one_move_from_the_door_solves_by_A_star_and_trains_on_what_it_found()
    {
        var dir = Directory.CreateTempSubdirectory("m69-xit-chunk");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new BlockDudeExpertIterationCampaign(AtFrontierOne());
            c.Resume(store);

            long samples = c.TrainChunk();
            var eval = c.Evaluate();

            Assert.True(samples > 0, "a solved round must produce training samples");

            // Every level is one move from its door, so the primary tier solves all of them and neither
            // fallback is reached. That is the assertion: not merely "something was solved", but that the
            // CHEAPEST tier did it. A regression that broke A* would still pass a samples>0 check by quietly
            // falling through to beam.
            Assert.Equal(1.0, Metric(eval, "search_rate"));
            Assert.Equal(0, Metric(eval, "beam_hits"));
            Assert.Equal(0, Metric(eval, "landmark_hits"));
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void Solving_everything_moves_every_levels_frontier_outward()
    {
        var dir = Directory.CreateTempSubdirectory("m69-xit-frontier");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new BlockDudeExpertIterationCampaign(AtFrontierOne());
            c.Resume(store);

            Assert.Equal(1, Metric(c.Evaluate(), "frontier_min"));

            c.TrainChunk();

            // solved/attempts = 1.0 clears AdvanceRate (0.75), so NextFrontier applies FrontierGrowth. This is
            // the curriculum's actual progress signal — if it stopped moving, the run would train forever one
            // move from the door and every loss/accuracy number would look excellent while it did.
            Assert.True(Metric(c.Evaluate(), "frontier_min") > 1,
                "clearing AdvanceRate must move the frontier outward");
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void A_starved_node_budget_falls_through_to_the_beam_tier()
    {
        var dir = Directory.CreateTempSubdirectory("m69-xit-beam");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new BlockDudeExpertIterationCampaign(AtFrontierOne() with
            {
                Expansions = 1,      // starve A* so it cannot reach the door
                BeamWidth = 8,
                BeamSeconds = 1,
                BeamSecondsMax = 1,
            });
            c.Resume(store);

            c.TrainChunk();
            var eval = c.Evaluate();

            // Beam costs width × depth, so it reaches a door one move away even on a 1-second budget. The point
            // is the HANDOFF: with A* starved, work that still gets done must show up in beam_hits.
            Assert.True(Metric(eval, "beam_hits") > 0,
                "with A* starved to one expansion, the beam tier must be what solves these");
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public void With_both_search_tiers_disabled_a_round_collects_nothing_and_the_frontier_does_not_advance()
    {
        var dir = Directory.CreateTempSubdirectory("m69-xit-nosolve");
        try
        {
            var store = new FileModelStore(dir.FullName);
            using var c = new BlockDudeExpertIterationCampaign(AtFrontierOne() with
            {
                Expansions = 1,   // A* starved
                BeamWidth = 0,    // beam tier switched off entirely
            });
            c.Resume(store);

            c.TrainChunk();
            var eval = c.Evaluate();

            Assert.Equal(0, Metric(eval, "beam_hits"));

            // The frontier must NOT move outward on a round that solved nothing. At depth 1 the retreat
            // multiplier floors at 1, so "did not advance" is the assertable half — and it is the half that
            // matters, since advancing on an unsolved round would march the curriculum away from the net.
            Assert.Equal(1, Metric(eval, "frontier_min"));
        }
        finally { dir.Delete(recursive: true); }
    }
}

namespace MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

/// <summary>
/// A greedy 1-ply baseline for FruitCake (PRD §4.5). For each of the <see cref="FruitCakeEnv.ColumnCount"/>
/// columns it clones the board, drops the current fruit there, simulates to rest, and scores the outcome —
/// strongly preferring merges, then a lower pile, and hard-avoiding a placement that loses. Surprisingly
/// strong for its cost, and reusable as both the live "Watch AI" policy and a demonstration generator for
/// warm-starting the learned agent.
///
/// <para>Planning runs on rotation-off clones (cheaper, deterministic; merges don't depend on orientation),
/// so it works against either a training or a live (rotation-on) world.</para>
/// </summary>
public sealed class FruitCakeHeuristic
{
    private const double MergeWeight = 1000.0;  // one merge point outweighs any pile-height difference
    private const double LosePenalty = 1e9;     // never choose a placement that ends the game (unless all do)
    private const double CenterBias = 0.001;    // tie-break toward the middle

    /// <summary>Pick the best column (0..ColumnCount-1) to drop <paramref name="currentTier"/> into.</summary>
    public int ChooseColumn(FruitCakeWorld world, int currentTier)
    {
        int columns = FruitCakeEnv.ColumnCount;
        double best = double.NegativeInfinity;
        int bestCol = columns / 2;
        double center = (columns - 1) / 2.0;

        for (int col = 0; col < columns; col++)
        {
            var sim = world.Clone(enableRotation: false);
            sim.SpawnFruit(currentTier, FruitCakeEnv.ColumnX(col, currentTier), FruitCakeEnv.HeldY(currentTier));
            int points = sim.SettleAfterDrop(FruitCakeEnv.SettleSpeedPx, FruitCakeEnv.MinSettleSubsteps, FruitCakeEnv.MaxSubsteps);
            bool lost = sim.AnyEjected() || sim.AnyRestingAboveDangerLine(FruitCakeEnv.RestSpeedPx);

            double score = points * MergeWeight - sim.PileHeight() - (lost ? LosePenalty : 0.0) - Math.Abs(col - center) * CenterBias;
            if (score > best)
            {
                best = score;
                bestCol = col;
            }
        }
        return bestCol;
    }
}

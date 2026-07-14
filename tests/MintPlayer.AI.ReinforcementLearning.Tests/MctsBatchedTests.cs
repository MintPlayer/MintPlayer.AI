using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// <see cref="Mcts.SearchBatched{TState}"/> (batched leaf inference via virtual loss) correctness:
/// (1) at leafBatch 1 it reproduces the sequential <see cref="Mcts.Search{TState}"/> BITWISE — proving the
/// virtual-loss / batched-expand / batched-backprop machinery reduces exactly to the trusted path when there's no
/// batching; (2) it's deterministic per (seed, leafBatch); (3) it still finds a forced mate and returns a valid
/// distribution over only-legal moves. These guard the GPU-throughput feature without a network dependency.
/// </summary>
public class MctsBatchedTests
{
    private static readonly Connect4Game Game = new();
    private const int C = Connect4State.Columns;

    // Deterministic, state-dependent synthetic evaluator: non-uniform priors + a non-zero value derived purely from
    // the position, so the bitwise-equivalence test actually exercises priors/value flow (not the trivial all-zero case).
    private static (float[] Priors, float Value) Synthetic(Connect4State s)
    {
        ulong h = 1469598103934665603UL;
        foreach (var c in s.Cells) h = (h ^ c) * 1099511628211UL;
        var rng = new Xoshiro256StarStar(h);
        var p = new float[C];
        for (int i = 0; i < C; i++) p[i] = (float)rng.NextDouble();
        return (p, (float)(rng.NextDouble() * 2 - 1));
    }

    private static readonly Mcts.BatchEvaluate<Connect4State> SyntheticBatch =
        states => states.Select(Synthetic).ToList();
    private static readonly Mcts.Evaluate<Connect4State> SyntheticSingle = Synthetic;

    // Neutral evaluator (uniform priors, zero value) — same one the sequential MCTS soundness test uses.
    private static readonly Mcts.BatchEvaluate<Connect4State> NeutralBatch =
        states => states.Select(_ => (new float[C], 0f)).ToList();

    private static Connect4State MidGame()
    {
        // A short opening so the tree has real depth/structure.
        var s = Game.Root();
        foreach (int col in new[] { 3, 3, 4, 2, 4 }) s = Game.Apply(s, col);
        return s;
    }

    [Fact]
    public void Batched_at_leafBatch1_reproduces_Search_bitwise()
    {
        var state = MidGame();
        var cfg = new Mcts.Config(Simulations: 200);

        float[] sequential = Mcts.Search(Game, state, SyntheticSingle, cfg, new Xoshiro256StarStar(123));
        float[] batched1 = Mcts.SearchBatched(Game, state, SyntheticBatch, cfg, new Xoshiro256StarStar(123), leafBatch: 1);

        Assert.Equal(sequential.Length, batched1.Length);
        for (int i = 0; i < sequential.Length; i++)
            Assert.Equal(sequential[i], batched1[i]); // exact float equality — the machinery reduces to sequential
    }

    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(32)]
    public void Batched_is_deterministic_per_seed_and_batch(int leafBatch)
    {
        var state = MidGame();
        var cfg = new Mcts.Config(Simulations: 256);
        float[] a = Mcts.SearchBatched(Game, state, SyntheticBatch, cfg, new Xoshiro256StarStar(7), leafBatch);
        float[] b = Mcts.SearchBatched(Game, state, SyntheticBatch, cfg, new Xoshiro256StarStar(7), leafBatch);
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i]);
    }

    [Fact]
    public void Batched_finds_the_mate_and_returns_a_valid_distribution()
    {
        // Player 1 to move, three-in-a-row on cols 0,1,2 (bottom row); col 3 wins immediately.
        var cells = new byte[C * Connect4State.Rows];
        cells[0 * C + 0] = 1; cells[0 * C + 1] = 1; cells[0 * C + 2] = 1;
        cells[1 * C + 0] = 2; cells[1 * C + 1] = 2;
        var state = new Connect4State(cells, toMove: 1);

        float[] pi = Mcts.SearchBatched(Game, state, NeutralBatch, new Mcts.Config(Simulations: 400), new Xoshiro256StarStar(42), leafBatch: 8);

        int best = 0;
        for (int i = 1; i < pi.Length; i++) if (pi[i] > pi[best]) best = i;
        Assert.Equal(3, best);                                 // concentrates visits on the mating move
        Assert.Equal(1f, pi.Sum(), 3);                         // a probability distribution
        Assert.All(Game.LegalMoves(state), m => Assert.True(pi[m] >= 0f));
    }
}

using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Connect4;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// Connect-4 rules, the negamax oracle, and — the point of M39.1 — that <see cref="Mcts"/> is sound: with a neutral
/// (uniform-prior, zero-value) evaluator it still finds a forced win purely from tree search, agreeing with negamax.
/// This validates the novel MCTS machinery (terminal backup, zero-sum sign flip) independently of any network.
/// </summary>
public class Connect4Tests
{
    private static readonly Connect4Game Game = new();
    private const int C = Connect4State.Columns; // 7

    // Neutral evaluator: uniform priors (all-zero → the search fills in uniform-over-legal) and value 0 — so the
    // only signal MCTS can act on is the terminal outcomes it discovers by searching.
    private static readonly Mcts.Evaluate<Connect4State> Neutral = _ => (new float[C], 0f);

    private static byte[] Empty() => new byte[C * Connect4State.Rows];
    private static void Set(byte[] cells, int row, int col, int player) => cells[row * C + col] = (byte)player;

    [Fact]
    public void EmptyBoard_allColumnsLegal_player1ToMove()
    {
        var root = Game.Root();
        Assert.Equal(1, root.ToMove);
        Assert.Equal(Enumerable.Range(0, C), Game.LegalMoves(root));
        Assert.Equal(GameResult.Ongoing, Game.Result(root));
    }

    [Fact]
    public void Apply_dropsToLowestEmptyRow_andFlipsSide()
    {
        var s1 = Game.Apply(Game.Root(), move: 3);
        Assert.Equal(1, s1.Cells[0 * C + 3]); // landed on the bottom row
        Assert.Equal(2, s1.ToMove);
        var s2 = Game.Apply(s1, move: 3);
        Assert.Equal(2, s2.Cells[1 * C + 3]); // stacked on top
        Assert.Equal(1, s2.ToMove);
    }

    [Fact]
    public void HorizontalFour_isALossForTheSideToMove()
    {
        // Player 1 has a completed horizontal four on the bottom row; it is now player 2's turn to move → Loss for 2.
        var cells = Empty();
        for (int col = 0; col < 4; col++) Set(cells, 0, col, 1);
        var state = new Connect4State(cells, toMove: 2);
        Assert.Equal(GameResult.Loss, Game.Result(state));
    }

    [Theory]
    [InlineData(3)]  // a mate-in-1 the winner completes by playing column 3
    public void Negamax_and_Mcts_both_find_the_mate_in_one(int winningColumn)
    {
        // Player 1 to move with three-in-a-row on cols 0,1,2 (bottom row); playing col 3 wins immediately.
        var cells = Empty();
        Set(cells, 0, 0, 1); Set(cells, 0, 1, 1); Set(cells, 0, 2, 1);
        Set(cells, 1, 0, 2); Set(cells, 1, 1, 2); // some opponent pieces, non-threatening
        var state = new Connect4State(cells, toMove: 1);

        var (score, bestMove) = Connect4Solver.Solve(state, maxDepth: 1);
        Assert.Equal(1, score);                 // a forced win for the side to move
        Assert.Equal(winningColumn, bestMove);

        float[] pi = Mcts.Search(Game, state, Neutral, new Mcts.Config(Simulations: 200), new Xoshiro256StarStar(42));
        int mctsMove = Argmax(pi);
        Assert.Equal(winningColumn, mctsMove);  // MCTS concentrates its visits on the winning move
    }

    [Fact]
    public void Mcts_returns_a_distribution_over_only_legal_moves()
    {
        // Fill column 0 so it is illegal; MCTS must assign it zero probability.
        var cells = Empty();
        for (int row = 0; row < Connect4State.Rows; row++) Set(cells, row, 0, row % 2 == 0 ? 1 : 2);
        var state = new Connect4State(cells, toMove: 1);

        float[] pi = Mcts.Search(Game, state, Neutral, new Mcts.Config(Simulations: 50), new Xoshiro256StarStar(7));
        Assert.Equal(0f, pi[0]);                                    // full column → never chosen
        Assert.Equal(1f, pi.Sum(), 3);                             // a probability distribution
        Assert.All(Game.LegalMoves(state), m => Assert.True(pi[m] >= 0f));
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}

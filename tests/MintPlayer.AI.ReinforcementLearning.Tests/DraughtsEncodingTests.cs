using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.AI.ReinforcementLearning.Environments.Draughts;
using Xunit.Abstractions;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The M47.2 gate: the (from, to) action encoding and the 5-plane observation. The encoding is
/// mover-relative (Black plays on the 180°-rotated board) and collapses rare maximum-capture forks
/// sharing (from, to) onto one index, resolved by a deterministic canonical pick — so the checks are:
/// encode→apply round-trips over thousands of random-playout positions with ZERO unmapped legal
/// moves (every complete move's index is offered by the seam), a collision audit (counted and
/// logged), a directed Turkish-fork position pinning the canonical pick, and observation planes
/// that rotate with the mover. The index math here is an independent reimplementation of the .pg
/// <c>moveIndex</c> — the test cross-checks the engine, not itself.
/// </summary>
public class DraughtsEncodingTests(ITestOutputHelper output)
{
    private const sbyte WM = 1, WK = 2, BM = -1, BK = -2;

    // Independent reimplementation of the .pg encoding: playable index = sq/2, action index =
    // fromPlayable · N²/2 + toPlayable, on the 180°-rotated board when Black moves.
    private static int MoveIndex(DraughtsState s, DraughtsMove m)
    {
        int cells = s.Size * s.Size, half = cells / 2;
        int from = s.WhiteToMove ? m.From : cells - 1 - m.From;
        int to = s.WhiteToMove ? m.To : cells - 1 - m.To;
        return from / 2 * half + to / 2;
    }

    // The engine's canonical pick for an index shared by several sequences: most captures first,
    // then the smallest sorted capture list.
    private static DraughtsMove Canonical(IEnumerable<DraughtsMove> candidates)
        => candidates
            .OrderByDescending(m => m.Captures.Count)
            .ThenBy(m => string.Join(",", m.Captures.Order().Select(c => c.ToString("D3"))))
            .First();

    [Theory]
    [InlineData(DraughtsVariant.International10, 2500, 500)]
    [InlineData(DraughtsVariant.English8, 1024, 320)]
    public void Action_and_observation_spaces_have_the_designed_sizes(DraughtsVariant variant, int policy, int observation)
    {
        var game = new DraughtsGame(variant);
        Assert.Equal(policy, game.PolicySize);
        Assert.Equal(observation, game.ObservationSize);
    }

    [Theory]
    [InlineData(DraughtsVariant.International10, 300)]
    [InlineData(DraughtsVariant.English8, 300)]
    public void Random_playouts_round_trip_every_action_index(DraughtsVariant variant, int games)
    {
        var game = new DraughtsGame(variant);
        var rng = new Random(12345);
        int positions = 0, collisions = 0;

        for (int g = 0; g < games; g++)
        {
            var state = game.Root();
            for (int ply = 0; ply < 200 && game.Result(state) == GameResult.Ongoing; ply++)
            {
                var moves = DraughtsRules.LegalMoves(state);
                var indices = game.LegalMoves(state);
                positions++;

                // The seam offers a distinct, in-range index for EVERY legal move (zero unmapped).
                Assert.Equal(indices.Count, indices.Distinct().Count());
                Assert.All(indices, i => Assert.InRange(i, 0, game.PolicySize - 1));
                var offered = indices.ToHashSet();
                Assert.All(moves, m => Assert.Contains(MoveIndex(state, m), offered));
                if (indices.Count < moves.Count) collisions++;

                // Applying an index ≡ making the canonical move it names, with the side flipped.
                int chosen = indices[rng.Next(indices.Count)];
                var applied = game.Apply(state, chosen);
                var expected = DraughtsRules.MakeMove(state, Canonical(moves.Where(m => MoveIndex(state, m) == chosen)));
                Assert.Equal(expected.Squares, applied.Squares);
                Assert.Equal(expected.NoProgress, applied.NoProgress);
                Assert.Equal(!state.WhiteToMove, applied.WhiteToMove);
                state = applied;
            }
        }
        output.WriteLine($"{variant}: {positions} positions, {collisions} with a (from,to) collision (canonical pick).");
    }

    // A directed Turkish fork: the white man reaches (2,4) by two complete 2-capture sequences over
    // DIFFERENT pieces — the left pair {(1,1),(1,3)} or the right pair {(3,1),(3,3)}. English rules
    // (no majority) keep both as distinct moves, but they share (from, to), so the seam offers ONE
    // index and applyIndex takes the canonical pick: equal capture counts, then the smallest sorted
    // capture list — the left pair.
    [Fact]
    public void Colliding_capture_fork_resolves_to_the_canonical_pick()
    {
        var board = new sbyte[64];
        board[2] = WM;                                       // (2,0)
        board[1 * 8 + 1] = BM; board[3 * 8 + 1] = BM;        // (1,1), (1,3)
        board[1 * 8 + 3] = BM; board[3 * 8 + 3] = BM;        // (3,1), (3,3)
        var s = new DraughtsState(DraughtsVariant.English8, board, whiteToMove: true);
        var game = new DraughtsGame(DraughtsVariant.English8);

        var moves = DraughtsRules.LegalMoves(s);
        Assert.Equal(2, moves.Count);
        Assert.All(moves, m => { Assert.Equal(2, m.From); Assert.Equal(4 * 8 + 2, m.To); Assert.Equal(2, m.Captures.Count); });

        int index = Assert.Single(game.LegalMoves(s));
        var next = game.Apply(s, index);
        Assert.Equal(0, next.Squares[1 * 8 + 1]);            // left pair captured…
        Assert.Equal(0, next.Squares[3 * 8 + 1]);
        Assert.Equal(BM, next.Squares[1 * 8 + 3]);           // …right pair untouched
        Assert.Equal(BM, next.Squares[3 * 8 + 3]);
    }

    // The start board is 180°-symmetric, so under a correct mover-relative encoding White's opening
    // indices and Black's (same board, Black to move) must be the SAME set.
    [Theory]
    [InlineData(DraughtsVariant.International10)]
    [InlineData(DraughtsVariant.English8)]
    public void Start_position_indices_are_mover_symmetric(DraughtsVariant variant)
    {
        var game = new DraughtsGame(variant);
        var white = game.Root();
        var black = new DraughtsState(variant, white.Squares, whiteToMove: false);
        Assert.Equal(game.LegalMoves(white).Order(), game.LegalMoves(black).Order());
    }

    [Fact]
    public void Observation_planes_are_mover_relative_and_carry_the_draw_clock()
    {
        var game = new DraughtsGame(DraughtsVariant.International10);
        var board = new sbyte[100];
        board[22] = WM; board[0] = WK; board[33] = BM; board[99] = BK;

        var whiteView = new float[game.ObservationSize];
        game.WriteObservation(new DraughtsState(DraughtsVariant.International10, board, true, noProgress: 20), whiteView);
        Assert.Equal(1f, whiteView[0 * 100 + 22]);           // my man
        Assert.Equal(1f, whiteView[1 * 100 + 0]);            // my king
        Assert.Equal(1f, whiteView[2 * 100 + 33]);           // their man
        Assert.Equal(1f, whiteView[3 * 100 + 99]);           // their king
        Assert.Equal(4f, whiteView.Take(400).Sum());         // exactly one hot cell per piece
        Assert.All(whiteView.Skip(400), v => Assert.Equal(20f / 80f, v));   // the draw clock, everywhere

        // Same board, Black to move: plane roles swap AND every square rotates 180° (sq → 99−sq).
        var blackView = new float[game.ObservationSize];
        game.WriteObservation(new DraughtsState(DraughtsVariant.International10, board, false, noProgress: 20), blackView);
        Assert.Equal(1f, blackView[0 * 100 + (99 - 33)]);    // my man (the black man, rotated)
        Assert.Equal(1f, blackView[1 * 100 + (99 - 99)]);    // my king
        Assert.Equal(1f, blackView[2 * 100 + (99 - 22)]);    // their man
        Assert.Equal(1f, blackView[3 * 100 + (99 - 0)]);     // their king
        Assert.Equal(4f, blackView.Take(400).Sum());
    }
}

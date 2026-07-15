using Microsoft.Extensions.DependencyInjection;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;
using MintPlayer.SourceGenerators.Attributes;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Draughts;

/// <summary>
/// Draughts as an <see cref="IZeroSumGame{TState}"/> — the self-play strength showcase (PLAN M47). It adapts
/// the perft-verified single-source engine (<c>draughts_solver.pg</c>) to the seam: a complete capture
/// sequence is ONE action (so <see cref="Apply"/> keeps its flips-side contract), the action space is
/// (from, to) over playable squares — (N²/2)² = 2500 on 10×10, 1024 on 8×8 — and the observation is 5 planes:
/// my men, my kings, their men, their kings, and the normalized no-progress clock (nets need to see the draw
/// rule coming). Everything is MOVER-RELATIVE: for Black both the planes and the action indices use the
/// 180°-rotated board (sq → N²−1−sq), so "my forward" always points the same way and the net learns one
/// perspective instead of two. DI registers the international 10×10 showcase; the english 8×8 pipeline-check
/// variant is a constructor argument away (M47.4 runs it first).
/// </summary>
[Register(typeof(IZeroSumGame<DraughtsState>), ServiceLifetime.Singleton, "ReinforcementLearningGames")]
[Register(typeof(IMaterialScore<DraughtsState>), ServiceLifetime.Singleton, "ReinforcementLearningGames")]
public sealed class DraughtsGame : IZeroSumGame<DraughtsState>, IMaterialScore<DraughtsState>
{
    private const int Planes = 5;

    // Standard relative values, indexed by |piece| (none, man, king). A king ≈ 3 men (M47 PRD §3.4).
    private static readonly int[] PieceValues = [0, 1, 3];

    private readonly DraughtsVariant _variant;
    private readonly int _size;

    public DraughtsGame() : this(DraughtsVariant.International10) { }

    public DraughtsGame(DraughtsVariant variant)
    {
        _variant = variant;
        _size = variant == DraughtsVariant.English8 ? 8 : 10;
    }

    public int PolicySize => (_size * _size / 2) * (_size * _size / 2);   // 2500 / 1024
    public int ObservationSize => Planes * _size * _size;                 // 500 / 320

    /// <summary>The side-to-move's material advantage in man-units (its pieces − the opponent's, king = 3).
    /// Dense reward signal for self-play value shaping + the difficulty ladder's strength metric.</summary>
    public float MaterialAdvantage(DraughtsState state)
    {
        int white = 0, black = 0;
        foreach (sbyte p in state.Squares)
        {
            int v = PieceValues[Math.Abs(p)];
            if (p > 0) white += v; else if (p < 0) black += v;
        }
        int diff = white - black;
        return state.WhiteToMove ? diff : -diff;
    }

    public DraughtsState Root(ulong? seed = null) => DraughtsState.StartPosition(_variant);

    // The seam methods delegate to the single-source core so training and the future browser client
    // (M47.5) share one implementation of "legal indices / apply an index / terminal result".
    public IReadOnlyList<int> LegalMoves(DraughtsState state) => state.Core.legalMoveIndices();

    public DraughtsState Apply(DraughtsState state, int move) => new(state.Core.applyIndex(move));

    public GameResult Result(DraughtsState state) => state.Core.result() switch
    {
        1 => GameResult.Loss,   // side to move cannot move (wiped out or blocked)
        2 => GameResult.Draw,   // no-progress rule
        _ => GameResult.Ongoing,
    };

    public void WriteObservation(DraughtsState state, Span<float> destination)
    {
        destination.Clear();
        int cells = _size * _size;
        bool white = state.WhiteToMove;
        var squares = state.Squares;
        // Planes 0/1 = my men/kings, 2/3 = theirs; Black sees the 180°-rotated board so its "forward"
        // matches White's (must stay consistent with the .pg moverSq action-index rotation).
        for (int sq = 0; sq < cells; sq++)
        {
            sbyte piece = squares[sq];
            if (piece == 0) continue;
            bool mine = piece > 0 == white;
            int plane = (mine ? 0 : 2) + Math.Abs(piece) - 1;
            int rel = white ? sq : cells - 1 - sq;
            destination[plane * cells + rel] = 1f;
        }
        float clock = Math.Min(1f, (float)state.NoProgress / state.NoProgressLimit);
        destination.Slice(4 * cells, cells).Fill(clock);
    }
}

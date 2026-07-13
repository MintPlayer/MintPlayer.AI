namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>A terminal game result, stated RELATIVE TO THE SIDE TO MOVE in the queried state.</summary>
public enum GameResult
{
    /// <summary>Not a terminal position.</summary>
    Ongoing,
    /// <summary>The side to move has already won (rare — most games are won on the mover's own move).</summary>
    Win,
    /// <summary>The side to move has lost (e.g. it is checkmated / the opponent just completed a line).</summary>
    Loss,
    /// <summary>A draw (stalemate, repetition, full board, …).</summary>
    Draw,
}

/// <summary>
/// A perfect-information, two-player, <b>zero-sum</b>, deterministic game — the shared shape that MCTS
/// (<see cref="Mcts"/>) and a self-play trainer both consume. It is deliberately distinct from
/// <c>IEnvironment</c> (which has a single-agent reward/episode loop) and from <see cref="IDeterministicModel{TState}"/>
/// (whose single-goal <c>IsGoal</c> and always-valid <c>ActionCount</c> can't express side-to-move,
/// win/loss/draw, or per-state legal moves).
/// <para>
/// Conventions the implementer MUST honour: <see cref="Apply"/> returns a FRESH successor and leaves the input
/// <c>state</c> untouched (search reuses one state across many candidate moves), and applying a move FLIPS the side
/// to move. <see cref="Result"/> and <see cref="WriteObservation"/> are always from the perspective of the side to
/// move in the given state, so the network sees one canonical "me vs. them" view regardless of whose turn it is.
/// </para>
/// </summary>
/// <typeparam name="TState">The immutable-from-the-caller's-view game state (a position).</typeparam>
public interface IZeroSumGame<TState>
{
    /// <summary>The fixed action-index space — equals the policy head width of the net that plays this game.
    /// A given state's legal subset is <see cref="LegalMoves"/>; every legal move is an index in [0, PolicySize).</summary>
    int PolicySize { get; }

    /// <summary>The length of the observation <see cref="WriteObservation"/> writes (the net's input width).</summary>
    int ObservationSize { get; }

    /// <summary>The start position. <paramref name="seed"/> is for games with a randomized setup (chess/Connect-4 ignore it).</summary>
    TState Root(ulong? seed = null);

    /// <summary>The legal action indices in <paramref name="state"/> (a subset of [0, PolicySize)). Never empty for a
    /// non-terminal state.</summary>
    IReadOnlyList<int> LegalMoves(TState state);

    /// <summary>The successor after playing <paramref name="move"/> — a fresh state, with the side to move flipped.
    /// <paramref name="state"/> is not mutated. <paramref name="move"/> must be one of <see cref="LegalMoves"/>.</summary>
    TState Apply(TState state, int move);

    /// <summary>The terminal result for the side to move in <paramref name="state"/>, or <see cref="GameResult.Ongoing"/>.</summary>
    GameResult Result(TState state);

    /// <summary>Writes the side-to-move-relative observation of <paramref name="state"/> into
    /// <paramref name="destination"/> (length <see cref="ObservationSize"/>).</summary>
    void WriteObservation(TState state, Span<float> destination);
}

using RLNet.Core.Environments;
using RLNet.Core.Random;

namespace RLNet.Environments.Game2048;

/// <summary>
/// 2048 as a Gymnasium-style environment. Observation: 16 floats, exponent/15
/// (0 = empty, 1.0 = the 32768 tile). Reward: sum of merged-tile EXPONENTS per move —
/// a log-scaled, well-conditioned signal (raw score increments span 4..2048+ and wreck
/// TD targets; the research flagged reward scaling as a top silent failure). The classic
/// game score is tracked separately on <see cref="Score"/> for reporting.
/// Illegal moves throw — consult <see cref="CurrentActionMask"/> (trainers do this
/// automatically via <see cref="IActionMaskProvider"/>). Terminal when no move is legal.
/// Pre-registered solved criterion (PRD §6): reach the 2048 tile (exponent ≥ 11) in
/// ≥ 10% of 100 eval games.
/// </summary>
public sealed class Env2048 : IEnvironment<float[], int>, IActionMaskProvider
{
    public const int MaxEpisodeMoves = 10_000; // safety net; real games end far earlier

    private readonly byte[] _board = new byte[16];
    private Xoshiro256StarStar _rng = new(0);
    private int _moves;
    private bool _done = true;

    public Env2048()
    {
        ObservationSpace = new BoxSpace(0f, 1f, 16);
        ActionSpace = new DiscreteSpace(Board2048.ActionCount);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    /// <summary>Classic 2048 score (sum of merged tile values) of the current episode.</summary>
    public int Score { get; private set; }

    public int MaxTile => 1 << Board2048.MaxExponent(_board);

    public ReadOnlySpan<byte> Board => _board;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        Array.Clear(_board);
        Board2048.Spawn(_board, _rng);
        Board2048.Spawn(_board, _rng);
        Score = 0;
        _moves = 0;
        _done = false;
        return (Observation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");

        if (!Board2048.ApplyMove(_board, action, out int mergedExponents, out int mergedValues))
            throw new InvalidOperationException(
                $"Illegal move {action} for the current board; consult CurrentActionMask().");

        Score += mergedValues;
        Board2048.Spawn(_board, _rng);
        _moves++;

        bool terminated = !Board2048.AnyMoveAvailable(_board);
        bool truncated = !terminated && _moves >= MaxEpisodeMoves;
        _done = terminated || truncated;
        return new StepResult<float[]>(Observation(), mergedExponents, terminated, truncated, EnvInfo.Empty);
    }

    public bool[] CurrentActionMask() => Board2048.ValidMoves(_board);

    /// <summary>The current state's observation (for driving the env from outside a trainer).</summary>
    public float[] CurrentObservation() => Observation();

    private float[] Observation()
    {
        var obs = new float[16];
        for (int i = 0; i < 16; i++) obs[i] = _board[i] / 15f;
        return obs;
    }

    public string RenderString() => Board2048.Render(_board) + $"score {Score}   moves {_moves}   best {MaxTile}\n";
}

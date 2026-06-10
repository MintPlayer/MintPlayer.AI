using RLNet.Core.Environments;
using RLNet.Core.Random;

namespace RLNet.Environments.RushHour;

/// <summary>
/// Rush Hour as an RL environment over a puzzle set: each episode picks a puzzle
/// (random by default, or pinned via <see cref="FixedPuzzleIndex"/> for per-puzzle
/// evaluation). Observation: two 36-cell planes — vehicle identity ((index+1)/16,
/// 0 = empty) and red-car occupancy. Action space: fixed 32 (vehicle·2 + direction),
/// masked. Reward: −1 per move, +100 on solving (sparse, as pre-registered in PRD §6);
/// optional potential-based shaping adds +2 per cell of red-car progress.
/// Gate (PRD): ≥ 90% of the easy puzzle set solved within 2× the BFS-optimal moves.
/// </summary>
public sealed class RushHourEnv : IEnvironment<float[], int>, IActionMaskProvider
{
    private readonly IReadOnlyList<RushHourPuzzle> _puzzles;
    private readonly int _maxMoves;
    private readonly bool _shapedReward;
    private Xoshiro256StarStar _rng = new(0);
    private RushHourPuzzle _puzzle;
    private int[] _positions;
    private int _moves;
    private bool _done = true;

    public RushHourEnv(IReadOnlyList<RushHourPuzzle> puzzles, int maxMoves = 100, bool shapedReward = false)
    {
        _puzzles = puzzles.Count > 0 ? puzzles : throw new ArgumentException("At least one puzzle required.");
        _maxMoves = maxMoves;
        _shapedReward = shapedReward;
        _puzzle = puzzles[0];
        _positions = RushHourBoard.InitialPositions(_puzzle);

        ObservationSpace = new BoxSpace(0f, 1f, 72);
        ActionSpace = new DiscreteSpace(RushHourBoard.ActionCount);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    /// <summary>Pin episodes to one puzzle (per-puzzle gate evaluation); null = random per episode.</summary>
    public int? FixedPuzzleIndex { get; set; }

    public RushHourPuzzle CurrentPuzzle => _puzzle;
    public int MovesUsed => _moves;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        _puzzle = _puzzles[FixedPuzzleIndex ?? _rng.NextInt(_puzzles.Count)];
        _positions = RushHourBoard.InitialPositions(_puzzle);
        _moves = 0;
        _done = false;
        return (Observation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");

        int vehicle = action / 2, direction = action % 2;
        Span<int> grid = stackalloc int[36];
        RushHourBoard.FillOccupancy(_puzzle, _positions, grid);
        if (!RushHourBoard.CanMove(_puzzle, _positions, grid, vehicle, direction))
            throw new InvalidOperationException(
                $"Illegal action {action} (vehicle {vehicle}, dir {direction}); consult CurrentActionMask().");

        int redBefore = _positions[0];
        _positions[vehicle] += direction == 0 ? -1 : 1;
        _moves++;

        bool solved = RushHourBoard.IsSolved(_puzzle, _positions);
        double reward = solved ? 100 : -1;
        if (_shapedReward)
            reward += 2.0 * (_positions[0] - redBefore);

        bool truncated = !solved && _moves >= _maxMoves;
        _done = solved || truncated;
        return new StepResult<float[]>(Observation(), reward, solved, truncated, EnvInfo.Empty);
    }

    public bool[] CurrentActionMask() => RushHourBoard.ActionMask(_puzzle, _positions);

    public float[] CurrentObservation() => Observation();

    private float[] Observation()
    {
        var obs = new float[RushHourBoard.ObservationSize];
        RushHourBoard.WriteObservation(_puzzle, _positions, obs);
        return obs;
    }

    public string RenderString()
        => RushHourBoard.Render(_puzzle, _positions)
           + $"moves {_moves}/{_maxMoves}   optimal {_puzzle.OptimalMoves}\n";
}

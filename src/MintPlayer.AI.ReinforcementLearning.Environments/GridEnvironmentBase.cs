using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments;

/// <summary>
/// Base for rectangular grid MDPs defined by a character map. Runtime dynamics are
/// sampled directly from <see cref="Model"/>, so the declared transition model and the
/// actual behavior cannot drift apart — testing one tests the other.
/// </summary>
public abstract class GridEnvironmentBase : ITabularEnvironment
{
    public const int ActionLeft = 0;
    public const int ActionDown = 1;
    public const int ActionRight = 2;
    public const int ActionUp = 3;

    private readonly string[] _map;
    private readonly int _maxEpisodeSteps;
    private readonly int _startState;
    private Xoshiro256StarStar _rng = new(0);
    private int _state;
    private int _elapsedSteps;
    private bool _done = true;

    protected GridEnvironmentBase(string[] map, int maxEpisodeSteps)
    {
        _map = map;
        _maxEpisodeSteps = maxEpisodeSteps;
        Rows = map.Length;
        Cols = map[0].Length;
        _startState = Array.FindIndex(map, row => row.Contains('S')) * Cols
            + map.First(row => row.Contains('S')).IndexOf('S');

        ObservationSpace = new DiscreteSpace(StateCount);
        ActionSpace = new DiscreteSpace(4);
    }

    public int Rows { get; }
    public int Cols { get; }
    public int StateCount => Rows * Cols;
    public int ActionCount => 4;
    public int CurrentState => _state;

    public Space<int> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    public char CellAt(int state) => _map[state / Cols][state % Cols];

    public bool IsTerminal(int state) => IsTerminalCell(CellAt(state));

    public (int Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        _state = _startState;
        _elapsedSteps = 0;
        _done = false;
        return (_state, EnvInfo.Empty);
    }

    public StepResult<int> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        if (!ActionSpace.Contains(action))
            throw new ArgumentOutOfRangeException(nameof(action));

        var transition = SampleTransition(_state, action);
        _state = transition.NextState;
        _elapsedSteps++;

        bool truncated = !transition.Terminated && _elapsedSteps >= _maxEpisodeSteps;
        _done = transition.Terminated || truncated;
        return new StepResult<int>(_state, transition.Reward, transition.Terminated, truncated, EnvInfo.Empty);
    }

    public abstract IEnumerable<Transition> Model(int state, int action);

    protected abstract bool IsTerminalCell(char cell);

    protected abstract double RewardFor(char destinationCell);

    /// <summary>Destination of a deterministic move; bumping the edge stays in place.</summary>
    protected int MoveFrom(int state, int action)
    {
        int row = state / Cols, col = state % Cols;
        switch (action)
        {
            case ActionLeft: col = Math.Max(col - 1, 0); break;
            case ActionDown: row = Math.Min(row + 1, Rows - 1); break;
            case ActionRight: col = Math.Min(col + 1, Cols - 1); break;
            case ActionUp: row = Math.Max(row - 1, 0); break;
        }
        return row * Cols + col;
    }

    protected Transition DeterministicTransition(int state, int action, double probability)
    {
        int next = MoveFrom(state, action);
        char cell = CellAt(next);
        return new Transition(probability, next, RewardFor(cell), IsTerminalCell(cell));
    }

    private Transition SampleTransition(int state, int action)
    {
        double roll = _rng.NextDouble();
        double cumulative = 0;
        Transition last = default;
        foreach (var t in Model(state, action))
        {
            cumulative += t.Probability;
            last = t;
            if (roll < cumulative) return t;
        }
        return last; // guards against cumulative rounding to slightly below 1.0
    }

    public string RenderString()
    {
        var sb = new StringBuilder();
        for (int row = 0; row < Rows; row++)
        {
            for (int col = 0; col < Cols; col++)
            {
                int state = row * Cols + col;
                sb.Append(state == _state ? '@' : _map[row][col]);
                sb.Append(' ');
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }
}

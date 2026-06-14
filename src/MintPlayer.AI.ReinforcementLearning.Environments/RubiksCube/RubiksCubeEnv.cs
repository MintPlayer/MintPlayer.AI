using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Rubik's Cube as an RL environment (PRD §11): each episode scrambles a solved cube
/// with d quarter-turns, d ~ U[1..MaxScrambleDepth] (the curriculum knob), and the agent
/// must invert the scramble. Observation: 54 stickers one-hot over 6 colors (Box(324)).
/// Action space: the 12 quarter-turns (<see cref="FaceletCube.QuarterTurnMoves"/>).
/// Physically every turn is always legal, but the mask forbids the INVERSE of the
/// previous move: undoing can never shorten a solution (the pre-move state is one move
/// away by construction), and it removes the dominant greedy failure mode — A A' A A'
/// oscillation. Reward: −1 per move, +100 on solved (same scheme as Rush Hour;
/// return = 101 − moves). Episodes truncate at <c>maxMoves</c> moves.
/// Gate: ≥ 90% of 100 eval scrambles (depths 1–6) solved within 20 moves.
/// </summary>
public sealed class RubiksCubeEnv : IEnvironment<float[], int>, IActionMaskProvider
{
    public const int ObservationSize = FaceletCube.FaceletCount * FaceletCube.FaceCount; // 324
    public const int ActionCount = 12;

    private readonly int _maxScrambleDepth;
    private readonly int _maxMoves;
    private Xoshiro256StarStar _rng = new(0);
    private FaceletCube _cube = new();
    private int _moves;
    private int _lastAction = -1;
    private bool _done = true;

    public RubiksCubeEnv(int maxScrambleDepth = 6, int maxMoves = 20)
    {
        _maxScrambleDepth = maxScrambleDepth > 0 ? maxScrambleDepth
            : throw new ArgumentOutOfRangeException(nameof(maxScrambleDepth));
        _maxMoves = maxMoves;

        ObservationSpace = new BoxSpace(0f, 1f, ObservationSize);
        ActionSpace = new DiscreteSpace(ActionCount);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    /// <summary>Pin every episode to one scramble depth (per-depth evaluation); null = U[1..max].</summary>
    public int? FixedScrambleDepth { get; set; }

    public int ScrambleDepth { get; private set; }
    public int MovesUsed => _moves;

    /// <summary>The episode's scramble sequence (diagnostics; future imitation labels via inversion).</summary>
    public IReadOnlyList<string> ScrambleMoves { get; private set; } = [];

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);

        ScrambleDepth = FixedScrambleDepth ?? 1 + _rng.NextInt(_maxScrambleDepth);
        do
        {
            _cube = new FaceletCube();
            ScrambleMoves = FaceletCube.ScrambleMoves(_rng, ScrambleDepth, quarterTurnsOnly: true);
            _cube.Apply(ScrambleMoves);
        } while (_cube.IsSolved); // long quarter-turn sequences can compose to identity; rescramble

        _moves = 0;
        _lastAction = -1;
        _done = false;
        return (Observation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        if (action == InverseAction(_lastAction))
            throw new InvalidOperationException(
                $"Illegal action {action}: undoing the previous move is masked; consult CurrentActionMask().");

        _cube.ApplyQuarterTurn(action);
        _moves++;
        _lastAction = action;

        bool solved = _cube.IsSolved;
        double reward = solved ? 100 : -1;
        bool truncated = !solved && _moves >= _maxMoves;
        _done = solved || truncated;
        return new StepResult<float[]>(Observation(), reward, solved, truncated, EnvInfo.Empty);
    }

    public bool[] CurrentActionMask() => ActionMask(_lastAction);

    /// <summary>All quarter-turns are legal except the inverse of the previous action (−1 = none).</summary>
    public static bool[] ActionMask(int lastAction)
    {
        var mask = new bool[ActionCount];
        Array.Fill(mask, true);
        int inverse = InverseAction(lastAction);
        if (inverse >= 0) mask[inverse] = false;
        return mask;
    }

    /// <summary>Quarter-turn action ids pair up as (face, face'): 0↔1, 2↔3, … (−1 passes through).</summary>
    public static int InverseAction(int action) => action < 0 ? -1 : action ^ 1;

    public float[] CurrentObservation() => Observation();

    private float[] Observation()
    {
        var obs = new float[ObservationSize];
        WriteObservation(_cube, obs);
        return obs;
    }

    /// <summary>One-hot sticker encoding, shared with the web app's solve rollout.</summary>
    public static void WriteObservation(FaceletCube cube, Span<float> observation)
    {
        observation.Clear();
        for (int i = 0; i < FaceletCube.FaceletCount; i++)
            observation[i * FaceletCube.FaceCount + cube[i]] = 1f;
    }

    public string RenderString()
        => _cube.RenderString() + $"moves {_moves}/{_maxMoves}   scramble depth {ScrambleDepth}\n";
}

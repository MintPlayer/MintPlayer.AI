using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.CrazyFruits;

/// <summary>
/// Crazy Fruits (match-3) as an RL environment (docs/prd/CRAZY_FRUITS_PRD.md §3.6). One <see cref="Step"/> =
/// one swap: apply the chosen adjacent swap, resolve cascades + refill in pure compute, return the new board.
///
/// <para>Action = one of the 112 adjacent swaps; only match-producing swaps are legal
/// (<see cref="CurrentActionMask"/> — masking is mandatory here, PRD §3.3). Observation = 448 floats (6
/// one-hot fruit planes + the would-match plane). Reward = points/30 (a plain 3-match ⇒ 1.0), no step
/// penalty. Episodes never terminate (a dead board reshuffles inside the engine); they truncate at the move
/// budget, so the learner bootstraps from the final state — the score-maximizing framing.</para>
/// </summary>
public sealed class CrazyFruitsEnv : IEnvironment<float[], int>, IActionMaskProvider, IStatefulEnvironment
{
    public const int ObservationSize = CrazyFruitsBoard.ObservationSize; // 448
    public const int ActionCount = CrazyFruitsBoard.ActionCount;         // 112
    /// <summary>Reward normalization: points per plain 3-match, so rewards sit near 1 (PRD §3.5).</summary>
    public const float RewardScale = 30f;

    private static readonly string[] FruitNames = ["strawberry", "banana", "orange", "grape", "apple", "lemon"];

    /// <summary>Plain-language name per observation feature, index order of <c>buildObservation</c>.</summary>
    public static readonly IReadOnlyList<string> ObservationLabels = BuildObservationLabels();

    /// <summary>Plain-language name per swap action (56 horizontal then 56 vertical).</summary>
    public static readonly IReadOnlyList<string> ActionLabels = BuildActionLabels();

    private static string[] BuildObservationLabels()
    {
        const int size = CrazyFruitsBoard.Size;
        var labels = new List<string>(ObservationSize);
        for (int f = 0; f < CrazyFruitsBoard.FruitTypes; f++)
            for (int r = 0; r < size; r++)
                for (int c = 0; c < size; c++)
                    labels.Add($"{FruitNames[f]} at row {r + 1}, column {c + 1} (1/0)");
        for (int r = 0; r < size; r++)
            for (int c = 0; c < size; c++)
                labels.Add($"Row {r + 1}, column {c + 1} can join a match after one swap (1/0)");
        // Not ActionLabels: static field initializers run in declaration order, and this builds first.
        var actions = BuildActionLabels();
        foreach (var label in actions) labels.Add($"Immediate points of \"{label}\" (÷100)");
        foreach (var label in actions) labels.Add($"Guaranteed cascade points of \"{label}\" (÷100)");
        return [.. labels];
    }

    private static string[] BuildActionLabels()
    {
        const int size = CrazyFruitsBoard.Size;
        var labels = new List<string>(ActionCount);
        for (int r = 0; r < size; r++)
            for (int c = 0; c < size - 1; c++)
                labels.Add($"Swap row {r + 1} col {c + 1} ↔ col {c + 2}");
        for (int r = 0; r < size - 1; r++)
            for (int c = 0; c < size; c++)
                labels.Add($"Swap col {c + 1} row {r + 1} ↔ row {r + 2}");
        return [.. labels];
    }

    private readonly int _moveBudget;
    private readonly CrazyFruitsBoard _board = new();
    private Xoshiro256StarStar _rng = new(0);
    private int _moves;
    private bool _done = true;

    /// <param name="moveBudget">Moves per episode (the locked default 30, PRD §4).</param>
    public CrazyFruitsEnv(int moveBudget = 30)
    {
        _moveBudget = moveBudget;
        ObservationSpace = new BoxSpace(0f, 1f, ObservationSize);
        ActionSpace = new DiscreteSpace(ActionCount);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    public int MoveBudget => _moveBudget;
    public int Score => _board.Score;
    public int MovesMade => _moves;
    /// <summary>The underlying board — for the scripted baselines and the serving/watch path.</summary>
    public CrazyFruitsBoard Board => _board;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        // Each episode's board deals from the env RNG stream: reproducible under an explicit seed, fresh
        // boards across unseeded resets.
        _board.Reset(_rng.NextUInt64());
        _moves = 0;
        _done = false;
        return (_board.BuildObservation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        int points = _board.ApplySwap(action);
        if (points < 0)
            throw new ArgumentOutOfRangeException(nameof(action), action,
                "Illegal swap (no match) — act only on CurrentActionMask().");
        _moves++;
        // The board never dies (deadlocks reshuffle in-engine, PRD §3.5): terminated stays false and the
        // budget end is TRUNCATION, so the learner bootstraps from the final observation.
        bool truncated = _moves >= _moveBudget;
        _done = truncated;
        return new StepResult<float[]>(_board.BuildObservation(), points / RewardScale, false, truncated, EnvInfo.Empty);
    }

    public bool[] CurrentActionMask() => _board.LegalMask();

    public float[] CurrentObservation() => _board.BuildObservation();

    public byte[] SaveState()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        var (s0, s1, s2, s3) = _rng.GetState();
        writer.Write(s0); writer.Write(s1); writer.Write(s2); writer.Write(s3);
        writer.Write(_moves);
        writer.Write(_done);
        _board.WriteState(writer);
        writer.Flush();
        return stream.ToArray();
    }

    public void RestoreState(byte[] state)
    {
        using var reader = new BinaryReader(new MemoryStream(state));
        _rng.SetState(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64());
        _moves = reader.ReadInt32();
        _done = reader.ReadBoolean();
        _board.ReadState(reader);
    }

    public string RenderString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"score={_board.Score} moves={_moves}/{_moveBudget} reshuffles={_board.Reshuffles} done={_done}");
        for (int r = 0; r < CrazyFruitsBoard.Size; r++)
        {
            for (int c = 0; c < CrazyFruitsBoard.Size; c++)
                sb.Append(_board.Fruit(r * CrazyFruitsBoard.Size + c));
            sb.AppendLine();
        }
        return sb.ToString();
    }
}

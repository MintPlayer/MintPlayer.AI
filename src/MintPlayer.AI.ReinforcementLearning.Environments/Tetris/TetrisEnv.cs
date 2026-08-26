using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Tetris;

/// <summary>
/// Tetris as an RL environment (docs/prd/TETRIS_PRD.md §3.6). One <see cref="Step"/> = one PLACEMENT
/// (afterstate macro-action: rot·10 + col, hard drop) — never a per-frame move; frame-level Tetris RL is
/// the literature's unanimous failure mode (PRD §2.2).
///
/// <para>Action = one of 40 placements, hard-masked (<see cref="CurrentActionMask"/>); the mask CAN go
/// all-false — that is the top-out, and the env reports it as <c>terminated</c> on the step that caused
/// it (unlike the other games' never-empty masks, PRD §3.3/risk 2). Observation = 454 floats (board +
/// piece one-hots + six per-action feature planes). Reward = LINES cleared (0–4, linear — the display
/// score is cosmetic, PRD §3.5); top-out ⇒ terminated (the ended reward stream is the penalty), the
/// piece budget ⇒ truncated (bootstraps).</para>
/// </summary>
public sealed class TetrisEnv : IEnvironment<float[], int>, IActionMaskProvider, IStatefulEnvironment
{
    public const int ObservationSize = TetrisBoard.ObservationSize; // 454
    public const int ActionCount = TetrisBoard.ActionCount;         // 40
    /// <summary>Reward normalization: reward IS lines cleared (0–4) — already O(1), so ÷1.</summary>
    public const float RewardScale = 1f;

    private static readonly string[] PieceNames = ["I", "O", "T", "S", "Z", "L", "J"];
    private static readonly string[] FeaturePlaneNames =
        ["landing height (÷20)", "eroded piece cells (÷8)", "Δ row transitions (÷20)",
         "Δ column transitions (÷20)", "Δ holes (÷10)", "Δ well depth (÷20)"];

    /// <summary>Plain-language name per observation feature, index order of <c>buildObservation</c>.</summary>
    public static readonly IReadOnlyList<string> ObservationLabels = BuildObservationLabels();

    /// <summary>Plain-language name per placement action (rot·10 + col).</summary>
    public static readonly IReadOnlyList<string> ActionLabels = BuildActionLabels();

    private static string[] BuildObservationLabels()
    {
        var labels = new List<string>(ObservationSize);
        for (int y = 0; y < TetrisBoard.Height; y++)
            for (int x = 0; x < TetrisBoard.Width; x++)
                labels.Add($"Cell row {y + 1}, column {x + 1} filled (1/0)");
        foreach (var p in PieceNames) labels.Add($"Current piece is {p} (1/0)");
        foreach (var p in PieceNames) labels.Add($"Next piece is {p} (1/0)");
        var actions = BuildActionLabels();
        foreach (var plane in FeaturePlaneNames)
            foreach (var label in actions)
                labels.Add($"{plane} of \"{label}\"");
        return [.. labels];
    }

    private static string[] BuildActionLabels()
    {
        var labels = new List<string>(ActionCount);
        for (int rot = 0; rot < 4; rot++)
            for (int col = 0; col < TetrisBoard.Width; col++)
                labels.Add($"Drop rotation {rot} at column {col + 1}");
        return [.. labels];
    }

    private readonly int _pieceBudget;
    private readonly bool _sevenBag;
    private readonly int _garbageEvery;
    private readonly TetrisBoard _board = new();
    private Xoshiro256StarStar _rng = new(0);
    private int _pieces;
    private bool _done = true;

    /// <param name="pieceBudget">Placements per episode (the locked training cap 500, PRD §4).</param>
    /// <param name="sevenBag">7-bag piece stream; training/benchmark stays uniform (PRD §1).</param>
    /// <param name="garbageEvery">Rising-garbage cadence; 0 = off (the training default).</param>
    public TetrisEnv(int pieceBudget = 500, bool sevenBag = false, int garbageEvery = 0)
    {
        _pieceBudget = pieceBudget;
        _sevenBag = sevenBag;
        _garbageEvery = garbageEvery;
        ObservationSpace = new BoxSpace(0f, 1f, ObservationSize);
        ActionSpace = new DiscreteSpace(ActionCount);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    public int PieceBudget => _pieceBudget;
    public int Score => _board.Score;
    public int Lines => _board.Lines;
    public int PiecesPlaced => _pieces;
    /// <summary>The underlying board — for the scripted baselines and the serving/watch path.</summary>
    public TetrisBoard Board => _board;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        _board.Reset(_rng.NextUInt64(), _sevenBag, _garbageEvery);
        _pieces = 0;
        _done = false;
        return (_board.BuildObservation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        int cleared = _board.ApplyPlacement(action);
        if (cleared < 0)
            throw new ArgumentOutOfRangeException(nameof(action), action,
                "Illegal placement — act only on CurrentActionMask().");
        _pieces++;
        // Top-out (all-masked new piece, or garbage overflow) is TERMINATION — the reward stream ends,
        // which is the death penalty. The budget end is TRUNCATION, so the learner bootstraps.
        bool terminated = _board.GameOver;
        bool truncated = !terminated && _pieces >= _pieceBudget;
        _done = terminated || truncated;
        return new StepResult<float[]>(_board.BuildObservation(), cleared / RewardScale, terminated, truncated, EnvInfo.Empty);
    }

    public bool[] CurrentActionMask() => _board.LegalMask();

    public float[] CurrentObservation() => _board.BuildObservation();

    public byte[] SaveState()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        var (s0, s1, s2, s3) = _rng.GetState();
        writer.Write(s0); writer.Write(s1); writer.Write(s2); writer.Write(s3);
        writer.Write(_pieces);
        writer.Write(_done);
        _board.WriteState(writer);
        writer.Flush();
        return stream.ToArray();
    }

    public void RestoreState(byte[] state)
    {
        using var reader = new BinaryReader(new MemoryStream(state));
        _rng.SetState(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64());
        _pieces = reader.ReadInt32();
        _done = reader.ReadBoolean();
        _board.ReadState(reader);
    }

    public string RenderString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"lines={_board.Lines} score={_board.Score} pieces={_pieces}/{_pieceBudget} " +
                      $"current={PieceNames[_board.CurrentPiece]} next={PieceNames[_board.NextPiece]} over={_board.GameOver}");
        for (int y = 0; y < TetrisBoard.Height; y++)
        {
            for (int x = 0; x < TetrisBoard.Width; x++)
                sb.Append(_board.Cell(x, y) ? '#' : '.');
            sb.AppendLine();
        }
        return sb.ToString();
    }
}

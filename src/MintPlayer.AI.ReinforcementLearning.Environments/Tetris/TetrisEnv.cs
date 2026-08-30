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
/// piece one-hots + six per-action feature planes). Reward = LINES cleared plus the tetris
/// bonus (owner amendment 2026-08-26: a 4-line clear pays 4+8=12 — the AI is asked to build for
/// tetrises; raw superlinear score as reward was rejected as the stack-and-camp trap, PRD §3.5); the
/// NES score (40/100/300/1200) is the reported metric. Top-out ⇒ terminated (the ended reward stream is the penalty), the
/// piece budget ⇒ truncated (bootstraps).</para>
/// </summary>
public sealed class TetrisEnv : IEnvironment<float[], int>, IActionMaskProvider, IStatefulEnvironment
{
    public const int ObservationSize = TetrisBoard.ObservationSize; // 454
    public const int ActionCount = TetrisBoard.ActionCount;         // 40
    /// <summary>Reward normalization: reward IS lines cleared (0–4) — already O(1), so ÷1.</summary>
    public const float RewardScale = 1f;

    private static readonly string[] PieceNames = ["I", "O", "T", "S", "Z", "L", "J"];
    // M57.5: the widened basis (15 planes). Absolute afterstate values, not deltas.
    private static readonly string[] FeaturePlaneNames =
        ["landing height (÷20)", "eroded piece cells (÷8)", "row transitions (÷40)",
         "column transitions (÷40)", "holes (÷20)", "well depth outside the well column (÷20)",
         "tetris-ready rows (÷4)", "covered-well depth (÷10)", "burned (non-tetris) lines (÷4)",
         "is a tetris (1/0)", "column 9 height over safe (÷10)", "rows above reachable wall height (÷10)",
         "DIG mode (1/0)", "LINEOUT mode (1/0)", "placement is legal (1/0)"];

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

    // ── Training-only potential-based shaping (M54.3 escalation, invoked 2026-08-26) ────────────────────────
    // The bare reward is too sparse to bootstrap from: random placements clear ~1 line per 450 steps, and
    // the 180K-step eval was still near-random (0.5 lines). PBRS (Ng et al.): reward += γ·Φ(s′) − Φ(s),
    // Φ = max(0, Ceiling − (4·holes + rowT + colT + wells))/scale — dense signal along the Dellacherie
    // basis, policy-invariant when PotentialGamma matches the learner's γ.
    //
    // ORIENTATION IS LOAD-BEARING (bug found 2026-08-26 on tet2train, which learned to be WORSE than the
    // unshaped run): termination uses Φ := 0, so every living state's Φ must sit ABOVE 0 — with a negative
    // Φ the terminal step pays 0 − Φ(s) = +|Φ|, a REWARD for dying that grows with how bad the board is.
    // The ceiling keeps a clean board at ~+6 potential (forfeited on top-out) and a terrible board near 0.
    // TRAIN env only — the eval env scores the bare game, so gates stay honest.

    /// <summary>Training-only distribution mix (owner request 2026-08-26: ONE net for garbage on+off):
    /// each episode flips the rising-garbage mode on (every 10) or off, 50/50 from the env RNG stream.
    /// Garbage boards (near-full rows, gaps, buried holes) are otherwise out-of-distribution for a net
    /// whose own play converges to clean flat stacks. At γ=0 + dense targets this is benign covariate
    /// mixing — the labels are computed from the observation, identically on both distributions (the M52
    /// lesson). Eval envs keep a FIXED garbage setting so both gate protocols stay comparable.</summary>
    public bool MixedGarbageTraining { get; set; }

    /// <summary>Enable the training-only board-potential shaping. Default off — a plain env scores the bare game.</summary>
    public bool ShapeBoardPotential { get; set; }
    /// <summary>Must match the learner's γ for policy invariance.</summary>
    public double PotentialGamma { get; set; } = 0.995;
    /// <summary>Board badness (~0–200 on real boards) above which the potential clamps to 0.</summary>
    public float PotentialCeiling { get; set; } = 200f;
    /// <summary>Divisor mapping potential into reward units (lines ≈ 1; clean board ⇒ Φ = 6).</summary>
    public float PotentialScale { get; set; } = 25f;

    private float Potential()
    {
        float badness = 4f * _board.Holes() + _board.RowTransitions() + _board.ColTransitions() + _board.WellSum();
        return Math.Max(0f, PotentialCeiling - badness) / PotentialScale;
    }

    public int PieceBudget => _pieceBudget;
    public int Score => _board.Score;
    public int Lines => _board.Lines;
    public int Tetrises => _board.Tetrises;
    public int PiecesPlaced => _pieces;
    /// <summary>The underlying board — for the scripted baselines and the serving/watch path.</summary>
    public TetrisBoard Board => _board;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        int garbage = MixedGarbageTraining
            ? ((_rng.NextUInt64() & 1) == 0 ? 0 : 10)
            : _garbageEvery;
        _board.Reset(_rng.NextUInt64(), _sevenBag, garbage);
        _pieces = 0;
        _done = false;
        return (_board.BuildObservation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        float potentialBefore = ShapeBoardPotential ? Potential() : 0f;
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
        float reward = (cleared + (cleared == 4 ? TetrisBoard.TetrisRewardBonus : 0)) / RewardScale;
        if (ShapeBoardPotential)
            reward += (float)(PotentialGamma * (terminated ? 0f : Potential())) - potentialBefore;
        return new StepResult<float[]>(_board.BuildObservation(), reward, terminated, truncated, EnvInfo.Empty);
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

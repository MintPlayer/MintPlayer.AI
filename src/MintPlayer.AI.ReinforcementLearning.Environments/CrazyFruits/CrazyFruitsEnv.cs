using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.CrazyFruits;

/// <summary>
/// Crazy Fruits (match-3) as an RL environment (docs/prd/CRAZY_FRUITS_PRD.md §3.6). One <see cref="Step"/> =
/// one swap: apply the chosen adjacent swap, resolve cascades + refill in pure compute, return the new board.
///
/// <para>Action = one of the 112 adjacent swaps; only match-producing swaps are legal
/// (<see cref="CurrentActionMask"/> — masking is mandatory here, PRD §3.3). Observation =
/// <see cref="ObservationSize"/> floats (fruit/kind/would-match planes + three per-action value planes).
/// Reward = points/<see cref="RewardScale"/>. Episodes never terminate (a dead board reshuffles inside the
/// engine); they truncate at the move budget, so the learner bootstraps from the final state — the
/// score-maximizing framing.</para>
/// </summary>
public sealed class CrazyFruitsEnv : IEnvironment<float[], int>, IActionMaskProvider, IStatefulEnvironment
{
    public const int ObservationSize = CrazyFruitsBoard.ObservationSize; // 1040
    public const int ActionCount = CrazyFruitsBoard.ActionCount;         // 112
    /// <summary>Reward normalization. Re-picked for specials (SPECIALS PRD §3.5): random-play means ~86
    /// points/move with auto-firing specials, so ÷100 keeps a typical move ≈ O(1) and a bomb+bomb board
    /// clear a manageable (Huber-linear) tail instead of a 50σ TD target.</summary>
    public const float RewardScale = 100f;

    // Matches the web renderer's picks from the FruitCake art catalog (crazy-fruits-render.ts FRUIT_TIER).
    private static readonly string[] FruitNames = ["strawberry", "grape", "orange", "pear", "pineapple", "watermelon"];

    /// <summary>Plain-language name per observation feature, index order of <c>buildObservation</c>.</summary>
    public static readonly IReadOnlyList<string> ObservationLabels = BuildObservationLabels();

    /// <summary>Plain-language name per swap action (56 horizontal then 56 vertical).</summary>
    public static readonly IReadOnlyList<string> ActionLabels = BuildActionLabels();

    private static readonly string[] KindNames = ["striped (row blast)", "striped (column blast)", "wrapped", "sugar bomb"];

    private static string[] BuildObservationLabels()
    {
        const int size = CrazyFruitsBoard.Size;
        var labels = new List<string>(ObservationSize);
        for (int f = 0; f < CrazyFruitsBoard.FruitTypes; f++)
            for (int r = 0; r < size; r++)
                for (int c = 0; c < size; c++)
                    labels.Add($"{FruitNames[f]} at row {r + 1}, column {c + 1} (1/0)");
        for (int k = 0; k < CrazyFruitsBoard.SpecialKinds; k++)
            for (int r = 0; r < size; r++)
                for (int c = 0; c < size; c++)
                    labels.Add($"{KindNames[k]} at row {r + 1}, column {c + 1} (1/0)");
        for (int r = 0; r < size; r++)
            for (int c = 0; c < size; c++)
                labels.Add($"Row {r + 1}, column {c + 1} takes part in some legal move (1/0)");
        // Not ActionLabels: static field initializers run in declaration order, and this builds first.
        var actions = BuildActionLabels();
        foreach (var label in actions) labels.Add($"Immediate points of \"{label}\" (÷300)");
        foreach (var label in actions) labels.Add($"Guaranteed cascade points of \"{label}\" (÷300)");
        foreach (var label in actions) labels.Add($"Guaranteed cascade points + creation bonus of \"{label}\" (÷300)");
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

    // ── Training-only creation shaping (owner decision 2026-07-24: specials score only when they FIRE) ──────
    // The game score carries no creation reward, so a γ=0 learner is blind to the value of MAKING a special
    // (the payoff lands on a later move). The TRAINING env re-adds that signal to the reward; the eval env
    // stays plain, so every gate measures the real fire-only game score.

    /// <summary>Enable the training-only creation-shaping term. Default off — a plain env scores the bare game.</summary>
    public bool ShapeCreationRewards { get; set; }
    /// <summary>Shaping bonus (game-score points) per striped/wrapped/bomb created, added to the reward only.</summary>
    public float StripedShaping { get; set; } = 40f;
    public float WrappedShaping { get; set; } = 60f;
    public float BombShaping { get; set; } = 100f;

    // ── Combo curriculum (COMBO_CURRICULUM PRD, M52 — train env only; both default OFF) ─────────────────────

    /// <summary>Probability that a fresh episode's board is dealt combo-ready: one ADJACENT special pair plus
    /// up to two singles, injected by overwriting plain cells' kinds (fruit types unchanged, so the deal's
    /// no-instant-match and has-legal-swap guarantees survive; a bomb never joins runs and its swaps are
    /// always legal). Draws come from the env RNG stream, never the board's refill stream — seeded runs stay
    /// deterministic, and the default 0 leaves every draw count untouched.</summary>
    public double SeedSpecialsProb { get; set; }

    /// <summary>Probability that an ε-exploration step with a legal special+special swap available picks
    /// uniformly among the combo swaps (the <c>DqnOptions.ExploreBias</c> hook,
    /// <see cref="SuggestComboExploration"/>) — realized combo experience on natural boards.</summary>
    public double ComboExploreBias { get; set; }

    /// <summary>The §3.6 escalation's shaping (use INSTEAD of <see cref="ShapeCreationRewards"/>, with γ&gt;0):
    /// potential-based, Φ(s) = Σ option value of on-board specials (same 40/60/100 weights), reward +=
    /// γ·Φ(s′) − Φ(s). Policy-invariant when <see cref="PotentialGamma"/> matches the learner's γ — it prices
    /// holding a special without rewarding hoarding (a no-op at γ=0, so only meaningful on the escalation).</summary>
    public bool ShapeSpecialsPotential { get; set; }
    public double PotentialGamma { get; set; } = 0.5;

    private float Potential()
    {
        float sum = 0f;
        for (int i = 0; i < CrazyFruitsBoard.Cells; i++)
            sum += _board.Kind(i) switch { 1 or 2 => StripedShaping, 3 => WrappedShaping, 4 => BombShaping, _ => 0f };
        return sum;
    }

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
        if (SeedSpecialsProb > 0 && _rng.NextDouble() < SeedSpecialsProb)
            SeedSpecials();
        _moves = 0;
        _done = false;
        return (_board.BuildObservation(), EnvInfo.Empty);
    }

    /// <summary>Deal the fresh board combo-ready (see <see cref="SeedSpecialsProb"/>): an adjacent special
    /// pair plus up to two singles, all on plain cells with their fruit type kept.</summary>
    private void SeedSpecials()
    {
        var grid = _board.GridSnapshot();
        bool horizontal = _rng.NextDouble() < 0.5;
        int r = _rng.NextInt(horizontal ? CrazyFruitsBoard.Size : CrazyFruitsBoard.Size - 1);
        int c = _rng.NextInt(horizontal ? CrazyFruitsBoard.Size - 1 : CrazyFruitsBoard.Size);
        int pairA = r * CrazyFruitsBoard.Size + c;
        int pairB = horizontal ? pairA + 1 : pairA + CrazyFruitsBoard.Size;
        grid[pairA] = Seeded(grid[pairA]);
        grid[pairB] = Seeded(grid[pairB]);
        int extras = _rng.NextInt(3);
        for (int i = 0; i < extras; i++)
        {
            int cell = _rng.NextInt(CrazyFruitsBoard.Cells);
            if (grid[cell] < 16) grid[cell] = Seeded(grid[cell]); // plain cells only
        }
        _board.LoadGrid(grid);
    }

    // Kinds 1..4 uniform (stripedH · stripedV · wrapped · bomb); striped/wrapped keep the cell's fruit type
    // (typewise match structure unchanged), the bomb is colorless.
    private int Seeded(int packedPlain)
    {
        int kind = 1 + _rng.NextInt(4);
        return kind == 4 ? 4 * 16 : kind * 16 + packedPlain % 16;
    }

    /// <summary>The <c>DqnOptions.ExploreBias</c> hook (see <see cref="ComboExploreBias"/>): −1 when the
    /// roll passes or no special+special swap is legal, else a uniform pick among the legal combo swaps.</summary>
    public int SuggestComboExploration(Xoshiro256StarStar rng)
    {
        if (rng.NextDouble() >= ComboExploreBias) return -1;
        var mask = _board.LegalMask();
        var combos = new List<int>();
        for (int a = 0; a < ActionCount; a++)
        {
            if (!mask[a]) continue;
            var (cellA, cellB) = _board.SwapCells(a);
            if (_board.Kind(cellA) > 0 && _board.Kind(cellB) > 0) combos.Add(a);
        }
        return combos.Count == 0 ? -1 : combos[rng.NextInt(combos.Count)];
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        float potentialBefore = ShapeSpecialsPotential ? Potential() : 0f;
        int points = _board.ApplySwap(action);
        if (points < 0)
            throw new ArgumentOutOfRangeException(nameof(action), action,
                "Illegal swap (no match) — act only on CurrentActionMask().");
        _moves++;
        // The board never dies (deadlocks reshuffle in-engine, PRD §3.5): terminated stays false and the
        // budget end is TRUNCATION, so the learner bootstraps from the final observation.
        bool truncated = _moves >= _moveBudget;
        _done = truncated;
        float reward = points / RewardScale;
        if (ShapeCreationRewards)
            reward += (StripedShaping * _board.MoveCreatedStriped
                     + WrappedShaping * _board.MoveCreatedWrapped
                     + BombShaping * _board.MoveCreatedBombs) / RewardScale;
        if (ShapeSpecialsPotential)
            reward += (float)(PotentialGamma * Potential() - potentialBefore) / RewardScale;
        return new StepResult<float[]>(_board.BuildObservation(), reward, false, truncated, EnvInfo.Empty);
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

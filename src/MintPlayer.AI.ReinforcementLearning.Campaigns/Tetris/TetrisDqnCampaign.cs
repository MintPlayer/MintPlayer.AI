using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.Tetris;
using MintPlayer.SourceGenerators.Attributes;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Tetris DQN campaign (`--game tetris`, PLAN M54) on the shared <see cref="DqnScoreCampaign"/> spine —
/// score-maximizing where the score currency is LINES: eval is the mean lines cleared over fixed-seed
/// greedy 500-piece episodes (uniform pieces, no garbage — the benchmark-honest protocol A; the
/// discriminative garbage-survival protocol B runs in the Lab's `--baselines`). Trains the masked
/// Double+Dueling <see cref="DuelingQNet"/> over afterstate macro-actions with γ=0.995 (TETRIS_PRD.md
/// §3.7 — long-horizon survival; the γ=0 dense-target recipe structurally does not transfer here).
/// ε-greedy by default; <see cref="TetrisDqnOptions.Noisy"/> is the pre-registered escalation lever.
/// </summary>
public sealed partial class TetrisDqnCampaign : DqnScoreCampaign
{
    [Inject] private readonly TetrisEnv evalEnv;

    private TetrisDqnOptions Typed => Options as TetrisDqnOptions ?? new TetrisDqnOptions();

    public override string Environment => "tetris";
    protected override string StepNoun => "placements";
    protected override string GateLabel => "mean score";
    protected override string DisplayName => "Tetris DQN";
    protected override string FreshStartDetail => $" (10×20, afterstate placements, {evalEnv.PieceBudget}-piece episodes)";
    protected override int ObservationSize => TetrisEnv.ObservationSize;
    protected override IReadOnlyList<string>? InputLabels => TetrisEnv.ObservationLabels;
    protected override IReadOnlyList<string>? OutputLabels => TetrisEnv.ActionLabels;

    /// <summary>The Crazy Fruits recipe re-pointed at long-horizon survival (PRD §3.7 locks).</summary>
    /// <summary>
    /// M57.5c warm-start. The observation gained the tetris-aware planes (454 -> 854), so a net trained
    /// before M57.5 must have its input grown to fit. This is function-preserving ONLY because planes 0-5
    /// are still the M54 basis, in the M54 order, occupying exactly indices 214..453 — i.e. the whole of
    /// the old observation is an unchanged PREFIX of the new one, and the added planes start at zero
    /// weight. <c>ObservationLayout_KeepsTheM54BasisAsItsPrefix</c> pins that.
    ///
    /// The rule this encodes: you cannot swap an input's MEANING when auto-widening a net. Growing is
    /// safe only as an append. Reordering or reinterpreting an existing plane would keep the width legal
    /// while silently feeding the transplanted weights different quantities — no guard would catch it,
    /// because both the input width and the action count would still match.
    /// </summary>
    protected override IValueNet AdaptWarmNet(DuelingQNet loaded)
    {
        if (loaded.InputSize == TetrisEnv.ObservationSize) return loaded;
        if (loaded.InputSize > TetrisEnv.ObservationSize)
            throw new InvalidOperationException(
                $"warm net input {loaded.InputSize} EXCEEDS the observation {TetrisEnv.ObservationSize} — " +
                "the observation shrank, so the old weights no longer line up. Retrain from scratch.");

        Log($"growing the loaded net's input {loaded.InputSize} → {TetrisEnv.ObservationSize} " +
            "(M54 basis preserved as the observation prefix; new planes zero-init, so the policy is unchanged at step 0)");
        return loaded.GrowInput(TetrisEnv.ObservationSize);
    }

    protected override DqnOptions BaseOptions => new()
    {
        Dueling = true,
        DoubleDqn = true,
        Hidden = Options.Hidden,
        Gamma = Options.Gamma,
        NStep = Typed.NStep,
        NoisyNets = Typed.Noisy,
        LearningRate = Options.LearningRate,
        BufferCapacity = Typed.BufferCapacity,
        BatchSize = 128,
        WarmupSteps = 2_000,
        TargetSyncEvery = 1_000,
        Epsilon = new LinearSchedule(Options.EpsilonStart, Typed.EpsilonEnd, 30_000),
        EvalEpisodes = Options.EvalEpisodes,
        DenseTargets = Typed.DenseRegression ? DenseTargetsFromObservation : null,
        DenseTargetWeight = Typed.DenseTargetWeight,
    };

    // The observation's six per-action feature planes carry exactly the Dellacherie basis; undoing the
    // plane normalizers reconstructs, per action, −landing + eroded − ΔrowT − ΔcolT − 4·Δholes − Δwells —
    // the canonical evaluator up to a PER-STATE constant (absolute-vs-delta transitions), which the dueling
    // V head absorbs. ÷10 into target units. A legal placement always has landing > 0 (row 19 lands at
    // 0.05·20); a zero landing plane ⇒ illegal ⇒ NaN (unsupervised).
    private const int PlaneBase = TetrisBoard.Width * TetrisBoard.Height + 2 * TetrisBoard.PieceCount;
    private const int A = TetrisEnv.ActionCount;

    // M57.5: reconstruct the WIDENED evaluator per action, exactly. The planes carry ABSOLUTE afterstate
    // quantities plus the DIG/LINEOUT mode flags, so this applies the same branch the engine does in
    // PgTetris.evalAfterstate — no per-state constant for the dueling V head to absorb, and no
    // hand-written inverse of the normalizers to drift (the M54 hazard).
    // Weights MUST match the .pg consts; TetrisEnvTests pins agreement against the engine's own scores.
    private const float WHoles = -5.582f, WWells = -0.847f, WReady = 3.402f, WCovered = -0.201f;
    private const float WBurn = -3.700f, WBurnDig = -0.650f, WHoleDig = -0.505f, WCol9 = -0.355f;
    private const float WTetris = 7.047f, WInacc = -0.975f;

    internal static float[] DenseTargetsFromObservation(float[] obs)
    {
        var targets = new float[A];
        for (int a = 0; a < A; a++)
        {
            float Plane(int i) => obs[PlaneBase + i * A + a];

            if (Plane(14) < 0.5f) { targets[a] = float.NaN; continue; } // explicit legality flag

            // Pure LINEAR combination — the planes are already gated, so there is no DIG/LINEOUT branch
            // here (a piecewise target only fitted to R^2 0.54, and spike s7 showed that accuracy tops
            // out 100% of episodes for ANY evaluator).
            // Planes 2-5 are DELTAS (kept in the M54 form so the shipped net transplants); the target
            // needs absolutes, but the two differ only by a per-state constant, which the centring below
            // removes exactly. Plane 15 carries the well-column-excluded well delta the evaluator wants.
            float s = -Plane(0) * 20f + Plane(1) * 8f - Plane(2) * 20f - Plane(3) * 20f
                    + WHoles * Plane(4) * 10f + WWells * Plane(15) * 20f
                    + WReady * Plane(6) * 4f + WCovered * Plane(7) * 10f
                    + WBurn * Plane(8) * 4f + WTetris * Plane(9)
                    + WCol9 * Plane(10) * 10f + WInacc * Plane(11) * 10f
                    + WBurnDig * Plane(12) * 4f + WHoleDig * Plane(13) * 20f;
            targets[a] = s;
        }

        // Centre the per-state targets on the mean of the LEGAL actions, then scale.
        // Measured: the widened evaluator's raw values sit at mean -92, sd 28.7, so a bare /10 produced
        // targets centred at -9.2 with sd 2.9 — against M54's roughly zero-centred, sd~1 — and training
        // regressed (score 120 -> 29 over 60K steps while loss fell: a target-SCALE failure, not the
        // distribution narrowing that signature usually means). Switching the planes from deltas to
        // absolute values for exactness is what reintroduced the offset; the old delta form existed to
        // avoid it. Centring is free: a dueling V head absorbs any per-state constant by construction,
        // and what the advantage head must learn is the RANKING, which centring leaves untouched.
        float sum = 0f; int n = 0;
        for (int a = 0; a < A; a++) if (!float.IsNaN(targets[a])) { sum += targets[a]; n++; }
        if (n == 0) return targets;
        float mean = sum / n;
        for (int a = 0; a < A; a++) if (!float.IsNaN(targets[a])) targets[a] = (targets[a] - mean) / 10f;
        return targets;
    }

    protected override (double Gate, IReadOnlyList<CampaignMetric> Metrics, string Summary) EvaluateNet(IValueNet net)
    {
        var (score, lines, tetrises, pieces, topOuts) = EvalNet(net);
        var metrics = new CampaignMetric[]
        {
            new("score", score, "F0"),
            new("lines", lines, "F1"),
            new("tetrises", tetrises, "F2"),
            new("pieces", pieces, "F1"),
            new("topouts", topOuts, "F0"),
        };
        return (score, metrics,
            $"mean score {score:F0} | lines {lines:F1} | tetrises {tetrises:F2} | pieces {pieces:F1} | top-outs {topOuts}/{Options.EvalEpisodes}");
    }

    /// <summary>Mean NES score (the gate metric — the owner's objective), lines, tetrises and pieces
    /// survived over fixed-seed greedy masked episodes. Top-outs are the stack-and-camp watchdog.</summary>
    private (double Score, double Lines, double Tetrises, double Pieces, int TopOuts) EvalNet(IValueNet net)
    {
        var agent = new GreedyQAgent(net, TetrisEnv.ActionCount);
        double totalScore = 0, totalLines = 0, totalTetrises = 0, totalPieces = 0;
        int topOuts = 0;
        for (int e = 0; e < Options.EvalEpisodes; e++)
        {
            var (obs, _) = evalEnv.Reset((ulong)(5_000 + e));
            while (true)
            {
                int action = agent.Act(obs, evalEnv.CurrentActionMask(), greedy: true);
                var step = evalEnv.Step(action);
                obs = step.Observation;
                if (step.Done)
                {
                    if (step.Terminated) topOuts++;
                    break;
                }
            }
            totalScore += evalEnv.Score;
            totalLines += evalEnv.Lines;
            totalTetrises += evalEnv.Tetrises;
            totalPieces += evalEnv.PiecesPlaced;
        }
        int n = Options.EvalEpisodes;
        return (totalScore / n, totalLines / n, totalTetrises / n, totalPieces / n, topOuts);
    }
}

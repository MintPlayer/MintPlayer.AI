using MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// The Lab's `--game blockdude` entry point: parses the campaign flags and runs the
/// <see cref="BlockDudeImitationCampaign"/> on the shared <see cref="CampaignRunner"/>. The loop, resume, eval
/// cadence and checkpointing live in the runner; console + CSV live in <see cref="CampaignCli"/>.
/// </summary>
/// <remarks>
/// <para><c>--fresh</c> is the blank-slate switch: it deletes this phase's net, Adam moments and progress
/// sidecar before training, and rotates the CSV so a restarted run does not silently append to the previous
/// run's log.</para>
///
/// <para>Without <c>--fresh</c>, a progress sidecar whose fingerprint disagrees with the current settings (seed,
/// learning rate, phase, curriculum version) is refused with a message rather than blended into this run.</para>
///
/// <para><c>--grow</c> adds capacity when the net SATURATES rather than on a clock: after
/// <c>--grow-patience</c> gate evaluations (default 6) without beating the window's best gate by
/// <c>--grow-min-improvement</c> (default 0.04), the net climbs one rung of
/// <see cref="BlockDudeGrowth.Stages"/>. Rung 0 is the non-growing trunk, so the flag never starts a run
/// smaller than the default — unlike the shared <c>DqnGrowth</c> ladder, whose top rung is narrower than this
/// game's default net.</para>
/// </remarks>
internal static class BlockDudeLab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);

        // Read-only benchmark against the shipped levels; no training, no checkpoint writes.
        if (a.Has("--eval-levels")) { BlockDudeLevelBench.Run(args); return; }

        // Read-only diagnostic: predicted distance-to-goal vs the oracle's exact optimal move count.
        if (a.Has("--value-calibration")) { BlockDudeValueCalibration.Run(args); return; }

        // Read-only: how far back along the human demonstrations can net + A* still finish?
        if (a.Has("--demo-probe")) { BlockDudeDemoProbe.Run(args); return; }

        var f = Parse(a);
        (double hours, string dataDir, bool evalOnly, bool fresh) = (f.Hours, f.DataDir, f.EvalOnly, f.Fresh);

        string csv = f.Csv;
        if (fresh) RotateLog(csv);

        // Phase 2 is a different campaign, not a flag on the first: it trains on the SHIPPED levels via the
        // human demonstrations rather than on generated boards, so it shares neither the curriculum, the gate,
        // nor the checkpoint ids.
        if (f.Phase >= 2)
        {
            var xit = ExpertIterationOptions(a, f);
            LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
                services => services.AddBlockDudeExpertIterationCampaign(xit),
                CampaignCli.ConsoleAndCsv(csv));
            return;
        }

        var imitation = ImitationOptions(f);
        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddBlockDudeImitationCampaign(imitation),
            CampaignCli.ConsoleAndCsv(csv));
    }

    /// <summary>The campaign flags, plus the CSV path <c>--phase</c> selects.</summary>
    /// <remarks>M63.6: extracted from <see cref="Run"/>, where every default — including the phase-2 CSV
    /// split that decides WHICH log a run appends to — was reachable only by starting a real training run.
    /// Read after the three read-only <c>--eval-levels</c>/<c>--value-calibration</c>/<c>--demo-probe</c>
    /// dispatches, exactly as before.</remarks>
    internal sealed record Flags(
        double Hours, string DataDir, ulong Seed, float LearningRate, bool EvalOnly, bool Grow,
        int GrowPatience, double GrowMinImprovement, bool Fresh, long TargetSamples, int BoardsPerRound,
        int SamplesPerBoard, long GateEvery, int MaxStage, int Pinned, int Phase, string Csv);

    /// <summary>Reads the pure flag head of <see cref="Run"/> — nothing is opened, rotated or trained.</summary>
    internal static Flags Parse(CliArgs a)
    {
        string dataDir = a.Str("--data", "data");
        int phase = a.Int("--phase", 1);
        return new Flags(
            Hours: a.Dbl("--hours", 9),
            DataDir: dataDir,
            Seed: a.ULong("--seed", 1),
            LearningRate: a.Flt("--lr", 3e-4f),
            EvalOnly: a.Has("--eval-only"),
            Grow: a.Has("--grow"),
            GrowPatience: a.Int("--grow-patience", 6),
            GrowMinImprovement: a.Dbl("--grow-min-improvement", 0.04),
            Fresh: a.Has("--fresh"),
            TargetSamples: a.Long("--target-samples", 0),
            BoardsPerRound: a.Int("--boards-per-round", 4),
            SamplesPerBoard: a.Int("--samples-per-board", 512),
            GateEvery: a.Long("--gate-every", 100_000),
            MaxStage: a.Int("--max-stage", BlockDudeCurriculum.LastStage),
            Pinned: a.Int("--stage", -1),
            Phase: phase,
            Csv: Path.Combine(dataDir, "logs", phase >= 2 ? "blockdude-xit.csv" : "blockdude.csv"));
    }

    /// <summary>Phase-1 (generated-board imitation) options. Pure — it neither builds a net nor reads a file.</summary>
    internal static BlockDudeImitationOptions ImitationOptions(Flags f)
        => new()
        {
            Seed = f.Seed,
            LearningRate = f.LearningRate,
            Grow = f.Grow,
            GrowPatience = f.GrowPatience,
            GrowMinImprovement = f.GrowMinImprovement,
            Fresh = f.Fresh,
            TargetSamples = f.TargetSamples,
            BoardsPerRound = f.BoardsPerRound,
            SamplesPerBoard = f.SamplesPerBoard,
            GateEverySamples = f.GateEvery,
            MaxStage = f.MaxStage,
            PinStage = f.Pinned >= 0 ? f.Pinned : null,
            Phase = f.Phase,
        };

    /// <summary>Phase-2 (expert-iteration on the shipped levels) options — a different campaign, sharing
    /// neither the curriculum, the gate nor the checkpoint ids with phase 1.</summary>
    internal static BlockDudeExpertIterationOptions ExpertIterationOptions(CliArgs a, Flags f)
        => new()
        {
            Seed = f.Seed,
            LearningRate = f.LearningRate,
            Fresh = f.Fresh,
            WarmStart = !a.Has("--no-warm-start"),
            TargetSamples = f.TargetSamples,
            AttemptsPerLevel = a.Int("--attempts", 4),
            Expansions = a.Int("--expansions", 40_000),
            SearchSeconds = a.Int("--search-seconds", 6),
            Weight = a.Flt("--weight", 2f),
            PolicyWeight = a.Flt("--policy-weight", 5f),
            BeamWidth = a.Int("--beam-width", 256),
            BeamSeconds = a.Int("--beam-seconds", 8),
            BeamSecondsMax = a.Int("--beam-seconds-max", 45),
            BeamMovesPerSecond = a.Int("--beam-moves-per-second", 12),
            InitialFrontier = a.Int("--frontier", 20),
            FrontierGrowth = a.Dbl("--frontier-growth", 1.5),
            AdvanceRate = a.Dbl("--advance-rate", 0.75),
            DemoShare = a.Dbl("--demo-share", 0.25),
        };

    /// <summary>Moves an existing CSV aside. <see cref="CampaignCli.ConsoleAndCsv"/> APPENDS when the file
    /// exists, so without this a blank-slate run would continue the previous run's log and quietly present two
    /// different trajectories as one.</summary>
    internal static void RotateLog(string csv)
    {
        if (!File.Exists(csv)) return;
        string stamped = Path.Combine(
            Path.GetDirectoryName(csv)!,
            $"{Path.GetFileNameWithoutExtension(csv)}.{DateTime.UtcNow:yyyyMMdd-HHmmss}{Path.GetExtension(csv)}");
        File.Move(csv, stamped);
        Console.WriteLine($"--fresh: rotated {Path.GetFileName(csv)} -> {Path.GetFileName(stamped)}");
    }
}

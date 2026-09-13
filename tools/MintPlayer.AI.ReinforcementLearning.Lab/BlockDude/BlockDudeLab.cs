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

        double hours = a.Dbl("--hours", 9);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        float learningRate = a.Flt("--lr", 3e-4f);
        bool evalOnly = a.Has("--eval-only");
        bool grow = a.Has("--grow");
        int growPatience = a.Int("--grow-patience", 6);
        double growMinImprovement = a.Dbl("--grow-min-improvement", 0.04);
        bool fresh = a.Has("--fresh");
        long targetSamples = a.Long("--target-samples", 0);
        int boardsPerRound = a.Int("--boards-per-round", 4);
        int samplesPerBoard = a.Int("--samples-per-board", 512);
        long gateEvery = a.Long("--gate-every", 100_000);
        int maxStage = a.Int("--max-stage", BlockDudeCurriculum.LastStage);
        int pinned = a.Int("--stage", -1);
        int phase = a.Int("--phase", 1);

        string csv = Path.Combine(dataDir, "logs", phase >= 2 ? "blockdude-xit.csv" : "blockdude.csv");
        if (fresh) RotateLog(csv);

        // Phase 2 is a different campaign, not a flag on the first: it trains on the SHIPPED levels via the
        // human demonstrations rather than on generated boards, so it shares neither the curriculum, the gate,
        // nor the checkpoint ids.
        if (phase >= 2)
        {
            LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
                services => services.AddBlockDudeExpertIterationCampaign(new BlockDudeExpertIterationOptions
                {
                    Seed = seed,
                    LearningRate = learningRate,
                    Fresh = fresh,
                    WarmStart = !a.Has("--no-warm-start"),
                    TargetSamples = targetSamples,
                    AttemptsPerLevel = a.Int("--attempts", 4),
                    Expansions = a.Int("--expansions", 40_000),
                    SearchSeconds = a.Int("--search-seconds", 6),
                    Weight = a.Flt("--weight", 2f),
                    PolicyWeight = a.Flt("--policy-weight", 5f),
                    InitialFrontier = a.Int("--frontier", 20),
                    FrontierGrowth = a.Dbl("--frontier-growth", 1.5),
                    AdvanceRate = a.Dbl("--advance-rate", 0.75),
                    DemoShare = a.Dbl("--demo-share", 0.25),
                }),
                CampaignCli.ConsoleAndCsv(csv));
            return;
        }

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddBlockDudeImitationCampaign(new BlockDudeImitationOptions
            {
                Seed = seed,
                LearningRate = learningRate,
                Grow = grow,
                GrowPatience = growPatience,
                GrowMinImprovement = growMinImprovement,
                Fresh = fresh,
                TargetSamples = targetSamples,
                BoardsPerRound = boardsPerRound,
                SamplesPerBoard = samplesPerBoard,
                GateEverySamples = gateEvery,
                MaxStage = maxStage,
                PinStage = pinned >= 0 ? pinned : null,
                Phase = phase,
            }),
            CampaignCli.ConsoleAndCsv(csv));
    }

    /// <summary>Moves an existing CSV aside. <see cref="CampaignCli.ConsoleAndCsv"/> APPENDS when the file
    /// exists, so without this a blank-slate run would continue the previous run's log and quietly present two
    /// different trajectories as one.</summary>
    private static void RotateLog(string csv)
    {
        if (!File.Exists(csv)) return;
        string stamped = Path.Combine(
            Path.GetDirectoryName(csv)!,
            $"{Path.GetFileNameWithoutExtension(csv)}.{DateTime.UtcNow:yyyyMMdd-HHmmss}{Path.GetExtension(csv)}");
        File.Move(csv, stamped);
        Console.WriteLine($"--fresh: rotated {Path.GetFileName(csv)} -> {Path.GetFileName(stamped)}");
    }
}

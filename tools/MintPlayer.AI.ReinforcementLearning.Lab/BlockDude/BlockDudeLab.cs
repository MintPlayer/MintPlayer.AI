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
/// </remarks>
internal static class BlockDudeLab
{
    public static void Run(string[] args)
    {
        var a = new CliArgs(args);
        double hours = a.Dbl("--hours", 9);
        string dataDir = a.Str("--data", "data");
        ulong seed = a.ULong("--seed", 1);
        float learningRate = a.Flt("--lr", 3e-4f);
        bool evalOnly = a.Has("--eval-only");
        bool grow = a.Has("--grow");
        int growEvery = a.Int("--grow-every", 200_000);
        bool fresh = a.Has("--fresh");
        long targetSamples = a.Long("--target-samples", 0);
        int boardsPerRound = a.Int("--boards-per-round", 4);
        int samplesPerBoard = a.Int("--samples-per-board", 512);
        long gateEvery = a.Long("--gate-every", 100_000);
        int maxStage = a.Int("--max-stage", BlockDudeCurriculum.LastStage);
        int pinned = a.Int("--stage", -1);
        int phase = a.Int("--phase", 1);

        string csv = Path.Combine(dataDir, "logs", "blockdude.csv");
        if (fresh) RotateLog(csv);

        LabHost.Run(args, dataDir, hours, evalOnly, useGpu: false,
            services => services.AddBlockDudeImitationCampaign(new BlockDudeImitationOptions
            {
                Seed = seed,
                LearningRate = learningRate,
                Grow = grow,
                GrowEvery = growEvery,
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

using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>A campaign-progress event, emitted after each evaluation (and once at the end of the run).</summary>
public readonly record struct CampaignProgress(long Progress, CampaignEval Eval, bool IsFinal);

/// <summary>
/// Options for <see cref="CampaignRunner.Run"/>. The runner performs NO console or filesystem IO of its own —
/// it surfaces each evaluation through <see cref="OnEval"/> so the host (the Lab tool) can log and write CSV.
/// </summary>
public sealed record CampaignOptions
{
    /// <summary>Wall-clock budget. The loop stops once this elapses (or the campaign reports complete).</summary>
    public TimeSpan Duration { get; init; } = TimeSpan.FromHours(9);

    /// <summary>How often to evaluate + checkpoint, measured against <see cref="Now"/>.</summary>
    public TimeSpan EvalEvery { get; init; } = TimeSpan.FromMinutes(10);

    /// <summary>Delay before the first (baseline) evaluation.</summary>
    public TimeSpan FirstEvalAfter { get; init; } = TimeSpan.FromMinutes(2);

    /// <summary>Evaluate once (or run the campaign's standalone eval) and return, without training.</summary>
    public bool EvalOnly { get; init; }

    /// <summary>Invoked after each evaluation (and the final one) — the host's hook for console + CSV. No-op if null.</summary>
    public Action<CampaignProgress>? OnEval { get; init; }

    /// <summary>
    /// Clock source. Injectable so the loop is unit-testable without real time; defaults to the system UTC clock.
    /// </summary>
    public Func<DateTime> Now { get; init; } = static () => DateTime.UtcNow;
}

/// <summary>
/// Drives a resumable, wall-clock-budgeted training campaign: resume once, then train in chunks and
/// evaluate + checkpoint on a fixed cadence until the budget elapses or the campaign reports complete. Shared by
/// both campaign families (goal-reaching and score-maximizing). IO-agnostic — it does no console or file IO;
/// evaluation results flow to the host through <see cref="CampaignOptions.OnEval"/>, and all persistence goes
/// through the campaign's own <see cref="ITrainingCampaign.Checkpoint"/> against the supplied
/// <see cref="IModelStore"/>. The campaign is disposed on exit (it may hold a device-resident GPU stack).
/// </summary>
public static class CampaignRunner
{
    /// <summary>Run <paramref name="campaign"/> against <paramref name="store"/> under <paramref name="options"/>.</summary>
    public static void Run(ITrainingCampaign campaign, IModelStore store, CampaignOptions options)
    {
        ArgumentNullException.ThrowIfNull(campaign);
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);

        using (campaign)
        {
            campaign.Resume(store);

            if (options.EvalOnly)
            {
                // A campaign-specific standalone mode (e.g. a value-curve probe) takes precedence; otherwise a
                // plain evaluation. Either way: no training, no checkpoint.
                if (!campaign.TryRunStandaloneEval(store))
                    options.OnEval?.Invoke(new CampaignProgress(0, campaign.Evaluate(), IsFinal: true));
                return;
            }

            DateTime deadline = options.Now() + options.Duration;
            DateTime nextEval = options.Now() + options.FirstEvalAfter;
            long progress = 0;
            bool trainedSinceEval = false;

            while (options.Now() < deadline && !campaign.IsComplete)
            {
                progress = campaign.TrainChunk();
                trainedSinceEval = true;

                if (options.Now() >= nextEval || campaign.IsComplete)
                {
                    Report(campaign, store, options, progress, isFinal: campaign.IsComplete);
                    trainedSinceEval = false;
                    nextEval = options.Now() + options.EvalEvery;
                }
            }

            // Final eval + checkpoint unless the last chunk already triggered one (avoids a duplicate when the
            // loop exits right on a cadence eval or a hard stop). Also covers a run that trained nothing.
            if (trainedSinceEval || progress == 0)
                Report(campaign, store, options, progress, isFinal: true);
        }
    }

    private static void Report(ITrainingCampaign campaign, IModelStore store, CampaignOptions options, long progress, bool isFinal)
    {
        CampaignEval eval = campaign.Evaluate();
        campaign.Checkpoint(store);
        options.OnEval?.Invoke(new CampaignProgress(progress, eval, isFinal));
    }
}

using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;

namespace MintPlayer.AI.ReinforcementLearning.Core.Training;

/// <summary>
/// Folds consecutive single-step transitions into <b>n-step</b> transitions before they enter the replay
/// buffer: a stored transition's reward becomes the discounted sum <c>Σ_{k=0}^{n-1} γ^k·r_{t+k}</c> and its
/// next-state becomes <c>s_{t+n}</c>, so the TD target bootstraps with <c>γ^n</c> (Sutton &amp; Barto §7).
/// This propagates a sparse, far-off reward backward ~n× faster than single-step DQN — the algorithmic half of
/// breaking a long-horizon plateau where the rare high-tier payoff is otherwise near-invisible under γ^horizon.
///
/// <para><b>n = 1 is exactly single-step DQN:</b> every push emits one transition immediately, the window is
/// empty at every step boundary, and the bootstrap discount is γ — bitwise-identical to the pre-n-step trainer.</para>
///
/// <para><b>Episode ends.</b> A genuine terminal flushes the whole window with <c>terminated = true</c> (the TD
/// target zeroes the bootstrap, so the shorter effective horizon is exact). A <i>truncation</i> (time-limit) emits
/// only the full-length head window and <b>drops the ≤ n−1 trailing partial windows</b> — those would each need a
/// different γ^k bootstrap discount, and the loss (a handful of steps per truncated episode) is negligible; keeping
/// the buffer's single global γ^n bootstrap is worth it. The in-flight window is part of the persisted state
/// (<see cref="DqnTrainingState"/>), so a resumed run stays bitwise-identical.</para>
/// </summary>
internal sealed class NStepAccumulator
{
    private sealed class Pending
    {
        public required float[] Obs;
        public required int Action;
        public required float Reward;
        public required float[] NextObs;
        public required bool Terminated;
        public required bool[]? NextMask; // null => all-legal (the buffer fills `true`)
    }

    /// <summary>One ready-to-store n-step transition. <c>NextMask</c> null => all-legal.</summary>
    public readonly record struct Emit(float[] Obs, int Action, float Reward, float[] NextObs, bool Terminated, bool[]? NextMask);

    private readonly int _obsDim, _actionCount;
    private readonly List<Pending> _window = [];

    public int N { get; }
    public double Gamma { get; }

    public NStepAccumulator(int n, double gamma, int obsDim, int actionCount)
    {
        if (n < 1) throw new ArgumentOutOfRangeException(nameof(n), "n-step horizon must be ≥ 1.");
        N = n;
        Gamma = gamma;
        _obsDim = obsDim;
        _actionCount = actionCount;
    }

    /// <summary>
    /// Feed one environment transition; returns the 0+ n-step transitions it completed (one in steady state,
    /// the whole window on a terminal). <paramref name="terminated"/> is the true terminal flag (never the
    /// time-limit), <paramref name="truncated"/> the time-limit flag — mirrors <see cref="StepResult{T}"/>.
    /// </summary>
    public List<Emit> Push(ReadOnlySpan<float> obs, int action, double reward, ReadOnlySpan<float> nextObs,
        bool terminated, bool truncated, ReadOnlySpan<bool> nextMask)
    {
        _window.Add(new Pending
        {
            Obs = obs.ToArray(),
            Action = action,
            Reward = (float)reward,
            NextObs = nextObs.ToArray(),
            Terminated = terminated,
            NextMask = nextMask.IsEmpty ? null : nextMask.ToArray(),
        });

        var emits = new List<Emit>();
        if (terminated)
        {
            // The window reaches a real terminal: flush every head, each accumulating to the terminal (bootstrap 0).
            while (_window.Count > 0) emits.Add(DequeueHead(terminated: true));
        }
        else if (truncated)
        {
            // Time-limit: keep the one full-length window (it bootstraps cleanly); drop the partial tails.
            if (_window.Count >= N) emits.Add(DequeueHead(terminated: false));
            _window.Clear();
        }
        else if (_window.Count >= N)
        {
            emits.Add(DequeueHead(terminated: false)); // steady state: one n-step transition completes per step
        }
        return emits;
    }

    // Builds the n-step transition for the current head over the whole live window, then removes the head.
    private Emit DequeueHead(bool terminated)
    {
        double r = 0, g = 1;
        foreach (var p in _window) { r += g * p.Reward; g *= Gamma; }
        var head = _window[0];
        var tail = _window[^1];
        _window.RemoveAt(0);
        return new Emit(head.Obs, head.Action, (float)r, tail.NextObs, terminated, tail.NextMask);
    }

    // ── persistence (embedded in DqnTrainingState; self-describing n/γ so resume can validate against options) ──

    public void Save(BinaryWriter writer)
    {
        writer.Write(N);
        writer.Write(Gamma);
        writer.Write(_window.Count);
        foreach (var p in _window)
        {
            CheckpointFormat.WriteFloats(writer, p.Obs);
            writer.Write(p.Action);
            writer.Write(p.Reward);
            CheckpointFormat.WriteFloats(writer, p.NextObs);
            writer.Write(p.Terminated);
            writer.Write(p.NextMask is not null);
            if (p.NextMask is not null) CheckpointFormat.WriteBools(writer, p.NextMask);
        }
    }

    public static NStepAccumulator Load(BinaryReader reader, int obsDim, int actionCount)
    {
        int n = reader.ReadInt32();
        double gamma = reader.ReadDouble();
        var acc = new NStepAccumulator(n, gamma, obsDim, actionCount);
        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            var obs = CheckpointFormat.ReadFloats(reader);
            int action = reader.ReadInt32();
            float reward = reader.ReadSingle();
            var nextObs = CheckpointFormat.ReadFloats(reader);
            bool terminated = reader.ReadBoolean();
            bool[]? mask = reader.ReadBoolean() ? CheckpointFormat.ReadBools(reader) : null;
            acc._window.Add(new Pending { Obs = obs, Action = action, Reward = reward, NextObs = nextObs, Terminated = terminated, NextMask = mask });
        }
        return acc;
    }
}

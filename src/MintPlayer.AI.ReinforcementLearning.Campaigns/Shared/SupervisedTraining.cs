using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// A rolling mean of the three per-batch supervised-training losses (cross-entropy, Huber, accuracy) accumulated
/// across the rounds between two evaluations. Every imitation/policy campaign kept the same four fields and the same
/// guarded divide-and-reset; this hides that bookkeeping behind <see cref="Add"/> and <see cref="MeanAndReset"/>.
/// </summary>
public struct TrainWindow
{
    private double _ce, _huber, _acc;
    private long _count;

    public void Add(double ce, double huber, double acc)
    {
        _ce += ce;
        _huber += huber;
        _acc += acc;
        _count++;
    }

    /// <summary>The mean of each metric since the last reset (0 when no batch ran), then clears the window.</summary>
    public (double Ce, double Huber, double Acc) MeanAndReset()
    {
        var mean = _count > 0 ? (_ce / _count, _huber / _count, _acc / _count) : (0d, 0d, 0d);
        _ce = _huber = _acc = 0;
        _count = 0;
        return mean;
    }
}

/// <summary>
/// Load/save an <see cref="Adam"/> optimizer's moment estimates alongside a supervised net in the model store. The
/// moments are keyed to the net's parameter tensors, so both sides pass <c>net.Parameters()</c>. Hides the
/// <see cref="BinaryReader"/>/<see cref="BinaryWriter"/> + UTF-8 + <c>leaveOpen</c> dance the imitation/policy
/// campaigns each copied, and defines the "no stored optimizer yet" case out of existence (a fresh Adam is
/// returned) so the caller never branches on it.
/// </summary>
public static class AdamState
{
    /// <summary>Restore Adam's moments from <paramref name="id"/> if present (re-pinning the CLI learning rate over
    /// the stored schedule position, and logging it), else return a fresh optimizer over <paramref name="parameters"/>.</summary>
    public static Adam LoadOrInit(IModelStore store, string environmentId, string id,
        IEnumerable<Tensor> parameters, float learningRate, Action<string> log)
    {
        using var stream = store.TryOpenRead(environmentId, id);
        if (stream is null) return new Adam(parameters, learningRate);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        var adam = AdamCheckpoint.Read(parameters, reader);
        adam.LearningRate = learningRate; // CLI overrides the stored schedule position
        log($"resumed Adam state (lr set to {learningRate:E1})");
        return adam;
    }

    /// <summary>Persist Adam's moments under <paramref name="id"/>.</summary>
    public static void Save(IModelStore store, string environmentId, string id, Adam adam)
        => store.Save(environmentId, id, s =>
        {
            using var writer = new BinaryWriter(s, Encoding.UTF8, leaveOpen: true);
            AdamCheckpoint.Write(adam, writer);
        });
}

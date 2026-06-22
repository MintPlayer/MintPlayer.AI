using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using Tensor = MintPlayer.AI.ReinforcementLearning.Core.Numerics.Tensor;

/// <summary>
/// Shared supervised training step for the two-headed <see cref="CubePolicyNet"/>, used identically by the
/// Kociemba-imitation (<see cref="CubeImitationCampaign"/>) and teacher-free EfficientCube
/// (<see cref="CubeEfficientCampaign"/>) campaigns: both label one optimal/reversing move per state and regress
/// distance-to-go, so the loss (CE on the move + Huber on distance) and the train loop are the same. The only
/// difference is how the labeled states are produced — which stays in each campaign.
/// </summary>
internal static class CubePolicyTraining
{
    /// <summary>
    /// One supervised batch: forward, CE(next move) + Huber(distance) loss, gradient-clipped Adam step. Returns
    /// the batch's CE, Huber and top-1 move accuracy (argmax == labeled action). The batch is samples
    /// [<paramref name="offset"/>, offset+<paramref name="batch"/>) of <paramref name="samples"/>.
    /// </summary>
    public static (double Ce, double Huber, double Acc) TrainStep(
        CubePolicyNet net, Adam adam, List<CubeOracle.LabeledState> samples, int offset, int batch)
    {
        var obs = new float[batch * RubiksCubeEnv.ObservationSize];
        var weights = new float[batch * RubiksCubeEnv.ActionCount];
        var targets = new float[batch];
        for (int i = 0; i < batch; i++)
        {
            var s = samples[offset + i];
            RubiksCubeEnv.WriteObservation(FaceletCube.FromFacelets(s.Facelets),
                obs.AsSpan(i * RubiksCubeEnv.ObservationSize, RubiksCubeEnv.ObservationSize));
            weights[i * RubiksCubeEnv.ActionCount + s.Action] = 1f;
            targets[i] = s.DistanceToGo / CubePolicyNet.DistanceScale;
        }

        var (logits, value) = net.Forward(new Tensor(obs, batch, RubiksCubeEnv.ObservationSize));
        var logProbs = logits.LogSoftmax();
        var ce = logProbs.Mul(new Tensor(weights, batch, RubiksCubeEnv.ActionCount)).Sum().MulScalar(-1f / batch);
        var huber = value.Reshape(batch).HuberLoss(new Tensor(targets, batch));
        var loss = ce.Add(huber);

        adam.ZeroGrad();
        loss.Backward();
        adam.ClipGradNorm(5f);
        adam.Step();

        int correct = 0;
        for (int i = 0; i < batch; i++)
        {
            int argmax = 0;
            for (int a = 1; a < RubiksCubeEnv.ActionCount; a++)
                if (logProbs.Data[i * RubiksCubeEnv.ActionCount + a] > logProbs.Data[i * RubiksCubeEnv.ActionCount + argmax])
                    argmax = a;
            if (argmax == samples[offset + i].Action) correct++;
        }
        return (ce.Data[0], huber.Data[0], correct / (double)batch);
    }

    /// <summary>In-place Fisher–Yates shuffle with the campaign's deterministic RNG (reproducible data order).</summary>
    public static void Shuffle<T>(IList<T> list, Xoshiro256StarStar rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.NextInt(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}

using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// CS2 gate (docs/prd/FRUITCAKE_CLIENT_SIDE_AI_PRD.md): the single-source inference forward pass now lives in
/// fruitcake_solver.pg as <c>PgDuelingNet</c> (f64), so C# serving and the browser run bit-identical inference
/// from one source. This pins that the generated f64 forward reproduces the SDK's float32
/// <see cref="DuelingQNet.Forward"/> — argmax exact, values within float32-vs-float64 GEMM tolerance — for a net
/// with FruitCake's shape. (C#↔TS byte-identity of PgDuelingNet comes free from the transpiler.)
/// PgDuelingNet is the internal generated type (global namespace), visible via InternalsVisibleTo.
/// </summary>
public class PolyglotNetParityTests
{
    [Fact]
    public void CoreDuelingForward_MatchesSdkDuelingQNet()
    {
        const int inputSize = FruitCakeEnv.ObservationSize; // 89
        const int actions = FruitCakeEnv.ColumnCount;       // 14
        int[] hidden = [256, 256];

        var rng = new Xoshiro256StarStar(12345);
        var sdk = new DuelingQNet(inputSize, hidden, actions, rng, noisy: false);

        // Build the .pg net from the SDK net's parameters. Parameters() order for a plain net:
        // trunk0.W, trunk0.b, trunk1.W, trunk1.b, value.W, value.b, adv.W, adv.b.
        var ps = sdk.Parameters().ToList();
        var trunkWFlat = new List<double>();
        var trunkBFlat = new List<double>();
        for (int l = 0; l < hidden.Length; l++)
        {
            foreach (var f in ps[l * 2].Data) trunkWFlat.Add(f);
            foreach (var f in ps[l * 2 + 1].Data) trunkBFlat.Add(f);
        }
        int h = hidden.Length * 2;
        List<double> ToD(int k) => ps[k].Data.Select(f => (double)f).ToList();
        var pg = new PgDuelingNet(inputSize, actions, hidden.ToList(),
            trunkWFlat, trunkBFlat, ToD(h), ToD(h + 1), ToD(h + 2), ToD(h + 3));

        double maxDiff = 0;
        for (int t = 0; t < 25; t++)
        {
            var obsF = new float[inputSize];
            var obsD = new List<double>(inputSize);
            for (int i = 0; i < inputSize; i++)
            {
                float val = ((i * 7 + t * 13) % 100) / 100f; // varied, deterministic [0,1)
                obsF[i] = val;
                obsD.Add(val);
            }

            var sdkQ = sdk.Forward(new Tensor(obsF, 1, inputSize)).Data;
            var pgQ = pg.forward(obsD);

            Assert.Equal(actions, pgQ.Count);
            Assert.Equal(ArgMaxF(sdkQ), ArgMaxD(pgQ)); // the decision must be identical
            for (int a = 0; a < actions; a++)
                maxDiff = Math.Max(maxDiff, Math.Abs(sdkQ[a] - pgQ[a]));
        }

        // float32 (SDK GEMM) vs f64 (.pg) accumulation over 256-wide dot products — a small, expected drift.
        Assert.True(maxDiff < 1e-2, $"max |Q_sdk - Q_pg| = {maxDiff} exceeds tolerance");
    }

    private static int ArgMaxF(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static int ArgMaxD(List<double> v)
    {
        int best = 0;
        for (int i = 1; i < v.Count; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}

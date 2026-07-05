using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.FruitCake;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// CS2 + CS4 gates (docs/prd/FRUITCAKE_CLIENT_SIDE_AI_PRD.md): the single-source inference path (net forward +
/// depth-3 search) now lives in fruitcake_solver.pg (f64), so C# serving and the browser run bit-identical
/// inference from one source. These pin that the generated f64 code reproduces the trusted SDK implementations:
/// <c>PgDuelingNet.forward</c> vs <see cref="DuelingQNet.Forward"/> (CS2), and <c>PgFruitCakeWorld.chooseColumn</c>
/// vs <see cref="FruitCakeSearch"/> (CS4). PgDuelingNet/PgFruitCakeWorld are the internal generated types (global
/// namespace), visible via InternalsVisibleTo. C#↔TS byte-identity comes free from the transpiler.
/// </summary>
public class PolyglotNetParityTests
{
    private const int InputSize = FruitCakeEnv.ObservationSize; // 89
    private const int Actions = FruitCakeEnv.ColumnCount;       // 14
    private static readonly int[] Hidden = [256, 256];

    // Build the .pg net from a fresh SDK net's parameters (Parameters() order for a plain net:
    // trunk0.W, trunk0.b, trunk1.W, trunk1.b, value.W, value.b, adv.W, adv.b).
    private static PgDuelingNet MakePgNet(Xoshiro256StarStar rng, out DuelingQNet sdk)
    {
        sdk = new DuelingQNet(InputSize, Hidden, Actions, rng, noisy: false);
        var ps = sdk.Parameters().ToList();
        var trunkWFlat = new List<double>();
        var trunkBFlat = new List<double>();
        for (int l = 0; l < Hidden.Length; l++)
        {
            foreach (var f in ps[l * 2].Data) trunkWFlat.Add(f);
            foreach (var f in ps[l * 2 + 1].Data) trunkBFlat.Add(f);
        }
        int h = Hidden.Length * 2;
        List<double> ToD(int k) => ps[k].Data.Select(f => (double)f).ToList();
        return new PgDuelingNet(InputSize, Actions, Hidden.ToList(),
            trunkWFlat, trunkBFlat, ToD(h), ToD(h + 1), ToD(h + 2), ToD(h + 3));
    }

    [Fact]
    public void CoreDuelingForward_MatchesSdkDuelingQNet()
    {
        var pg = MakePgNet(new Xoshiro256StarStar(12345), out var sdk);

        double maxDiff = 0;
        for (int t = 0; t < 25; t++)
        {
            var obsF = new float[InputSize];
            var obsD = new List<double>(InputSize);
            for (int i = 0; i < InputSize; i++)
            {
                float val = ((i * 7 + t * 13) % 100) / 100f; // varied, deterministic [0,1)
                obsF[i] = val;
                obsD.Add(val);
            }

            var sdkQ = sdk.Forward(new Tensor(obsF, 1, InputSize)).Data;
            var pgQ = pg.forward(obsD);

            Assert.Equal(Actions, pgQ.Count);
            Assert.Equal(ArgMaxF(sdkQ), ArgMaxD(pgQ)); // the decision must be identical
            for (int a = 0; a < Actions; a++)
                maxDiff = Math.Max(maxDiff, Math.Abs(sdkQ[a] - pgQ[a]));
        }

        // float32 (SDK GEMM) vs f64 (.pg) accumulation over 256-wide dot products — a small, expected drift.
        Assert.True(maxDiff < 1e-2, $"max |Q_sdk - Q_pg| = {maxDiff} exceeds tolerance");
    }

    [Fact]
    public void CoreSearch_MatchesCsFruitCakeSearch_SameColumn()
    {
        var pg = MakePgNet(new Xoshiro256StarStar(999), out _);

        // Build the identical board on the core (for the .pg search) and the facade (for the C# reference search),
        // seeding both with the same float-cast coordinates so their f64 states are bit-identical.
        (int Tier, double X)[] script =
            [(1, 305), (2, 312), (1, 260), (3, 410), (1, 300), (1, 306), (2, 258), (4, 360), (1, 500), (1, 506), (5, 200)];
        var core = new PgFruitCakeWorld();
        var facade = new FruitCakeWorld();
        foreach (var (t, x) in script)
        {
            core.spawnFruit(t, (float)x, 90f);
            core.settleAfterDrop(30.0, 8, 600, 1.0 / 60.0);
            facade.SpawnFruit(t, (float)x, 90f);
            facade.SettleAfterDrop(30f, 8, 600);
        }

        // C# reference search with a leaf identical to the .pg's inlined leaf: the same PgDuelingNet over the same
        // raw f64 observation. With identical physics (single-source core) and identical leaf, the two searches
        // must choose the exact same column — this pins the ported search LOGIC (top-k, expectimax, pruning, argmax).
        double Leaf(FruitCakeWorld w)
        {
            double sum = 0;
            for (int d = 1; d <= FruitCatalog.MaxDroppableTier; d++)
                sum += pg.forward([.. w.BuildObservationF64(d, d)]).Max();
            return sum / FruitCatalog.MaxDroppableTier;
        }
        var cs = new FruitCakeSearch(Leaf) { MaxDepth = 3, TopK = 5, TopK2 = 2 };

        foreach (var (cur, next) in new[] { (1, 2), (3, 5), (5, 1), (2, 4) })
        {
            int pgCol = core.chooseColumn(pg, cur, next, 3, 5, 2);
            int csCol = cs.ChooseColumn(facade, cur, next);
            Assert.Equal(csCol, pgCol);
        }
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

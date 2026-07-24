using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.CrazyFruits;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The M49.5 browser-inference gate (the draughts M47.5 pattern): the single-source net forward in
/// <c>crazyfruits_solver.pg</c> (transpiled to C# here, to TypeScript for the web client) must match the
/// training stack through the REAL .ckpt bytes. The byte-level parser here is the line-for-line reference
/// for <c>crazyfruits-net.ts</c> (magic → kind "dueling-q" → version → dims → noisy byte → params in
/// Parameters() order). The SDK net accumulates in float32, the generated one in f64 over float32 weights,
/// so they agree to an f32 tolerance — and since the TS twin runs the identical f64 code on the identical
/// parse, browser inference matches by construction.
/// </summary>
public class CrazyFruitsNetParityTests
{
    [Fact]
    public void GeneratedNet_MatchesDuelingQNetForward_ThroughRealCheckpointBytes()
    {
        var net = new DuelingQNet(CrazyFruitsEnv.ObservationSize, [64, 64], CrazyFruitsEnv.ActionCount, new Xoshiro256StarStar(999));
        using var ms = new MemoryStream();
        DuelingQNetCheckpoint.Save(net, ms);
        ms.Position = 0;
        var pg = ParseLikeTheBrowser(ms);

        var core = new PgCrazyFruits();
        core.reset(42);
        var obsF64 = core.buildObservation();
        var obs = new float[obsF64.Count];
        for (int i = 0; i < obs.Length; i++) obs[i] = (float)obsF64[i];

        var q = net.Forward(new Tensor(obs, 1, obs.Length));
        var pgQ = pg.forward([.. obs.Select(f => (double)f)]);

        Assert.Equal(CrazyFruitsEnv.ActionCount, pgQ.Count);
        double maxDiff = 0;
        for (int a = 0; a < CrazyFruitsEnv.ActionCount; a++)
            maxDiff = Math.Max(maxDiff, Math.Abs(q.Data[a] - pgQ[a]));
        Assert.True(maxDiff < 2e-3, $"max Q diff {maxDiff} exceeds the f32 tolerance");

        // The masked argmax the browser director runs must return a LEGAL action.
        int action = core.netAction(pg);
        Assert.True(core.swapProducesMatch(action), $"netAction returned illegal action {action}");
    }

    // Line-for-line mirror of crazyfruits-net.ts parseDuelingQNet — keep the two in sync.
    private static PgCfDuelingNet ParseLikeTheBrowser(Stream stream)
    {
        using var r = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        List<double> ReadFloats()
        {
            int n = r.ReadInt32();
            var a = new List<double>(n);
            for (int i = 0; i < n; i++) a.Add(r.ReadSingle()); // float32 → exact f64, like the DataView read
            return a;
        }

        Assert.Equal(0x434E4C52u, r.ReadUInt32());       // "RLNC"
        Assert.Equal("dueling-q", r.ReadString());
        int version = r.ReadInt32();
        int inputSize = r.ReadInt32();
        int hiddenCount = r.ReadInt32();
        var hidden = new List<int>();
        for (int i = 0; i < hiddenCount; i++) hidden.Add(r.ReadInt32());
        int actions = r.ReadInt32();
        bool noisy = version >= 2 && r.ReadByte() != 0;

        var trunkW = new List<double>();
        var trunkB = new List<double>();
        for (int l = 0; l < hiddenCount; l++)
        {
            trunkW.AddRange(ReadFloats());
            trunkB.AddRange(ReadFloats());
        }

        List<double> valueW, valueB, advW, advB;
        if (!noisy)
        {
            valueW = ReadFloats(); valueB = ReadFloats();
            advW = ReadFloats(); advB = ReadFloats();
        }
        else
        {
            valueW = ReadFloats(); ReadFloats(); valueB = ReadFloats(); ReadFloats();
            advW = ReadFloats(); ReadFloats(); advB = ReadFloats(); ReadFloats();
        }

        return new PgCfDuelingNet(inputSize, actions, hidden, trunkW, trunkB, valueW, valueB, advW, advB);
    }
}

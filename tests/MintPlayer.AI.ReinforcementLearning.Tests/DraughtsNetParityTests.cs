using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Draughts;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The M47.5 browser-inference gate (chess M40.2/M42.4 pattern): the single-source net + MCTS in
/// <c>draughts_solver.pg</c> (transpiled to C# here, to TypeScript for the web client) must match the
/// training stack. The conv forward is compared against <see cref="ConvResidualPolicyValueNet"/> through
/// the REAL .ckpt bytes — the byte-level parser here doubles as the reference for <c>draughts-net.ts</c>
/// (magic → kind "selfplay-pv-conv" → version → WriteInts dims → params in Parameters() order, per-block
/// layers concatenated). The C# net accumulates in float32, the generated one in f64, so they agree to an
/// f32 tolerance — all the browser AI needs.
/// </summary>
public class DraughtsNetParityTests
{
    [Fact]
    public void Generated_conv_net_matches_ConvResidualPolicyValueNet_forward_within_f32_tolerance()
    {
        var game = new DraughtsGame(DraughtsVariant.English8);
        var net = new ConvResidualPolicyValueNet(planes: 5, boardH: 8, boardW: 8, actions: game.PolicySize,
            filters: 8, blocks: 2, new Xoshiro256StarStar(999));

        using var ms = new MemoryStream();
        net.Save(ms, "selfplay-pv-conv");
        ms.Position = 0;
        var pg = LoadPgConv(ms, game.PolicySize);

        var obs = new float[game.ObservationSize];
        game.WriteObservation(DraughtsState.StartPosition(DraughtsVariant.English8), obs);

        var (logits, value) = net.Forward(new Tensor(obs, 1, game.ObservationSize));
        var pgOut = pg.forward([.. obs.Select(f => (double)f)]);

        Assert.Equal(game.PolicySize, pgOut.logits.Count);
        double maxLogitDiff = 0;
        for (int m = 0; m < game.PolicySize; m++)
            maxLogitDiff = Math.Max(maxLogitDiff, Math.Abs(logits.Data[m] - pgOut.logits[m]));

        Assert.True(maxLogitDiff < 2e-3, $"max logit diff {maxLogitDiff} exceeds tolerance");
        Assert.True(Math.Abs(value.Data[0] - pgOut.value) < 2e-3, $"value diff {Math.Abs(value.Data[0] - pgOut.value)}");
    }

    [Fact]
    public void Generated_writeObservation_matches_DraughtsGame_WriteObservation()
    {
        // Both movers over a random playout — exercises the piece planes, the 180° rotation, and the clock plane.
        foreach (var variant in new[] { DraughtsVariant.International10, DraughtsVariant.English8 })
        {
            var game = new DraughtsGame(variant);
            var rng = new Random(31);
            var state = game.Root();
            for (int ply = 0; ply < 40 && game.Result(state) == Core.Planning.GameResult.Ongoing; ply++)
            {
                var expected = new float[game.ObservationSize];
                game.WriteObservation(state, expected);
                var actual = state.Core.writeObservation();

                Assert.Equal(game.ObservationSize, actual.Count);
                for (int i = 0; i < game.ObservationSize; i++)
                    Assert.Equal(expected[i], (float)actual[i]);

                var legal = game.LegalMoves(state);
                state = game.Apply(state, legal[rng.Next(legal.Count)]);
            }
        }
    }

    [Fact]
    public void Generated_mcts_returns_a_valid_legal_move_distribution()
    {
        // The inference MCTS (PgDraughtsMcts) is single-source code the forward-parity test doesn't exercise:
        // search must return a distribution over the action space that sums to 1 with mass ONLY on legal moves.
        var game = new DraughtsGame(DraughtsVariant.International10);
        var net = new ConvResidualPolicyValueNet(5, 10, 10, game.PolicySize, filters: 4, blocks: 1, new Xoshiro256StarStar(7));
        using var ms = new MemoryStream();
        net.Save(ms, "selfplay-pv-conv");
        ms.Position = 0;
        var pg = LoadPgConv(ms, game.PolicySize);

        var root = DraughtsState.StartPosition(DraughtsVariant.International10).Core;
        var legal = new HashSet<int>(root.legalMoveIndices());
        Assert.Equal(9, legal.Count); // 9 opening moves (perft d1)

        var pi = PgDraughtsMcts.search(pg, root, 16, 1.25);
        Assert.Equal(game.PolicySize, pi.Count);
        double sum = 0;
        for (int i = 0; i < pi.Count; i++)
        {
            sum += pi[i];
            if (pi[i] > 0) Assert.Contains(i, legal); // no mass on illegal moves
        }
        Assert.True(Math.Abs(sum - 1.0) < 1e-9, $"visit distribution sums to {sum}, expected 1");
        Assert.Contains(PgDraughtsMcts.chooseMove(pg, root, 16, 1.25), legal);
    }

    // Mirror of ConvResidualPolicyValueNet.Save (kind "selfplay-pv-conv", version 1): magic → kind → version →
    // WriteInts [planes,H,W,filters,blocks] (leading int32 count=5) → params in Parameters() order (each count+floats).
    // Per-block layers (conv1/norm1/conv2/norm2 × blocks) concatenate into the flat per-role arrays. This is the
    // byte-level reference draughts-net.ts implements in the browser. `actions` comes from the environment.
    private static PgDraughtsNet LoadPgConv(Stream stream, int actions)
    {
        using var r = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        Assert.Equal(0x434E4C52u, r.ReadUInt32());       // "RLNC"
        Assert.Equal("selfplay-pv-conv", r.ReadString());
        Assert.Equal(1, r.ReadInt32());                  // version
        Assert.Equal(5, r.ReadInt32());                  // dim count
        int planes = r.ReadInt32(), h = r.ReadInt32(), w = r.ReadInt32(), filters = r.ReadInt32(), blocks = r.ReadInt32();

        List<double> ReadFloats()
        {
            int n = r.ReadInt32();
            var list = new List<double>(n);
            for (int i = 0; i < n; i++) list.Add(r.ReadSingle());
            return list;
        }

        var conv = new PgDraughtsConvNet(planes, h, w, filters, blocks, actions)
        {
            stemW = ReadFloats(),
            stemB = ReadFloats(),
            stemNG = ReadFloats(),
            stemNB = ReadFloats(),
            b1W = [], b1B = [], n1G = [], n1B = [], b2W = [], b2B = [], n2G = [], n2B = [],
        };
        for (int i = 0; i < blocks; i++)
        {
            conv.b1W.AddRange(ReadFloats()); conv.b1B.AddRange(ReadFloats());
            conv.n1G.AddRange(ReadFloats()); conv.n1B.AddRange(ReadFloats());
            conv.b2W.AddRange(ReadFloats()); conv.b2B.AddRange(ReadFloats());
            conv.n2G.AddRange(ReadFloats()); conv.n2B.AddRange(ReadFloats());
        }
        conv.pConvW = ReadFloats(); conv.pConvB = ReadFloats();
        conv.pNG = ReadFloats(); conv.pNB = ReadFloats();
        conv.pHeadW = ReadFloats(); conv.pHeadB = ReadFloats();
        conv.vConvW = ReadFloats(); conv.vConvB = ReadFloats();
        conv.vNG = ReadFloats(); conv.vNB = ReadFloats();
        conv.vHidW = ReadFloats(); conv.vHidB = ReadFloats();
        conv.vHeadW = ReadFloats(); conv.vHeadB = ReadFloats();
        return PgDraughtsNet.withConv(conv);
    }
}

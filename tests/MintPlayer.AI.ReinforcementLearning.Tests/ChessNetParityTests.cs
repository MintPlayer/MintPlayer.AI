using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Chess;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// M40.2 gate: the single-source browser inference net (<c>chess_solver.pg</c> → <c>PgPolicyValueNet</c>, transpiled
/// to C# here and to TypeScript for the web client) must compute the SAME forward pass as the training net
/// (<see cref="PolicyValueNet"/>). We serialize a real net through its actual <c>.ckpt</c> path, parse the bytes into
/// the generated <c>PgPolicyValueNet</c> exactly as <c>chess-net.ts</c> does in the browser, and compare logits +
/// value on a fixed position. The C# net accumulates in float32 and the generated net in f64 (loaded from the same
/// float32 weights), so they agree only to within an f32 tolerance — which is all the browser AI needs.
/// <para>The byte-level parser here doubles as the reference for <c>chess-net.ts</c>: same magic/kind/order.</para>
/// </summary>
public class ChessNetParityTests
{
    private const int InputSize = 18 * 64;              // 1152
    private static readonly int Actions = ChessMoveEncoding.Size; // 4672

    [Fact]
    public void Generated_net_matches_PolicyValueNet_forward_within_f32_tolerance()
    {
        int[] hidden = [24, 16];
        var net = new PolicyValueNet(InputSize, hidden, Actions, new Xoshiro256StarStar(12345));

        // Round-trip through the real checkpoint format, then rebuild the generated net from the bytes.
        using var ms = new MemoryStream();
        net.Save(ms, "selfplay-pv");
        ms.Position = 0;
        var pg = LoadPg(ms, InputSize, Actions);

        // Same start-position observation into both nets.
        var obs = new float[InputSize];
        new ChessGame().WriteObservation(ChessState.StartPosition(), obs);

        var (logits, value) = net.Forward(new Tensor(obs, 1, InputSize));
        var pgOut = pg.forward([.. obs.Select(f => (double)f)]);

        Assert.Equal(Actions, pgOut.logits.Count);
        double maxLogitDiff = 0;
        for (int m = 0; m < Actions; m++)
            maxLogitDiff = Math.Max(maxLogitDiff, Math.Abs(logits.Data[m] - pgOut.logits[m]));

        Assert.True(maxLogitDiff < 1e-3, $"max logit diff {maxLogitDiff} exceeds f32 tolerance");
        Assert.True(Math.Abs(value.Data[0] - pgOut.value) < 1e-3, $"value diff {Math.Abs(value.Data[0] - pgOut.value)}");
    }

    [Fact]
    public void Generated_mcts_returns_a_valid_legal_move_distribution()
    {
        // The inference MCTS (PgChessMcts) is new single-source code the forward-parity test doesn't exercise.
        // Runtime check via the generated C# twin (mirrors the browser TS): search must return a distribution over
        // the 4672 space that sums to 1 and puts mass ONLY on legal moves; chooseMove must pick a legal move.
        var net = new PolicyValueNet(InputSize, [16], Actions, new Xoshiro256StarStar(7));
        using var ms = new MemoryStream();
        net.Save(ms, "selfplay-pv");
        ms.Position = 0;
        var pg = LoadPg(ms, InputSize, Actions);

        var root = ChessState.StartPosition().Core;
        var legal = new HashSet<int>(root.legalMoveIndices());
        Assert.Equal(20, legal.Count); // 20 legal moves from the start position

        var pi = PgChessMcts.search(pg, root, 24, 1.25);
        Assert.Equal(Actions, pi.Count);
        double sum = 0;
        for (int i = 0; i < pi.Count; i++)
        {
            sum += pi[i];
            if (pi[i] > 0) Assert.Contains(i, legal); // no mass on illegal moves
        }
        Assert.True(Math.Abs(sum - 1.0) < 1e-9, $"visit distribution sums to {sum}, expected 1");
        Assert.Contains(PgChessMcts.chooseMove(pg, root, 24, 1.25), legal);
    }

    [Fact]
    public void Generated_writeObservation_matches_ChessGame_WriteObservation()
    {
        // The single-source observation (used by the browser net) must be identical to the training one.
        foreach (var fen in new[]
        {
            ChessFen.StartFen,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete (castling rights)
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",       // en-passant target set
        })
        {
            var state = ChessFen.Parse(fen);
            var expected = new float[InputSize];
            new ChessGame().WriteObservation(state, expected);
            var actual = state.Core.writeObservation();

            Assert.Equal(InputSize, actual.Count);
            for (int i = 0; i < InputSize; i++)
                Assert.Equal(expected[i], (float)actual[i]);
        }
    }

    // Mirror of chess-net.ts / PolicyValueNet.Save (kind "selfplay-pv"): magic → kind → version → trunk widths →
    // per layer [trunk…, policyHead, valueHead] (wCount+floats, bCount+floats). inputSize/actions supplied externally.
    private static PgPolicyValueNet LoadPg(Stream stream, int inputSize, int actions)
    {
        using var r = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        Assert.Equal(0x434E4C52u, r.ReadUInt32());       // "RLNC"
        Assert.Equal("selfplay-pv", r.ReadString());
        Assert.Equal(2, r.ReadInt32());                  // version

        int trunkCount = r.ReadInt32();
        var hidden = new List<int>(trunkCount);
        for (int i = 0; i < trunkCount; i++) hidden.Add(r.ReadInt32());

        List<double> ReadFloats()
        {
            int n = r.ReadInt32();
            var list = new List<double>(n);
            for (int i = 0; i < n; i++) list.Add(r.ReadSingle());
            return list;
        }

        var trunkW = new List<double>();
        var trunkB = new List<double>();
        for (int l = 0; l < trunkCount; l++) { trunkW.AddRange(ReadFloats()); trunkB.AddRange(ReadFloats()); }
        var policyW = ReadFloats(); var policyB = ReadFloats();
        var valueW = ReadFloats(); var valueB = ReadFloats();

        return new PgPolicyValueNet(inputSize, actions, hidden, trunkW, trunkB, policyW, policyB, valueW, valueB);
    }
}

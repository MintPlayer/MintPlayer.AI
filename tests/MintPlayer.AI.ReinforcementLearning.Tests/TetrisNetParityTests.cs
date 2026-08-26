using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Tetris;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

/// <summary>
/// The M54.5 browser-inference gate (the Crazy Fruits M49.5 pattern): the single-source net forward in
/// <c>tetris_solver.pg</c> (transpiled to C# here, to TypeScript for the web client) must match the
/// training stack through the REAL .ckpt bytes. The parser under test is the facade's
/// <see cref="TetrisBoard.LoadNet"/> — the line-for-line reference for <c>tetris-net.ts</c>. The SDK net
/// accumulates in float32, the generated one in f64 over float32 weights, so they agree to an f32
/// tolerance — and since the TS twin runs the identical f64 code on the identical parse, browser
/// inference matches by construction.
/// </summary>
public class TetrisNetParityTests
{
    [Fact]
    public void GeneratedNet_MatchesDuelingQNetForward_ThroughRealCheckpointBytes()
    {
        var net = new DuelingQNet(TetrisEnv.ObservationSize, [64, 64], TetrisEnv.ActionCount, new Xoshiro256StarStar(999));
        using var ms = new MemoryStream();
        DuelingQNetCheckpoint.Save(net, ms);

        var board = new TetrisBoard();
        board.Reset(42);
        ms.Position = 0;
        Assert.True(board.LoadNet(ms));
        ms.Position = 0;
        var pg = TetrisBoard.ParseDuelingQCheckpoint(ms)!;

        var obs = board.BuildObservation();
        var q = net.Forward(new Tensor(obs, 1, obs.Length));
        var pgQ = pg.forward([.. obs.Select(f => (double)f)]);

        Assert.Equal(TetrisEnv.ActionCount, pgQ.Count);
        double maxDiff = 0;
        for (int a = 0; a < TetrisEnv.ActionCount; a++)
            maxDiff = Math.Max(maxDiff, Math.Abs(q.Data[a] - pgQ[a]));
        Assert.True(maxDiff < 2e-3, $"max Q diff {maxDiff} exceeds the f32 tolerance");

        // The masked argmax the browser director runs must return a LEGAL action — and so must the
        // net+search tier (the beam rollout must never surface an illegal placement).
        int action = board.NetAction();
        Assert.True(board.IsLegal(action), $"NetAction returned illegal action {action}");
        int searchAction = board.NetSearchAction(8);
        Assert.True(board.IsLegal(searchAction), $"NetSearchAction returned illegal action {searchAction}");
    }

    [Fact]
    public void LoadNet_RejectsAWrongWidthCheckpoint()
    {
        // The stale-ckpt guard: a net trained on an older observation layout must be refused, not crash.
        var stale = new DuelingQNet(100, [16], TetrisEnv.ActionCount, new Xoshiro256StarStar(1));
        using var ms = new MemoryStream();
        DuelingQNetCheckpoint.Save(stale, ms);
        ms.Position = 0;
        var board = new TetrisBoard();
        board.Reset(1);
        Assert.False(board.LoadNet(ms));
        Assert.Equal(-1, board.NetAction());
    }
}

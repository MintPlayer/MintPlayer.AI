using MintPlayer.AI.ReinforcementLearning.Campaigns;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;
using Xunit;
using Xunit.Abstractions;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public sealed class ZzTempDistributionProbe(ITestOutputHelper output)
{
    private static (int x, int y) PlayerCell(BlockDudeBoard b) => (b.PlayerX, b.PlayerY);

    private static List<(int x, int y)> Doors(BlockDudeBoard b)
    {
        var list = new List<(int, int)>();
        for (int y = 0; y < b.Height; y++)
            for (int x = 0; x < b.Width; x++)
                if (b.TileAt(x, y) == BlockDudeTile.Door) list.Add((x, y));
        return list;
    }

    private static (int x, int y) NearestDoor(BlockDudeBoard b)
    {
        var (px, py) = PlayerCell(b);
        return Doors(b).OrderBy(d => Math.Abs(d.x - px) + Math.Abs(d.y - py)).First();
    }

    /// <summary>Walk-only reachability (Left/Right/Climb, never Grab): where can the dude get with no block use,
    /// and which blocks can he pick up from there.</summary>
    private static (bool DoorReachable, HashSet<int> GrabbableBlockX, int Visited) WalkOnly(BlockDudeBoard start, int cap = 200_000)
    {
        var seen = new HashSet<int>();
        var q = new Queue<BlockDudeBoard>();
        var grab = new HashSet<int>();
        bool won = false;
        seen.Add(start.StateHash);
        q.Enqueue(start);
        int visited = 0;
        while (q.Count > 0 && visited < cap)
        {
            var cur = q.Dequeue();
            visited++;
            if (cur.Won) { won = true; continue; }
            if (!cur.Carrying && cur.IsLegal(BlockDudeAction.Grab))
            {
                int bx = cur.FacingRight ? cur.PlayerX + 1 : cur.PlayerX - 1;
                grab.Add(bx);
            }
            foreach (var a in new[] { BlockDudeAction.Left, BlockDudeAction.Right, BlockDudeAction.Climb })
            {
                if (!cur.IsLegal(a)) continue;
                var next = cur.Apply(a);
                if (seen.Add(next.StateHash)) q.Enqueue(next);
            }
        }
        return (won, grab, visited);
    }

    private static int BlockCount(BlockDudeBoard b) => b.BlockCells.Count;

    private static int FreeCells(BlockDudeBoard b)
    {
        int n = 0;
        for (int y = 0; y < b.Height; y++)
            for (int x = 0; x < b.Width; x++)
                if (b.TileAt(x, y) != BlockDudeTile.Wall) n++;
        return n;
    }

    /// <summary>Structure line for one board: sizes, block counts, and whether the first fetchable block sits on
    /// the OPPOSITE side of the player from the door.</summary>
    private string Structure(string name, BlockDudeBoard b)
    {
        var (px, py) = PlayerCell(b);
        var (dx, dy) = NearestDoor(b);
        int dir = Math.Sign(dx - px);
        var (doorWalkable, grab, visited) = WalkOnly(b);
        int toward = grab.Count(x => dir != 0 && Math.Sign(x - px) == dir);
        int away = grab.Count(x => dir != 0 && Math.Sign(x - px) == -dir);
        bool mustWalkAway = dir != 0 && toward == 0 && away > 0;
        int nearestAway = 0;
        if (grab.Count > 0 && dir != 0)
        {
            int nx = grab.OrderBy(x => Math.Abs(x - px)).First();
            nearestAway = Math.Sign(nx - px) == -dir ? 1 : 0;
        }
        int blocksLeft = 0, blocksRight = 0;
        foreach (int cell in b.BlockCells)
        {
            int bx = cell % b.Width;
            if (Math.Sign(bx - px) == dir) blocksRight++; else if (Math.Sign(bx - px) == -dir) blocksLeft++;
        }
        return $"{name}\tW={b.Width}\tH={b.Height}\tfree={FreeCells(b)}\tblocks={BlockCount(b)}\tP=({px},{py})\tD=({dx},{dy})\tdoorAbove={(dy < py ? 1 : 0)}\tdoorsN={Doors(b).Count}\tgrabT={toward}\tgrabA={away}\tmustAway={(mustWalkAway ? 1 : 0)}\tnearestAway={nearestAway}\tblkToward={blocksRight}\tblkAway={blocksLeft}\twalkWin={(doorWalkable ? 1 : 0)}\tvisited={visited}";
    }

    [Fact]
    public void ShippedLevels()
    {
        foreach (var level in BlockDudeLevels.All)
            output.WriteLine(Structure(level.Name.Replace(' ', '_'), BlockDudeBoard.FromGrid(level.Grid)));
    }

    /// <summary>Rolls out one optimal solution and reports how far the dude walks AWAY from the door before he
    /// ever heads toward it.</summary>
    private static (int Optimal, int AwayDepth, int GrabsBeforeTowardProgress) OptimalWalkAway(BlockDudeBoard start, int cap)
    {
        var oracle = new BlockDudeOracle(start, cap);
        if (oracle.Truncated || oracle.OptimalFromStart < 0) return (-1, -1, -1);
        var mask = new Dictionary<int, int>();
        foreach (var (board, _, opt) in oracle.LabelledStates())
            mask.TryAdd(board.StateHash, opt);

        var (px0, _) = PlayerCell(start);
        var (dx, _) = NearestDoor(start);
        int dir = Math.Sign(dx - px0);
        var cur = start;
        int awayDepth = 0;
        int steps = 0;
        while (!cur.Won && steps < 4000)
        {
            if (!mask.TryGetValue(cur.StateHash, out int m) || m == 0) break;
            int a = 0;
            while ((m & (1 << a)) == 0) a++;
            cur = cur.Apply((BlockDudeAction)a);
            steps++;
            int away = (px0 - cur.PlayerX) * dir;
            if (away > awayDepth) awayDepth = away;
        }
        return (oracle.OptimalFromStart, awayDepth, cur.Won ? 1 : 0);
    }

    [Fact]
    public void GeneratedStages()
    {
        foreach (int stage in new[] { 0, 2, 4, 6 })
        {
            var spec = BlockDudeCurriculum.Stages[stage].Spec;
            var rng = new Xoshiro256StarStar(90210UL + (ulong)stage);
            int n = 0;
            for (int attempt = 0; attempt < 40_000 && n < 30; attempt++)
            {
                var board = BlockDudeGenerator.TryGenerate(rng, spec, out var outcome);
                if (outcome != BlockDudeGenerationOutcome.Accepted) continue;
                var oracleInfo = OptimalWalkAway(board!, spec.OracleMaxStates);
                if (oracleInfo.Optimal < 0) continue;
                n++;
                output.WriteLine($"{Structure($"S{stage}#{n}", board!)}\topt={oracleInfo.Optimal}\tawayDepth={oracleInfo.AwayDepth}\tsolved={oracleInfo.GrabsBeforeTowardProgress}");
            }
            output.WriteLine($"STAGE {stage} accepted-and-labelled {n}");
        }
    }
}

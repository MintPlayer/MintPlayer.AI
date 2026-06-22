using Microsoft.AspNetCore.Mvc;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Environments.Game2048;
using RLDemo.Web.Services;

namespace RLDemo.Web.Controllers;

/// <summary>A drawn board: 16 tile VALUES row-major (0 = empty, otherwise a power of two ≤ 32768).</summary>
public sealed record Board2048Dto(int[] Cells);

/// <summary>
/// One playout step, compact form: the move plus the random spawn it triggered.
/// 2048's resulting states are derivable deterministically from (action, spawn), so —
/// unlike Rush Hour — full per-step boards are omitted; <see cref="SolveResponse2048.FinalCells"/>
/// acts as the replay checksum.
/// </summary>
public sealed record PlayoutStepDto(int Action, int SpawnIndex, int SpawnValue, int ScoreGained);

public sealed record SolveResponse2048(
    int[] InitialCells,
    PlayoutStepDto[] Steps,
    int[] FinalCells,
    int Score,
    int MaxTile,
    bool Reached2048);

public sealed record Status2048Response(string Status, string? Error);

[ApiController]
[Route("api/2048")]
public sealed class Game2048Controller(Game2048ModelService model, GalleryStore gallery) : ControllerBase
{
    [HttpGet("status")]
    public Status2048Response Status()
    {
        _ = model.Agent; // touch: lazily loads a stored checkpoint
        return new(model.Status.ToString().ToLowerInvariant(), model.Error);
    }

    /// <summary>Lets the n-tuple agent play the drawn board to game over; deterministic per board.</summary>
    [HttpPost("solve")]
    public ActionResult<SolveResponse2048> Solve(Board2048Dto board)
    {
        if (!TryParseBoard(board, out byte[] cells, out string? error))
            return BadRequest(new { error });

        var agent = model.Agent;
        if (agent is null)
            return StatusCode(StatusCodes.Status503ServiceUnavailable, Status());

        // Deterministic spawns per drawn board, so gallery replays are reproducible.
        var rng = new Xoshiro256StarStar(BoardSeed(cells));
        int[] initialValues = (int[])board.Cells.Clone();
        Span<byte> current = cells.AsSpan();
        Span<byte> afterstate = stackalloc byte[16];
        var steps = new List<PlayoutStepDto>();
        int score = 0;

        while (Board2048.AnyMoveAvailable(current))
        {
            int action = agent.ChooseMove(current, out int gained, afterstate);
            if (action < 0) break; // defensive: AnyMoveAvailable said otherwise

            afterstate.CopyTo(current);
            Board2048.Spawn(current, rng);
            int spawnIndex = FindSpawn(afterstate, current);
            score += gained;
            steps.Add(new PlayoutStepDto(action, spawnIndex, 1 << current[spawnIndex], gained));
        }

        int maxExponent = Board2048.MaxExponent(current);
        var response = new SolveResponse2048(
            InitialCells: initialValues,
            Steps: [.. steps],
            FinalCells: ToValues(current),
            Score: score,
            MaxTile: 1 << maxExponent,
            Reached2048: maxExponent >= 11);

        gallery.Add("2048", $"AI played {steps.Count} moves, scored {score:N0}, best tile {1 << maxExponent}",
            board, response);
        return response;
    }

    private static bool TryParseBoard(Board2048Dto board, out byte[] cells, out string? error)
    {
        cells = new byte[16];
        error = null;
        if (board.Cells is not { Length: 16 })
        {
            error = "A board needs exactly 16 cells (row-major 4×4).";
            return false;
        }

        int tiles = 0;
        for (int i = 0; i < 16; i++)
        {
            int value = board.Cells[i];
            if (value == 0) continue;
            int exponent = int.TrailingZeroCount(value);
            if (value < 2 || value > 32768 || value != 1 << exponent)
            {
                error = $"Cell {i} has value {value}; tiles must be powers of two between 2 and 32768.";
                return false;
            }
            cells[i] = (byte)exponent;
            tiles++;
        }
        if (tiles == 0)
        {
            error = "Place at least one tile.";
            return false;
        }
        if (!Board2048.AnyMoveAvailable(cells))
        {
            error = "This board is already game over — no legal move exists.";
            return false;
        }
        return true;
    }

    private static int FindSpawn(ReadOnlySpan<byte> before, ReadOnlySpan<byte> after)
    {
        for (int i = 0; i < 16; i++)
            if (before[i] == 0 && after[i] != 0) return i;
        throw new InvalidOperationException("No spawn found.");
    }

    private static int[] ToValues(ReadOnlySpan<byte> exponents)
    {
        var values = new int[16];
        for (int i = 0; i < 16; i++)
            values[i] = exponents[i] == 0 ? 0 : 1 << exponents[i];
        return values;
    }

    /// <summary>Stable seed from the board content (FNV-1a) — same drawing, same playout.</summary>
    private static ulong BoardSeed(ReadOnlySpan<byte> cells)
    {
        ulong hash = 14695981039346656037UL;
        foreach (byte b in cells)
            hash = (hash ^ b) * 1099511628211UL;
        return hash;
    }
}

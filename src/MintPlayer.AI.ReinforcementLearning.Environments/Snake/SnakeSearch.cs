using System.Text;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Snake;

/// <summary>
/// Tuning for the look-ahead planner (<see cref="SnakeEnv.ChooseActionSearch"/>, M34). Defaults are the measured sweet
/// spot: <b>depth 12 / beam 16</b> lifts food@12 from the reactive ~50 plateau to ~70 (deeper/wider MISRANKS under beam
/// pruning and scores worse — measured d20/b32 ≈ 66). The flood-fill survival search does the work; the net only breaks
/// ties between equally-safe root moves, so <see cref="NetWeight"/> is deliberately <b>small</b> — a big weight lets the
/// net override a better survival move and slightly hurts (measured net 500 ≈ 68 vs pure-survival ≈ 71 over 12 eps). The
/// <b>algorithm</b> lives once in <c>snake_solver.pg</c> (<c>chooseActionSearch</c>); this record is just the knobs.
/// </summary>
/// <param name="MaxDepth">Plies to look ahead. The sweet spot is ~12; deeper misranks under beam pruning.</param>
/// <param name="BeamWidth">Live nodes carried to the next ply, best-scoring first. Wider keeps greedy food-grabs that trap later.</param>
/// <param name="FoodWeight">Reward per food eaten along a line — the dominant term; eating safely beats any non-eating line.</param>
/// <param name="TrapPenalty">Penalty for a leaf whose reachable space can no longer hold the body (a guaranteed future self-trap).</param>
/// <param name="NetWeight">Weight on the trained net's Q, applied ONCE per move as a root-move tiebreak (not per node). Small = pure tiebreak; 0 = ignore the net.</param>
/// <param name="SpaceWeight">Weight on reachable free space — keep more room open.</param>
/// <param name="FoodDistWeight">Pull toward food when no line eats within the horizon (tiebreak by L1 head→food distance).</param>
public sealed record SnakeSearchConfig(
    int MaxDepth = 12,
    int BeamWidth = 16,
    double FoodWeight = 10_000,
    double TrapPenalty = 50_000,
    double NetWeight = 50,
    double SpaceWeight = 50,
    double FoodDistWeight = 1);

/// <summary>
/// Reads a trained dueling-Q checkpoint (RLNC / kind <c>"dueling-q"</c>) into the single-source transpiled
/// <see cref="PgSnakeNet"/> the planner evaluates leaves with. This is the C# twin of the browser's
/// <c>snake-net.ts</c> <c>parseSnakeNet</c> — same byte layout, so C# eval and the browser score identical positions.
/// Kept internal: <see cref="PgSnakeNet"/> is an internal transpiled type, exposed only through <see cref="SnakeEnv"/>.
/// </summary>
internal static class SnakeNetIo
{
    private const uint Magic = 0x434e4c52; // "RLNC"
    private const string Kind = "dueling-q";

    public static PgSnakeNet Parse(Stream checkpoint)
    {
        using var r = new BinaryReader(checkpoint, Encoding.UTF8, leaveOpen: true);
        if (r.ReadUInt32() != Magic)
            throw new InvalidDataException("Not an RLNC checkpoint.");
        string kind = r.ReadString();
        if (kind != Kind)
            throw new InvalidDataException($"Expected checkpoint kind '{Kind}', got '{kind}'.");

        int version = r.ReadInt32();
        int inputSize = r.ReadInt32();
        int hiddenCount = r.ReadInt32();
        var hidden = new List<int>(hiddenCount);
        for (int i = 0; i < hiddenCount; i++) hidden.Add(r.ReadInt32());
        int actions = r.ReadInt32();
        bool noisy = version >= 2 && r.ReadByte() != 0;

        List<double> ReadFloats()
        {
            int n = r.ReadInt32();
            var a = new List<double>(n);
            for (int i = 0; i < n; i++) a.Add(r.ReadSingle()); // float32 → its exact f64, matching the browser
            return a;
        }

        var trunkWFlat = new List<double>();
        var trunkBFlat = new List<double>();
        for (int l = 0; l < hiddenCount; l++)
        {
            trunkWFlat.AddRange(ReadFloats());
            trunkBFlat.AddRange(ReadFloats());
        }

        List<double> valueW, valueB, advW, advB;
        if (!noisy)
        {
            valueW = ReadFloats(); valueB = ReadFloats();
            advW = ReadFloats(); advB = ReadFloats();
        }
        else
        {
            // Noisy heads store Mean/Sigma pairs; inference runs with noise off, so keep the Mean tensors only.
            valueW = ReadFloats(); ReadFloats(); valueB = ReadFloats(); ReadFloats();
            advW = ReadFloats(); ReadFloats(); advB = ReadFloats(); ReadFloats();
        }

        return new PgSnakeNet(inputSize, actions, hidden, trunkWFlat, trunkBFlat, valueW, valueB, advW, advB);
    }
}

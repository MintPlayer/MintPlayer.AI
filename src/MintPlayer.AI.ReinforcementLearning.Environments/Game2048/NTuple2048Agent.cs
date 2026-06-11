using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Game2048;

/// <summary>
/// Afterstate TD(0) learner with an n-tuple network (Szubert &amp; Jaśkowski 2014) —
/// the literature-standard approach for 2048. The value function is a sum of lookup
/// tables over 17 4-cell tuples (4 rows, 4 columns, 9 2×2 squares; 16⁴ entries each,
/// ~4.5 MB total). Learning targets AFTERSTATES (the board after the slide/merge,
/// before the random spawn), which factors the stochasticity out of the value function:
///   choose a* = argmax_a [ r_a + V(afterstate_a) ]
///   V(after_{t-1}) += α/17 · (r_t + V(after_t) − V(after_{t-1}))  per table.
/// Rewards here are classic score increments (merged tile values).
/// </summary>
public sealed class NTuple2048Agent
{
    private static readonly int[][] Tuples = BuildTuples();
    private readonly float[][] _tables;

    public NTuple2048Agent()
    {
        _tables = new float[Tuples.Length][];
        for (int t = 0; t < Tuples.Length; t++)
            _tables[t] = new float[65536];
    }

    /// <summary>Total learning rate, split evenly across the 17 tables.</summary>
    public double Alpha { get; set; } = 0.1;

    public const string CheckpointKind = "ntuple2048";
    private const int CheckpointVersion = 1;

    /// <summary>Versioned binary checkpoint: alpha + the 17 weight tables (~4.5 MB).</summary>
    public void Save(Stream destination)
    {
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.WriteHeader(writer, CheckpointKind, CheckpointVersion);
        writer.Write(Alpha);
        writer.Write(_tables.Length);
        foreach (var table in _tables)
            CheckpointFormat.WriteFloats(writer, table);
    }

    public static NTuple2048Agent Load(Stream source)
    {
        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        CheckpointFormat.ReadHeader(reader, CheckpointKind, CheckpointVersion);
        var agent = new NTuple2048Agent { Alpha = reader.ReadDouble() };

        int tableCount = reader.ReadInt32();
        if (tableCount != agent._tables.Length)
            throw new InvalidDataException($"Checkpoint has {tableCount} tuple tables, expected {agent._tables.Length}.");
        for (int t = 0; t < tableCount; t++)
        {
            var table = CheckpointFormat.ReadFloats(reader);
            if (table.Length != 65536)
                throw new InvalidDataException($"Tuple table {t} has {table.Length} entries, expected 65536.");
            agent._tables[t] = table;
        }
        return agent;
    }

    private static int[][] BuildTuples()
    {
        var tuples = new List<int[]>();
        for (int r = 0; r < 4; r++)
            tuples.Add([r * 4, r * 4 + 1, r * 4 + 2, r * 4 + 3]);
        for (int c = 0; c < 4; c++)
            tuples.Add([c, c + 4, c + 8, c + 12]);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                tuples.Add([r * 4 + c, r * 4 + c + 1, (r + 1) * 4 + c, (r + 1) * 4 + c + 1]);
        return [.. tuples];
    }

    public double Evaluate(ReadOnlySpan<byte> board)
    {
        double value = 0;
        for (int t = 0; t < Tuples.Length; t++)
            value += _tables[t][TupleIndex(board, Tuples[t])];
        return value;
    }

    private void Update(ReadOnlySpan<byte> board, double totalDelta)
    {
        float perTable = (float)(totalDelta / Tuples.Length);
        for (int t = 0; t < Tuples.Length; t++)
            _tables[t][TupleIndex(board, Tuples[t])] += perTable;
    }

    private static int TupleIndex(ReadOnlySpan<byte> board, int[] cells)
        => board[cells[0]] | board[cells[1]] << 4 | board[cells[2]] << 8 | board[cells[3]] << 12;

    /// <summary>Best move by r + V(afterstate); returns −1 when no move is legal.</summary>
    public int ChooseMove(ReadOnlySpan<byte> board, out int reward, Span<byte> bestAfterstate)
    {
        Span<byte> scratch = stackalloc byte[16];
        int bestAction = -1;
        double bestValue = double.NegativeInfinity;
        reward = 0;

        for (int action = 0; action < Board2048.ActionCount; action++)
        {
            board.CopyTo(scratch);
            if (!Board2048.ApplyMove(scratch, action, out _, out int mergedValues))
                continue;

            double value = mergedValues + Evaluate(scratch);
            if (value > bestValue)
            {
                bestValue = value;
                bestAction = action;
                reward = mergedValues;
                scratch.CopyTo(bestAfterstate);
            }
        }
        return bestAction;
    }

    /// <summary>Plays one game; with <paramref name="learn"/>, applies TD(0) updates along the way.</summary>
    public (int Score, int MaxExponent) PlayGame(Xoshiro256StarStar rng, bool learn)
    {
        Span<byte> board = stackalloc byte[16];
        Span<byte> after = stackalloc byte[16];
        Span<byte> prevAfter = stackalloc byte[16];
        board.Clear();
        Board2048.Spawn(board, rng);
        Board2048.Spawn(board, rng);

        int score = 0;
        bool hasPrev = false;

        while (true)
        {
            int action = ChooseMove(board, out int reward, after);
            if (action < 0) break; // defensive; terminal is detected after the spawn below

            if (learn && hasPrev)
                Update(prevAfter, Alpha * (reward + Evaluate(after) - Evaluate(prevAfter)));

            score += reward;
            after.CopyTo(board);
            Board2048.Spawn(board, rng);

            if (!Board2048.AnyMoveAvailable(board))
            {
                if (learn)
                    Update(after, Alpha * (0 - Evaluate(after)));
                break;
            }

            after.CopyTo(prevAfter);
            hasPrev = true;
        }

        return (score, Board2048.MaxExponent(board));
    }
}

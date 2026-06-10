using System.Text.Json;
using RLNet.Core.Agents.Tabular;

namespace RLNet.Core.Checkpoints;

/// <summary>
/// JSON checkpoint for tabular Q-tables (PRD decision: human-inspectable, exact —
/// doubles round-trip losslessly through System.Text.Json's "R" formatting).
/// </summary>
public static class TabularCheckpoint
{
    public const string Kind = "tabular-q";
    private const int Version = 1;

    private sealed record Dto(string Kind, int Version, int StateCount, int ActionCount, double[][] Q);

    public static void Save(TabularAgent agent, Stream destination)
    {
        int states = agent.Q.GetLength(0);
        int actions = agent.Q.GetLength(1);
        var q = new double[states][];
        for (int s = 0; s < states; s++)
        {
            q[s] = new double[actions];
            for (int a = 0; a < actions; a++)
                q[s][a] = agent.Q[s, a];
        }
        JsonSerializer.Serialize(destination, new Dto(Kind, Version, states, actions, q),
            new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>Fills <paramref name="agent"/>'s Q-table; dimensions must match exactly.</summary>
    public static void LoadInto(TabularAgent agent, Stream source)
    {
        var dto = JsonSerializer.Deserialize<Dto>(source)
            ?? throw new InvalidDataException("Empty tabular checkpoint.");
        if (dto.Kind != Kind)
            throw new InvalidDataException($"Checkpoint kind mismatch: expected '{Kind}', found '{dto.Kind}'.");
        if (dto.Version != Version)
            throw new InvalidDataException($"Unsupported '{Kind}' checkpoint version {dto.Version}.");
        if (dto.StateCount != agent.Q.GetLength(0) || dto.ActionCount != agent.Q.GetLength(1))
            throw new InvalidDataException(
                $"Checkpoint is {dto.StateCount}x{dto.ActionCount}, agent is {agent.Q.GetLength(0)}x{agent.Q.GetLength(1)}.");

        for (int s = 0; s < dto.StateCount; s++)
            for (int a = 0; a < dto.ActionCount; a++)
                agent.Q[s, a] = dto.Q[s][a];
    }
}

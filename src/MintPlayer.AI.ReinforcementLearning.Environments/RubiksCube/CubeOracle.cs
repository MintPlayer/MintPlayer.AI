using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Imitation-data source (PLAN M16): Kociemba is the cube's oracle, and every random
/// scramble solved once yields a whole labeled trajectory. Each state along the solution
/// path is labeled with the next quarter-turn action and the quarter-turn distance-to-go
/// (the policy and value targets). Kociemba's solutions are not optimal in quarter-turn
/// metric — they are a consistent, always-available teacher, which is what imitation
/// needs (the M11 lesson).
/// </summary>
public static class CubeOracle
{
    public sealed record LabeledState(byte[] Facelets, int Action, int DistanceToGo);

    /// <summary>
    /// Scrambles to a random depth (1..<paramref name="maxScrambleDepth"/>), solves with
    /// Kociemba and returns the labeled states along the solution path (scrambled state
    /// first, last state one move before solved). Null when the solver errs (never on
    /// scramble-generated cubes; defensive).
    /// </summary>
    public static List<LabeledState>? LabelScramblePath(Xoshiro256StarStar rng, int maxScrambleDepth = 22)
    {
        var cube = new FaceletCube();
        cube.Apply(FaceletCube.ScrambleMoves(rng, 1 + rng.NextInt(maxScrambleDepth)));
        if (cube.IsSolved) return [];

        var result = CubeSolver.Solve(cube);
        if (!result.Solved) return null;

        int[] actions = ExpandToQuarterTurnActions(result.Moves);
        var states = new List<LabeledState>(actions.Length);
        for (int i = 0; i < actions.Length; i++)
        {
            states.Add(new LabeledState(cube.Facelets.ToArray(), actions[i], actions.Length - i));
            cube.ApplyQuarterTurn(actions[i]);
        }
        return states;
    }

    /// <summary>Half-turn-metric moves ("R2", "U'", …) → quarter-turn action ids ("R2" becomes two turns).</summary>
    public static int[] ExpandToQuarterTurnActions(IReadOnlyList<string> moves)
    {
        var actions = new List<int>(moves.Count * 2);
        foreach (string move in moves)
        {
            if (move.Length == 2 && move[1] == '2')
            {
                int action = QuarterTurnAction(move[..1]);
                actions.Add(action);
                actions.Add(action);
            }
            else
            {
                actions.Add(QuarterTurnAction(move));
            }
        }
        return [.. actions];
    }

    private static int QuarterTurnAction(string move)
    {
        int action = Array.IndexOf(FaceletCube.QuarterTurnMoves, move);
        return action >= 0 ? action : throw new ArgumentException($"Not a quarter-turn move: '{move}'.");
    }
}

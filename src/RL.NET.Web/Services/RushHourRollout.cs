using RLNet.Core.Training;
using RLNet.Environments.RushHour;

namespace RLNet.Web.Services;

/// <summary>
/// Greedy rollout of a Q-network with cycle avoidance: among legal actions, the
/// highest-Q action leading to a NOT-yet-visited state is taken. A plain greedy
/// rollout is deterministic, so one badly-ranked state makes it shuttle a vehicle
/// back and forth until the budget runs out; refusing to revisit states converts
/// those infinite loops into progress (or an honest, fast exhaustion).
/// </summary>
public static class RushHourRollout
{
    public readonly record struct RolloutStep(int Vehicle, int Direction, int[] Positions);

    public static (bool Solved, List<RolloutStep> Steps) Run(GreedyQAgent agent, RushHourPuzzle puzzle, int maxMoves)
    {
        var env = new RushHourEnv([puzzle], maxMoves) { FixedPuzzleIndex = 0 };
        env.Reset(1);
        var positions = RushHourBoard.InitialPositions(puzzle);
        var visited = new HashSet<ulong> { RushHourSolver.Encode(positions) };
        var steps = new List<RolloutStep>();

        while (true)
        {
            var mask = env.CurrentActionMask();
            var q = agent.QValues(env.CurrentObservation());

            int best = -1, fallback = -1;
            for (int a = 0; a < RushHourBoard.ActionCount; a++)
            {
                if (!mask[a]) continue;
                if (fallback < 0 || q[a] > q[fallback]) fallback = a;

                positions[a / 2] += a % 2 == 0 ? -1 : 1;
                bool fresh = !visited.Contains(RushHourSolver.Encode(positions));
                positions[a / 2] -= a % 2 == 0 ? -1 : 1;
                if (fresh && (best < 0 || q[a] > q[best])) best = a;
            }

            // All successors already visited: concede to the plain argmax (budget will end it).
            int action = best >= 0 ? best : fallback;
            var step = env.Step(action);
            positions[action / 2] += action % 2 == 0 ? -1 : 1;
            visited.Add(RushHourSolver.Encode(positions));
            steps.Add(new RolloutStep(action / 2, action % 2, [.. positions]));

            if (step.Done) return (step.Terminated, steps);
        }
    }
}

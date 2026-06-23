using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Inference strategies on top of <see cref="CubePolicyNet"/>, mirroring
/// <see cref="RushHour.RushHourPolicySearch"/>: a reactive greedy rollout (no-undo mask +
/// visited-state cycle avoidance) and policy-guided A* with the value head as heuristic.
/// Solutions are capped at <see cref="MaxSolutionMoves"/> quarter-turns — Kociemba stays
/// the oracle for guaranteed short answers; this is "the AI, with lookahead".
/// </summary>
public static class CubePolicySearch
{
    /// <summary>Kociemba ≤ 22 HTM ≈ ≤ 30–40 QTM; a learned solver gets the same generous cap.</summary>
    public const int MaxSolutionMoves = 40;

    /// <summary>Reactive play: best non-undo logit leading to an unvisited state.</summary>
    public static (bool Solved, List<int> Actions) GreedyRollout(CubePolicyNet net, FaceletCube start, int maxMoves = MaxSolutionMoves)
    {
        var cube = start.Clone();
        var visited = new HashSet<string> { cube.ToKociembaString() };
        var actions = new List<int>();
        int lastAction = -1;

        for (int move = 0; move < maxMoves; move++)
        {
            if (cube.IsSolved) return (true, actions);
            var (logits, _) = net.Evaluate(cube, lastAction);

            int best = -1, fallback = -1;
            for (int a = 0; a < RubiksCubeEnv.ActionCount; a++)
            {
                if (float.IsNegativeInfinity(logits[a])) continue;
                if (fallback < 0 || logits[a] > logits[fallback]) fallback = a;

                var next = cube.Clone();
                next.ApplyQuarterTurn(a);
                if (!visited.Contains(next.ToKociembaString()) && (best < 0 || logits[a] > logits[best]))
                    best = a;
            }
            if (fallback < 0) break;

            int action = best >= 0 ? best : fallback;
            cube.ApplyQuarterTurn(action);
            visited.Add(cube.ToKociembaString());
            actions.Add(action);
            lastAction = action;
        }
        return (cube.IsSolved, actions);
    }

    public sealed record SearchResult(bool Solved, string[] Moves, int Expansions);

    /// <summary>A* with the learned distance-to-solved as heuristic; budgeted by node expansions.</summary>
    public static SearchResult Solve(CubePolicyNet net, FaceletCube start, int maxExpansions = 20_000)
    {
        if (start.IsSolved)
            return new(true, [], 0);

        var open = new PriorityQueue<Node, double>();
        var bestDepth = new Dictionary<string, int> { [start.ToKociembaString()] = 0 };
        open.Enqueue(new Node(start, null, -1, 0), net.Evaluate(start).Distance);

        int expansions = 0;
        while (open.Count > 0 && expansions < maxExpansions)
        {
            var node = open.Dequeue();
            expansions++;

            int undo = RubiksCubeEnv.InverseAction(node.Action);
            for (int action = 0; action < RubiksCubeEnv.ActionCount; action++)
            {
                if (action == undo) continue;

                var child = node.Cube.Clone();
                child.ApplyQuarterTurn(action);
                int depth = node.Depth + 1;

                if (child.IsSolved)
                    return new(true, ExtractMoves(node, action), expansions);
                if (depth >= MaxSolutionMoves) continue;

                string key = child.ToKociembaString();
                if (bestDepth.TryGetValue(key, out int seen) && seen <= depth) continue;
                bestDepth[key] = depth;

                open.Enqueue(new Node(child, node, action, depth),
                    depth + net.Evaluate(child, action).Distance);
            }
        }

        return new(false, [], expansions);
    }

    private static string[] ExtractMoves(Node parent, int finalAction)
    {
        var actions = new List<int> { finalAction };
        for (var node = parent; node is not null && node.Action >= 0; node = node.Parent)
            actions.Add(node.Action);
        actions.Reverse();
        return [.. actions.Select(a => FaceletCube.QuarterTurnMoves[a])];
    }

    private sealed record Node(FaceletCube Cube, Node? Parent, int Action, int Depth);

    /// <summary>CPU overload: runs the policy net's batched forward through the autograd backend.</summary>
    public static SearchResult BeamSearch(CubePolicyNet net, FaceletCube start, int beamWidth = 2_000, int maxDepth = MaxSolutionMoves)
        => BeamSearch((features, rows) =>
        {
            using (GradMode.NoGrad())
                return net.Forward(new Tensor(features, rows, RubiksCubeEnv.ObservationSize)).Logits.Data;
        }, start, beamWidth, maxDepth);

    /// <summary>
    /// EfficientCube-style beam search: keep the <paramref name="beamWidth"/> highest cumulative
    /// log-probability move sequences, expand every node's non-undo children each step, and re-prune
    /// to the beam (deduping repeated states, keeping the best-scoring route to each). Ranked purely
    /// by the policy — the value head is unused — so it needs no admissible heuristic; it finds short
    /// solutions by stitching together the moves the policy is locally most confident about. A wider
    /// beam reaches deeper scrambles at the cost of more net evaluations per step.
    /// <para>
    /// <paramref name="policyLogits"/> maps a row-major observation batch (rows × ObservationSize) to
    /// row-major raw policy logits (rows × ActionCount): a CPU autograd forward or, for the bulk of the
    /// solve cost, a GPU-resident <c>DeviceMlp</c> whose weights stay on the device across steps.
    /// </para>
    /// </summary>
    public static SearchResult BeamSearch(Func<float[], int, float[]> policyLogits, FaceletCube start, int beamWidth = 2_000, int maxDepth = MaxSolutionMoves)
    {
        if (start.IsSolved)
            return new(true, [], 0);

        var beam = new List<BeamNode> { new(start, -1, [], 0.0, start.ToKociembaString()) };
        int expansions = 0;

        for (int depth = 0; depth < maxDepth && beam.Count > 0; depth++)
        {
            // One batched forward over the whole beam (the net dominates the step cost).
            int n = beam.Count;
            var obs = new float[n * RubiksCubeEnv.ObservationSize];
            for (int i = 0; i < n; i++)
                RubiksCubeEnv.WriteObservation(beam[i].Cube, obs.AsSpan(i * RubiksCubeEnv.ObservationSize, RubiksCubeEnv.ObservationSize));

            float[] logits = policyLogits(obs, n);

            var candidates = new List<BeamNode>(n * RubiksCubeEnv.ActionCount);
            for (int i = 0; i < n; i++)
            {
                expansions++;
                var node = beam[i];
                int off = i * RubiksCubeEnv.ActionCount;
                int undo = RubiksCubeEnv.InverseAction(node.LastAction);
                // Forbid a 3rd consecutive identical quarter-turn. U U is a legitimate half-turn, but U U U ≡ U'
                // (always reducible to one opposite quarter), so cap same-move runs at 2 — with the immediate-inverse
                // mask, the only same-face repeat left is the half-turn. This is what stops degenerate runs like U'×5.
                int blockedRepeat = node.Path.Count >= 2 && node.Path[^1] == node.Path[^2] ? node.LastAction : -1;

                // log-softmax over the allowed moves (undo and the 3rd-repeat excluded).
                float max = float.NegativeInfinity;
                for (int a = 0; a < RubiksCubeEnv.ActionCount; a++)
                    if (a != undo && a != blockedRepeat && logits[off + a] > max) max = logits[off + a];
                double sumExp = 0;
                for (int a = 0; a < RubiksCubeEnv.ActionCount; a++)
                    if (a != undo && a != blockedRepeat) sumExp += Math.Exp(logits[off + a] - max);
                double logZ = max + Math.Log(sumExp);

                for (int a = 0; a < RubiksCubeEnv.ActionCount; a++)
                {
                    if (a == undo || a == blockedRepeat) continue;
                    var child = node.Cube.Clone();
                    child.ApplyQuarterTurn(a);
                    var path = new List<int>(node.Path) { a };
                    if (child.IsSolved)
                        return new(true, Canonicalize(path), expansions);
                    double cum = node.CumLogProb + (logits[off + a] - logZ);
                    candidates.Add(new BeamNode(child, a, path, cum, child.ToKociembaString()));
                }
            }

            // Prune to the top-`beamWidth` distinct states by cumulative log-probability.
            candidates.Sort((x, y) => y.CumLogProb.CompareTo(x.CumLogProb));
            var next = new List<BeamNode>(Math.Min(beamWidth, candidates.Count));
            var taken = new HashSet<string>();
            foreach (var c in candidates)
            {
                if (!taken.Add(c.Key)) continue;
                next.Add(c);
                if (next.Count >= beamWidth) break;
            }
            beam = next;
        }

        return new(false, [], expansions);
    }

    /// <summary>
    /// Collapse a raw quarter-turn path into its canonical minimal form: fold each maximal run of same-face
    /// turns modulo 4 (so U'×5 → U', U U U U → nothing, U U → a half-turn kept as two quarter-turns, X X' →
    /// cancel — and cancellations re-expose neighbours so e.g. U F F' U → U U). The result is algebraically
    /// identical, so it still solves; it just removes redundancy the beam could otherwise leave in (and so the
    /// reported move count reflects real work). The action space is quarter-turns only, hence no U2 token.
    /// </summary>
    public static string[] Canonicalize(IReadOnlyList<int> path)
    {
        var runs = new List<(int Face, int Net)>(); // Net = net clockwise quarter-turns in 1..3
        foreach (int a in path)
        {
            int face = a / 2;
            int dir = a % 2 == 0 ? 1 : 3; // prime (odd action) = −1 ≡ +3 mod 4
            if (runs.Count > 0 && runs[^1].Face == face)
            {
                int net = (runs[^1].Net + dir) % 4;
                if (net == 0) runs.RemoveAt(runs.Count - 1); // a full turn / inverse pair — vanishes
                else runs[^1] = (face, net);
            }
            else
            {
                runs.Add((face, dir));
            }
        }

        var moves = new List<string>(runs.Count);
        foreach (var (face, net) in runs)
        {
            int cw = face * 2, prime = face * 2 + 1;
            if (net == 1) moves.Add(FaceletCube.QuarterTurnMoves[cw]);
            else if (net == 3) moves.Add(FaceletCube.QuarterTurnMoves[prime]);
            else { moves.Add(FaceletCube.QuarterTurnMoves[cw]); moves.Add(FaceletCube.QuarterTurnMoves[cw]); } // half-turn
        }
        return [.. moves];
    }

    private sealed record BeamNode(FaceletCube Cube, int LastAction, List<int> Path, double CumLogProb, string Key);
}

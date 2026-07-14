using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>
/// AlphaZero-style PUCT Monte-Carlo Tree Search over an <see cref="IZeroSumGame{TState}"/>, guided by a neural
/// leaf evaluator (policy priors + a value in [-1,1]). It composes over the game seam + an <see cref="Evaluate{TState}"/>
/// delegate exactly as <see cref="ValueGuidedSearch"/> composes over <see cref="IDeterministicModel{TState}"/> + a
/// cost-to-go delegate, so Core carries no game or NN dependency.
/// <para>
/// Each simulation walks the tree by maximizing <c>Q + U</c> (mean action value + a prior-weighted exploration
/// bonus), expands one new leaf, evaluates it with the net, and backs the value up the path <b>negating it every
/// ply</b> (zero-sum: a position good for the child's mover is bad for the parent's). At the root, Dirichlet noise is
/// mixed into the priors so self-play keeps exploring. <see cref="Search{TState}"/> returns the root's visit-count
/// distribution — the policy target π for training, from which the caller samples (early game) or takes the argmax.
/// </para>
/// </summary>
public static class Mcts
{
    /// <param name="Simulations">Tree simulations per move (more = stronger, slower).</param>
    /// <param name="Cpuct">PUCT exploration constant.</param>
    /// <param name="DirichletAlpha">Concentration of the root-noise Dirichlet (chess ≈ 0.3; smaller = spikier).</param>
    /// <param name="RootNoiseFrac">Fraction of the root prior replaced by Dirichlet noise (0 disables).</param>
    public sealed record Config(int Simulations = 200, float Cpuct = 1.25f, float DirichletAlpha = 0.3f, float RootNoiseFrac = 0.25f);

    /// <summary>Evaluates a leaf: a probability distribution over <c>PolicySize</c> (illegal moves ≈ 0; the search
    /// renormalizes over the legal set) and the position value in [-1,1] from the side-to-move's perspective. Backed
    /// by a <c>PolicyValueNet.Forward</c> under <c>NoGrad</c>, softmaxed over legal moves + <c>tanh</c> value.</summary>
    public delegate (float[] Priors, float Value) Evaluate<TState>(TState state);

    /// <summary>Batched leaf evaluator: evaluates <paramref name="states"/> in ONE forward pass and returns a result
    /// per state, in order (same shape as <see cref="Evaluate{TState}"/>). This is the seam that lets a batched
    /// <see cref="SearchBatched{TState}"/> feed the net batch-of-N observations instead of batch-1 — the difference
    /// between a GPU sitting idle and a GPU doing real work. Backed by one <c>net.Forward([N, obsSize])</c>.</summary>
    public delegate IReadOnlyList<(float[] Priors, float Value)> BatchEvaluate<TState>(IReadOnlyList<TState> states);

    private sealed class Node
    {
        public required int[] Moves;      // legal action indices in this state
        public float[] P = [];            // prior per move (aligned to Moves)
        public int[] N = [];              // visit count per move
        public float[] W = [];            // summed value per move, from THIS node's mover perspective
        public int[] Vloss = [];          // in-flight virtual-loss count per move (batched search only; 0 otherwise)
        public Node?[] Children = [];
        public bool Expanded;
        public bool Terminal;
        public float TerminalValue;       // set when Terminal: +1 win / -1 loss / 0 draw, mover-relative
    }

    /// <summary>Runs <paramref name="config"/>.Simulations from <paramref name="rootState"/> and returns the root
    /// visit-count distribution over <c>game.PolicySize</c> (sums to 1 over the legal moves; 0 elsewhere). Falls back
    /// to the raw priors only if no simulation recorded a visit (Simulations == 0).</summary>
    public static float[] Search<TState>(IZeroSumGame<TState> game, TState rootState,
        Evaluate<TState> evaluate, Config config, Xoshiro256StarStar rng)
    {
        var root = NewNode(game, rootState);
        if (!root.Terminal)
        {
            ExpandLeaf(root, evaluate, rootState);
            AddRootNoise(root, config, rng);
            for (int sim = 0; sim < config.Simulations; sim++)
                Simulate(root, game, rootState, evaluate, config);
        }

        var pi = new float[game.PolicySize];
        int total = 0;
        for (int i = 0; i < root.Moves.Length; i++) total += root.N[i];
        if (total == 0)
        {
            for (int i = 0; i < root.Moves.Length; i++) pi[root.Moves[i]] = root.P[i];
            return pi;
        }
        for (int i = 0; i < root.Moves.Length; i++) pi[root.Moves[i]] = root.N[i] / (float)total;
        return pi;
    }

    /// <summary>
    /// Batched variant of <see cref="Search{TState}"/>: identical PUCT tree, but simulations run in waves of up to
    /// <paramref name="leafBatch"/>. Within a wave each descent applies <b>virtual loss</b> to the edges it traverses
    /// (a transient pessimistic visit) so successive descents diverge to different leaves; the wave's leaves are then
    /// evaluated in a SINGLE <paramref name="batchEvaluate"/> call, expanded, and backed up (removing the virtual
    /// loss). This is the only way an MCTS self-play run keeps a GPU busy — batch-1 leaf eval barely uses one.
    /// <para>Deterministic per <c>(seed, leafBatch)</c>. At <paramref name="leafBatch"/> == 1 no virtual loss is ever
    /// observed during selection, so it reproduces <see cref="Search{TState}"/> bitwise.</para>
    /// </summary>
    public static float[] SearchBatched<TState>(IZeroSumGame<TState> game, TState rootState,
        BatchEvaluate<TState> batchEvaluate, Config config, Xoshiro256StarStar rng, int leafBatch)
    {
        if (leafBatch < 1) throw new ArgumentOutOfRangeException(nameof(leafBatch));
        var root = NewNode(game, rootState);
        if (!root.Terminal)
        {
            ExpandWith(root, batchEvaluate([rootState])[0].Priors); // one-position batch to seed the root
            AddRootNoise(root, config, rng);

            var leafState = new Dictionary<Node, TState>();          // distinct non-terminal leaves awaiting eval
            var leafValue = new Dictionary<Node, float>();           // resolved value per leaf (terminal or evaluated)
            var paths = new List<(List<(Node Node, int Edge)> Path, Node Leaf)>(leafBatch);
            for (int done = 0; done < config.Simulations;)
            {
                int wave = Math.Min(leafBatch, config.Simulations - done);
                paths.Clear(); leafState.Clear(); leafValue.Clear();
                for (int b = 0; b < wave; b++)
                {
                    var (path, leaf, state, terminal) = DescendVirtual(root, rootState, game, config);
                    paths.Add((path, leaf));
                    if (terminal) leafValue[leaf] = leaf.TerminalValue;
                    else if (!leafState.ContainsKey(leaf)) leafState[leaf] = state; // dedupe collisions on one leaf
                }
                if (leafState.Count > 0)
                {
                    var nodes = new List<Node>(leafState.Keys);
                    var states = new List<TState>(nodes.Count);
                    foreach (var n in nodes) states.Add(leafState[n]);
                    var results = batchEvaluate(states);
                    for (int i = 0; i < nodes.Count; i++)
                    {
                        ExpandWith(nodes[i], results[i].Priors);
                        leafValue[nodes[i]] = results[i].Value;
                    }
                }
                foreach (var (path, leaf) in paths) BackpropVirtual(path, leafValue[leaf]);
                done += wave;
            }
        }

        var pi = new float[game.PolicySize];
        int total = 0;
        for (int i = 0; i < root.Moves.Length; i++) total += root.N[i];
        if (total == 0)
        {
            for (int i = 0; i < root.Moves.Length; i++) pi[root.Moves[i]] = root.P[i];
            return pi;
        }
        for (int i = 0; i < root.Moves.Length; i++) pi[root.Moves[i]] = root.N[i] / (float)total;
        return pi;
    }

    // Descend from the root selecting edges (virtual-loss-aware), applying virtual loss to each traversed edge, until
    // a terminal or unexpanded leaf. Returns the path (nodes+edges) and the leaf (+its state, +whether terminal).
    private static (List<(Node Node, int Edge)> Path, Node Leaf, TState LeafState, bool Terminal) DescendVirtual<TState>(
        Node root, TState rootState, IZeroSumGame<TState> game, Config config)
    {
        var path = new List<(Node, int)>();
        var node = root;
        var state = rootState;
        while (true)
        {
            int edge = SelectChildVirtual(node, config);
            node.Vloss[edge]++;                          // discourage the rest of this wave from re-taking this edge
            path.Add((node, edge));
            var childState = game.Apply(state, node.Moves[edge]);
            node.Children[edge] ??= NewNode(game, childState);
            var child = node.Children[edge]!;
            if (child.Terminal) return (path, child, childState, true);
            if (!child.Expanded) return (path, child, childState, false);
            node = child;
            state = childState;
        }
    }

    // PUCT with virtual loss folded in: each in-flight visit counts as an extra visit valued -1 to the mover, so a
    // path already chosen this wave looks worse. With Vloss all 0 this is identical to SelectChild.
    private static int SelectChildVirtual(Node node, Config config)
    {
        int sumN = 0;
        for (int i = 0; i < node.N.Length; i++) sumN += node.N[i] + node.Vloss[i];
        float sqrtSum = MathF.Sqrt(sumN);

        int best = 0;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < node.Moves.Length; i++)
        {
            int nEff = node.N[i] + node.Vloss[i];
            float q = nEff > 0 ? (node.W[i] - node.Vloss[i]) / nEff : 0f;
            float u = config.Cpuct * node.P[i] * sqrtSum / (1 + nEff);
            float score = q + u;
            if (score > bestScore) { bestScore = score; best = i; }
        }
        return best;
    }

    // Back up the leaf value along the path (negating each ply, zero-sum) and remove the virtual loss applied on
    // the way down. The leaf's parent edge takes -value; the sign flips going up.
    private static void BackpropVirtual(List<(Node Node, int Edge)> path, float leafValue)
    {
        float sign = -1f;
        for (int j = path.Count - 1; j >= 0; j--)
        {
            var (node, edge) = path[j];
            node.N[edge]++;
            node.W[edge] += sign * leafValue;
            node.Vloss[edge]--;
            sign = -sign;
        }
    }

    // Returns the value of 'state' from the perspective of its side to move.
    private static float Simulate<TState>(Node node, IZeroSumGame<TState> game, TState state,
        Evaluate<TState> evaluate, Config config)
    {
        if (node.Terminal) return node.TerminalValue;
        if (!node.Expanded) return ExpandLeaf(node, evaluate, state);

        int edge = SelectChild(node, config);
        var childState = game.Apply(state, node.Moves[edge]);
        node.Children[edge] ??= NewNode(game, childState);
        float value = -Simulate(node.Children[edge]!, game, childState, evaluate, config); // flip: child mover's value
        node.N[edge]++;
        node.W[edge] += value;
        return value;
    }

    private static int SelectChild(Node node, Config config)
    {
        int sumN = 0;
        for (int i = 0; i < node.N.Length; i++) sumN += node.N[i];
        float sqrtSum = MathF.Sqrt(sumN);

        int best = 0;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < node.Moves.Length; i++)
        {
            float q = node.N[i] > 0 ? node.W[i] / node.N[i] : 0f;
            float u = config.Cpuct * node.P[i] * sqrtSum / (1 + node.N[i]);
            float score = q + u;
            if (score > bestScore) { bestScore = score; best = i; }
        }
        return best;
    }

    private static Node NewNode<TState>(IZeroSumGame<TState> game, TState state)
    {
        var result = game.Result(state);
        if (result != GameResult.Ongoing)
            return new Node { Moves = [], Terminal = true, TerminalValue = result switch { GameResult.Win => 1f, GameResult.Loss => -1f, _ => 0f } };
        return new Node { Moves = [.. game.LegalMoves(state)] };
    }

    // Evaluate the leaf, seed per-move priors (masked to legal + renormalized), and return the net's value estimate.
    private static float ExpandLeaf<TState>(Node node, Evaluate<TState> evaluate, TState state)
    {
        var (priors, value) = evaluate(state);
        ExpandWith(node, priors);
        return value;
    }

    // Seed a leaf's per-move priors from an already-computed policy distribution (masked to legal + renormalized).
    // Shared by the sequential and batched paths; the latter supplies priors from a batched forward.
    private static void ExpandWith(Node node, float[] priors)
    {
        int k = node.Moves.Length;
        node.P = new float[k];
        node.N = new int[k];
        node.W = new float[k];
        node.Vloss = new int[k];
        node.Children = new Node?[k];

        float sum = 0f;
        for (int i = 0; i < k; i++) { node.P[i] = MathF.Max(priors[node.Moves[i]], 0f); sum += node.P[i]; }
        if (sum > 0f) { for (int i = 0; i < k; i++) node.P[i] /= sum; }
        else { for (int i = 0; i < k; i++) node.P[i] = 1f / k; } // net gave no mass to legal moves → uniform

        node.Expanded = true;
    }

    private static void AddRootNoise(Node root, Config config, Xoshiro256StarStar rng)
    {
        if (config.RootNoiseFrac <= 0f || root.Moves.Length == 0) return;
        var noise = SampleDirichlet(root.Moves.Length, config.DirichletAlpha, rng);
        float frac = config.RootNoiseFrac;
        for (int i = 0; i < root.P.Length; i++) root.P[i] = (1f - frac) * root.P[i] + frac * noise[i];
    }

    // ── Dirichlet sampling (Gamma via Marsaglia–Tsang; normals via Box–Muller), on the game's own RNG stream ──
    private static float[] SampleDirichlet(int k, double alpha, Xoshiro256StarStar rng)
    {
        var g = new double[k];
        double sum = 0;
        for (int i = 0; i < k; i++) { g[i] = SampleGamma(alpha, rng); sum += g[i]; }
        var d = new float[k];
        if (sum <= 0) { for (int i = 0; i < k; i++) d[i] = 1f / k; return d; }
        for (int i = 0; i < k; i++) d[i] = (float)(g[i] / sum);
        return d;
    }

    private static double SampleGamma(double alpha, Xoshiro256StarStar rng)
    {
        if (alpha < 1.0)
        {
            double u = rng.NextDouble();
            return SampleGamma(alpha + 1.0, rng) * Math.Pow(u <= 0 ? double.Epsilon : u, 1.0 / alpha);
        }
        double d = alpha - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double x = NextNormal(rng);
            double v = 1 + c * x;
            if (v <= 0) continue;
            v = v * v * v;
            double u2 = rng.NextDouble();
            if (u2 < 1 - 0.0331 * x * x * x * x) return d * v;
            if (Math.Log(u2 <= 0 ? double.Epsilon : u2) < 0.5 * x * x + d * (1 - v + Math.Log(v))) return d * v;
        }
    }

    private static double NextNormal(Xoshiro256StarStar rng)
    {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1 <= 0 ? double.Epsilon : u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

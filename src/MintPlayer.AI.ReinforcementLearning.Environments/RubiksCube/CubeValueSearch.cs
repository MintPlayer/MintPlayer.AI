using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Shortest-move inference on top of the teacher-free DAVI value net (a <see cref="ResidualMlp"/>
/// trained by <see cref="ValueIterationTrainer{TState}"/>): batch-weighted A* (BWAS) over the
/// <see cref="CubeModel"/>, guided by the net's learned cost-to-go. This is the "self-taught AI"
/// solver — it learned the cube purely from the forward model (no Kociemba), and with a real search
/// budget it returns quarter-turn-OPTIMAL solutions out to ~depth 15 (measured), beating Kociemba's
/// QTM on every solve. Counterpart to <see cref="CubePolicySearch"/> (the imitation policy net).
/// <para>
/// The search <c>weight</c> trades optimality for reach/speed: 1 is true A* (optimal under an
/// admissible value), &gt; 1 expands fewer nodes and reaches deeper for possibly-longer solutions.
/// The web default leans slightly greedy so an interactive solve stays responsive; deep scrambles
/// past the net's accurate band fail honestly (null) rather than returning a bad answer.
/// </para>
/// </summary>
public static class CubeValueSearch
{
    /// <summary>
    /// Weight 1.5 is the measured optimality sweet spot — it returned exact-length (optimal) solutions
    /// through depth 15 in the campaign diagnostic, where a greedier 1.8+ trades length for reach. The
    /// 50k expansion ceiling suits an offline/GPU caller; an interactive CPU caller should pass a tighter
    /// budget (each expansion is ~2 ms on the multithreaded CPU backend — see the web layer).
    /// </summary>
    public const float DefaultWeight = 1.5f;
    public const int DefaultMaxExpansions = 50_000;
    private const int ExpandBatch = 100;

    public sealed record SearchResult(bool Solved, string[] Moves);

    /// <summary>
    /// Solve <paramref name="start"/> in the fewest quarter-turns the net can find within
    /// <paramref name="maxExpansions"/> expansions. Empty move list ⇒ already solved; not-solved ⇒
    /// the search exhausted its budget (honest failure on a scramble past the net's reach).
    /// </summary>
    public static SearchResult Solve(
        ResidualMlp valueNet, FaceletCube start,
        int maxExpansions = DefaultMaxExpansions, float weight = DefaultWeight)
    {
        var model = new CubeModel();
        var solution = ValueGuidedSearch.SolveBatched(
            model, states => CostToGo(valueNet, states), start, maxExpansions, weight, ExpandBatch);

        if (solution is null) return new(false, []);
        return new(true, [.. solution.Select(a => FaceletCube.QuarterTurnMoves[a])]);
    }

    /// <summary>
    /// Cost-to-go (in moves, ≥ 0) for a batch of cubes in ONE forward — the BWAS hot path. The net was
    /// trained with <c>DistanceScale = 1</c>, so the raw scalar output IS the move estimate; clamp at 0
    /// (a negative cost-to-go is meaningless). No autograd tape during inference.
    /// </summary>
    private static float[] CostToGo(ResidualMlp valueNet, IReadOnlyList<FaceletCube> states)
    {
        int n = states.Count;
        var features = new float[n * RubiksCubeEnv.ObservationSize];
        for (int i = 0; i < n; i++)
            RubiksCubeEnv.WriteObservation(states[i], features.AsSpan(i * RubiksCubeEnv.ObservationSize));

        using (GradMode.NoGrad())
        {
            var values = valueNet.Forward(new Tensor(features, n, RubiksCubeEnv.ObservationSize));
            var result = new float[n];
            for (int i = 0; i < n; i++)
                result[i] = MathF.Max(0f, values.Data[i]);
            return result;
        }
    }
}

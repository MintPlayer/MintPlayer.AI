using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// Teacher-FREE imitation-data source (EfficientCube, <c>--game cube-policy</c>): the label is
/// derived from the scramble itself, never from a solver. A cube is scrambled by L random
/// quarter-turns s0(solved) → s1 → … → sL; reversing them is a known solution, so at state s_i
/// the move toward solved is the INVERSE of the i-th scramble move and the reverse-path length
/// is i. Contrast <see cref="CubeOracle"/>, which asks Kociemba "what would you do?" and is
/// therefore forever bounded by Kociemba: this signal has no teacher, so the trained solver is
/// limited only by the policy's generalization and the search budget — beam search stitches the
/// locally-confident moves into solutions shorter than the random scrambles it learned from.
/// </summary>
public static class CubeSelfSupervised
{
    /// <summary>
    /// Scrambles to a random depth (1..<paramref name="maxScrambleDepth"/>) and returns the states
    /// along the (known) reverse path, each labeled with the next quarter-turn toward solved (the
    /// inverse of the scramble move) and its distance back to solved. No solver is invoked.
    /// <para>
    /// DistanceToGo is the scramble-path length — an UPPER bound on the true distance (the random
    /// walk may cancel), which is all the value head needs as a soft search heuristic.
    /// </para>
    /// </summary>
    public static List<CubeOracle.LabeledState> LabelScramblePath(Xoshiro256StarStar rng, int maxScrambleDepth = 30)
    {
        int length = 1 + rng.NextInt(maxScrambleDepth);
        var cube = new FaceletCube();
        var states = new List<CubeOracle.LabeledState>(length);
        int lastAction = -1;
        for (int i = 0; i < length; i++)
        {
            // Random quarter-turn, never the immediate inverse of the previous one: a move that
            // directly cancels its predecessor would make "undo the last move" a wasted label.
            int action;
            do { action = rng.NextInt(RubiksCubeEnv.ActionCount); }
            while (action == RubiksCubeEnv.InverseAction(lastAction));

            cube.ApplyQuarterTurn(action);
            // From this state the move toward solved is the inverse of what we just applied, and we
            // are (i+1) moves deep along the reverse path.
            states.Add(new CubeOracle.LabeledState(cube.Facelets.ToArray(), RubiksCubeEnv.InverseAction(action), i + 1));
            lastAction = action;
        }
        return states;
    }
}

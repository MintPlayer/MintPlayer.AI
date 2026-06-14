using MintPlayer.AI.ReinforcementLearning.Core.Planning;

namespace MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;

/// <summary>
/// The cube as a deterministic planning model: quarter-turns are the actions, a solved cube
/// is the goal, and the facelet bytes are the state key. This is the teacher-free forward
/// model — no Kociemba — that <see cref="BreadthFirstPlanner"/> (optimal, for shallow
/// scrambles) and, later, a value-iteration learner query to look ahead.
/// </summary>
public sealed class CubeModel : IDeterministicModel<FaceletCube>
{
    public int ActionCount => RubiksCubeEnv.ActionCount; // 12 quarter-turns

    public bool IsGoal(FaceletCube state) => state.IsSolved;

    public FaceletCube Apply(FaceletCube state, int action)
    {
        var next = state.Clone();
        next.ApplyQuarterTurn(action);
        return next;
    }

    public string StateKey(FaceletCube state) => Convert.ToHexString(state.Facelets);
}

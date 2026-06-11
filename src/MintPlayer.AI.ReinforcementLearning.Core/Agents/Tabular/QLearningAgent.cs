using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Core.Agents.Tabular;

/// <summary>Off-policy TD(0) control: Q(s,a) += α · (r + γ·max_a' Q(s',a') − Q(s,a)).</summary>
public sealed class QLearningAgent(int stateCount, int actionCount, Xoshiro256StarStar rng)
    : TabularAgent(stateCount, actionCount, rng)
{
    public override void Observe(int state, int action, double reward, int nextState, bool terminated)
    {
        // Bootstrap only when the episode did NOT truly terminate. At a time-limit
        // truncation the caller passes terminated=false, so the target still bootstraps.
        double target = terminated ? reward : reward + Gamma * MaxQ(nextState);
        Q[state, action] += Alpha * (target - Q[state, action]);
    }
}

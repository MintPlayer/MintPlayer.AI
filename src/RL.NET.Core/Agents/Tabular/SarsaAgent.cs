using RLNet.Core.Random;

namespace RLNet.Core.Agents.Tabular;

/// <summary>
/// On-policy TD(0) control: Q(s,a) += α · (r + γ·Q(s',a') − Q(s,a)) where a' is the
/// action the current (epsilon-greedy) policy will actually take in s'. The agent picks
/// a' inside <see cref="Observe"/> and replays it on the next <see cref="Act"/> call so
/// the update and the behavior stay on-policy.
/// </summary>
public sealed class SarsaAgent(int stateCount, int actionCount, Xoshiro256StarStar rng)
    : TabularAgent(stateCount, actionCount, rng)
{
    private int _pendingState = -1;
    private int _pendingAction;

    public override int Act(int observation, bool greedy = false)
    {
        if (!greedy && observation == _pendingState)
        {
            _pendingState = -1;
            return _pendingAction;
        }
        _pendingState = -1;
        return base.Act(observation, greedy);
    }

    public override void Observe(int state, int action, double reward, int nextState, bool terminated)
    {
        double target;
        if (terminated)
        {
            target = reward;
            _pendingState = -1;
        }
        else
        {
            _pendingAction = base.Act(nextState);
            _pendingState = nextState;
            target = reward + Gamma * Q[nextState, _pendingAction];
        }
        Q[state, action] += Alpha * (target - Q[state, action]);
    }
}

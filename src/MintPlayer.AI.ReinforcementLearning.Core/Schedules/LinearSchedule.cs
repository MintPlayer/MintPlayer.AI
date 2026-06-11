namespace MintPlayer.AI.ReinforcementLearning.Core.Schedules;

/// <summary>Linearly interpolates from Start to End over DecaySteps, then stays at End.</summary>
public sealed record LinearSchedule(double Start, double End, int DecaySteps)
{
    public double Value(int step)
    {
        if (step >= DecaySteps) return End;
        double fraction = (double)step / DecaySteps;
        return Start + fraction * (End - Start);
    }
}

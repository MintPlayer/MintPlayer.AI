using System.Text;
using RLNet.Core.Environments;
using RLNet.Core.Random;

namespace RLNet.Environments;

/// <summary>
/// Faithful port of Gymnasium CartPole-v1 (classic_control/cartpole.py), verified against
/// recorded reference trajectories (tests/Fixtures/cartpole_golden.json).
/// Porting traps the constants encode: <c>Length</c> is HALF the pole length; the Euler
/// update applies positions BEFORE velocities (old derivatives); reward is +1 on every
/// step including the terminating one; termination angle is 12° while the observation
/// bound is 24°. Solved: mean return ≥ 475 over 100 consecutive episodes.
/// </summary>
public sealed class CartPoleEnv : IEnvironment<float[], int>
{
    public const double Gravity = 9.8;
    public const double MassCart = 1.0;
    public const double MassPole = 0.1;
    public const double TotalMass = MassCart + MassPole;
    public const double Length = 0.5; // half the pole length
    public const double PoleMassLength = MassPole * Length;
    public const double ForceMag = 10.0;
    public const double Tau = 0.02;
    public const double ThetaThresholdRadians = 12 * 2 * Math.PI / 360;
    public const double XThreshold = 2.4;
    public const int MaxEpisodeSteps = 500;

    private Xoshiro256StarStar _rng = new(0);
    private double _x, _xDot, _theta, _thetaDot;
    private int _elapsedSteps;
    private bool _done = true;

    public CartPoleEnv()
    {
        ObservationSpace = new BoxSpace(
            [(float)(-XThreshold * 2), float.NegativeInfinity, (float)(-ThetaThresholdRadians * 2), float.NegativeInfinity],
            [(float)(XThreshold * 2), float.PositiveInfinity, (float)(ThetaThresholdRadians * 2), float.PositiveInfinity]);
        ActionSpace = new DiscreteSpace(2);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        _x = _rng.NextDouble(-0.05, 0.05);
        _xDot = _rng.NextDouble(-0.05, 0.05);
        _theta = _rng.NextDouble(-0.05, 0.05);
        _thetaDot = _rng.NextDouble(-0.05, 0.05);
        _elapsedSteps = 0;
        _done = false;
        return (Observation(), EnvInfo.Empty);
    }

    /// <summary>Pins the physics state directly (golden-trajectory tests, demos).</summary>
    public void SetState(double x, double xDot, double theta, double thetaDot)
    {
        (_x, _xDot, _theta, _thetaDot) = (x, xDot, theta, thetaDot);
        _elapsedSteps = 0;
        _done = false;
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        if (!ActionSpace.Contains(action))
            throw new ArgumentOutOfRangeException(nameof(action));

        double force = action == 1 ? ForceMag : -ForceMag;
        double cosTheta = Math.Cos(_theta);
        double sinTheta = Math.Sin(_theta);

        double temp = (force + PoleMassLength * _thetaDot * _thetaDot * sinTheta) / TotalMass;
        double thetaAcc = (Gravity * sinTheta - cosTheta * temp)
            / (Length * (4.0 / 3.0 - MassPole * cosTheta * cosTheta / TotalMass));
        double xAcc = temp - PoleMassLength * thetaAcc * cosTheta / TotalMass;

        // Gymnasium's default "euler" integrator: positions first, with the OLD velocities.
        _x += Tau * _xDot;
        _xDot += Tau * xAcc;
        _theta += Tau * _thetaDot;
        _thetaDot += Tau * thetaAcc;
        _elapsedSteps++;

        bool terminated = Math.Abs(_x) > XThreshold || Math.Abs(_theta) > ThetaThresholdRadians;
        bool truncated = !terminated && _elapsedSteps >= MaxEpisodeSteps;
        _done = terminated || truncated;
        return new StepResult<float[]>(Observation(), 1.0, terminated, truncated, EnvInfo.Empty);
    }

    private float[] Observation() => [(float)_x, (float)_xDot, (float)_theta, (float)_thetaDot];

    public string RenderString()
    {
        const int width = 61, poleRows = 6;
        var sb = new StringBuilder();
        int cartCol = (int)Math.Round((_x + XThreshold) / (2 * XThreshold) * (width - 1));
        cartCol = Math.Clamp(cartCol, 2, width - 3);

        for (int row = poleRows; row >= 1; row--)
        {
            int offset = (int)Math.Round(Math.Sin(_theta) * row * 1.6);
            int col = Math.Clamp(cartCol + offset, 0, width - 1);
            char glyph = Math.Abs(_theta) < 0.04 ? '|' : _theta > 0 ? '/' : '\\';
            sb.Append(' ', col).Append(glyph).AppendLine();
        }
        sb.Append(' ', cartCol - 2).Append("#####").AppendLine();
        sb.Append('─', width).AppendLine();
        return sb.ToString();
    }
}

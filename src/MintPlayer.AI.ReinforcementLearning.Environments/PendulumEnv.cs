using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments;

/// <summary>
/// Faithful port of Gymnasium Pendulum-v1 (classic_control/pendulum.py): swing a frictionless rod upright
/// and hold it there by applying a bounded torque. The SDK's first <b>continuous-action</b> environment —
/// observation [cos θ, sin θ, θ̇] (Box, 3-dim), action a single torque ∈ [−2, 2] (Box, 1-dim). Reward is the
/// negative cost −(θ² + 0.1·θ̇² + 0.001·τ²) measured from the angle BEFORE the integration step (0 upright,
/// at rest, no torque is the best attainable). There is no terminal state — every episode is truncated at the
/// 200-step cap. A competent SAC policy reaches a mean return around −150; random ≈ −1200.
/// </summary>
public sealed class PendulumEnv : IEnvironment<float[], float[]>, IStatefulEnvironment
{
    public const double Gravity = 10.0;
    public const double Mass = 1.0;
    public const double Length = 1.0;
    public const double Dt = 0.05;
    public const double MaxTorque = 2.0;
    public const double MaxSpeed = 8.0;
    public const int DefaultMaxEpisodeSteps = 200;

    private readonly int _maxEpisodeSteps;

    private Xoshiro256StarStar _rng = new(0);
    private double _theta, _thetaDot;
    private int _elapsedSteps;
    private bool _done = true;

    public PendulumEnv(int maxEpisodeSteps = DefaultMaxEpisodeSteps)
    {
        _maxEpisodeSteps = maxEpisodeSteps;
        // Observation is already well-scaled: cos/sin ∈ [-1,1] and θ̇ ∈ [-8,8], so (unlike MountainCar) no
        // extra normalization is needed for a dense net to see every component.
        ObservationSpace = new BoxSpace([-1f, -1f, (float)-MaxSpeed], [1f, 1f, (float)MaxSpeed]);
        ActionSpace = new BoxSpace((float)-MaxTorque, (float)MaxTorque, 1);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<float[]> ActionSpace { get; }

    public double Theta => _theta;
    public double AngularVelocity => _thetaDot;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        _theta = _rng.NextDouble(-Math.PI, Math.PI);
        _thetaDot = _rng.NextDouble(-1.0, 1.0);
        _elapsedSteps = 0;
        _done = false;
        return (Observation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(float[] action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");

        // Continuous control clamps out-of-range torque rather than throwing (Gym clips; "define errors out
        // of existence" — a policy whose σ pushes past the bound shouldn't crash the env).
        double torque = Math.Clamp(action.Length > 0 ? action[0] : 0.0, -MaxTorque, MaxTorque);

        // Cost uses the angle BEFORE integration, normalized to (−π, π].
        double normTheta = NormalizeAngle(_theta);
        double cost = normTheta * normTheta + 0.1 * _thetaDot * _thetaDot + 0.001 * torque * torque;

        // Gymnasium semi-implicit Euler: θ advances using the NEW angular velocity.
        double newThetaDot = _thetaDot + (3.0 * Gravity / (2.0 * Length) * Math.Sin(_theta)
            + 3.0 / (Mass * Length * Length) * torque) * Dt;
        newThetaDot = Math.Clamp(newThetaDot, -MaxSpeed, MaxSpeed);
        _theta += newThetaDot * Dt;
        _thetaDot = newThetaDot;
        _elapsedSteps++;

        bool truncated = _elapsedSteps >= _maxEpisodeSteps;
        _done = truncated;
        return new StepResult<float[]>(Observation(), -cost, false, truncated, EnvInfo.Empty);
    }

    /// <summary>Pins the physics state directly (golden-trajectory tests, demos).</summary>
    public void SetState(double theta, double thetaDot)
    {
        (_theta, _thetaDot) = (theta, thetaDot);
        _elapsedSteps = 0;
        _done = false;
    }

    private float[] Observation() => [(float)Math.Cos(_theta), (float)Math.Sin(_theta), (float)_thetaDot];

    // Wrap to (−π, π], matching Gym's angle_normalize.
    private static double NormalizeAngle(double theta)
    {
        double wrapped = (theta + Math.PI) % (2.0 * Math.PI);
        if (wrapped < 0) wrapped += 2.0 * Math.PI;
        return wrapped - Math.PI;
    }

    public byte[] SaveState()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        var (s0, s1, s2, s3) = _rng.GetState();
        writer.Write(s0); writer.Write(s1); writer.Write(s2); writer.Write(s3);
        writer.Write(_theta);
        writer.Write(_thetaDot);
        writer.Write(_elapsedSteps);
        writer.Write(_done);
        writer.Flush();
        return stream.ToArray();
    }

    public void RestoreState(byte[] state)
    {
        using var reader = new BinaryReader(new MemoryStream(state));
        _rng.SetState(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64());
        _theta = reader.ReadDouble();
        _thetaDot = reader.ReadDouble();
        _elapsedSteps = reader.ReadInt32();
        _done = reader.ReadBoolean();
    }

    public string RenderString()
    {
        // A clock-face glyph: the rod points toward the current angle (θ=0 is straight up).
        const int width = 21;
        double x = Math.Sin(_theta), y = Math.Cos(_theta);
        int col = (int)Math.Round((x + 1) / 2 * (width - 1));
        var sb = new StringBuilder();
        sb.Append(y >= 0 ? "up   " : "down ");
        for (int i = 0; i < width; i++) sb.Append(i == col ? 'O' : '-');
        return sb.ToString();
    }
}

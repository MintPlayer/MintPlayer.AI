using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments;

/// <summary>
/// Faithful port of Gymnasium MountainCar-v0 (classic_control/mountain_car.py): an underpowered car must
/// build momentum by swinging back and forth to escape a valley. State [position, velocity], 3 actions
/// (push left / none / right), reward −1 every step, terminated at the goal (position ≥ 0.5), truncated at
/// the step cap. Solved: mean return ≥ −110 over 100 episodes.
/// <para>
/// Two training aids (off by default to keep the env v0-faithful for eval): <c>maxEpisodeSteps</c>
/// can be raised above 200 during training so a fresh high-entropy policy ever reaches the goal (the −1 reward
/// is otherwise flat and PPO never bootstraps), and <c>shapeReward</c> adds a small speed-gain bonus as a
/// fallback if plain PPO stalls. Gate/eval always use the standard 200-step, unshaped env.
/// </para>
/// </summary>
public sealed class MountainCarEnv : IEnvironment<float[], int>, IStatefulEnvironment
{
    public const double MinPosition = -1.2;
    public const double MaxPosition = 0.6;
    public const double MaxSpeed = 0.07;
    public const double GoalPosition = 0.5;
    public const double Force = 0.001;
    public const double Gravity = 0.0025;
    public const int DefaultMaxEpisodeSteps = 200;
    // Speed bonus (training only): reward += ShapeScale·|velocity|, comparable in magnitude to the −1 step
    // cost (|v| ≤ 0.07 → bonus ≤ ~0.9). Drives the back-and-forth that builds momentum to escape the valley.
    private const double ShapeScale = 13.0;

    private readonly int _maxEpisodeSteps;
    private readonly bool _shapeReward;

    private Xoshiro256StarStar _rng = new(0);
    private double _position, _velocity;
    private int _elapsedSteps;
    private bool _done = true;

    public MountainCarEnv(int maxEpisodeSteps = DefaultMaxEpisodeSteps, bool shapeReward = false)
    {
        _maxEpisodeSteps = maxEpisodeSteps;
        _shapeReward = shapeReward;
        // Observation is NORMALISED to ~[-1,1] (see Observation): raw velocity (~0.07) is ~14× smaller than
        // raw position, so an un-normalised dense net effectively can't see velocity — the signal it must
        // condition on to pump correctly. Normalising both is what makes the swing-up learnable.
        ObservationSpace = new BoxSpace([-1f, -1f], [1f, 1f]);
        ActionSpace = new DiscreteSpace(3);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    public double Position => _position;
    public double Velocity => _velocity;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        _position = _rng.NextDouble(-0.6, -0.4);
        _velocity = 0.0;
        _elapsedSteps = 0;
        _done = false;
        return (Observation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        if (!ActionSpace.Contains(action))
            throw new ArgumentOutOfRangeException(nameof(action));

        double velocity = _velocity + (action - 1) * Force + Math.Cos(3 * _position) * -Gravity;
        velocity = Math.Clamp(velocity, -MaxSpeed, MaxSpeed);
        double position = Math.Clamp(_position + velocity, MinPosition, MaxPosition);
        if (position <= MinPosition && velocity < 0) velocity = 0.0; // inelastic left wall

        _position = position;
        _velocity = velocity;
        _elapsedSteps++;

        bool terminated = position >= GoalPosition;
        bool truncated = !terminated && _elapsedSteps >= _maxEpisodeSteps;
        _done = terminated || truncated;

        double reward = -1.0;
        if (_shapeReward) reward += ShapeScale * Math.Abs(velocity); // speed bonus (training aid)
        return new StepResult<float[]>(Observation(), reward, terminated, truncated, EnvInfo.Empty);
    }

    /// <summary>Pins the physics state directly (golden-trajectory tests, demos).</summary>
    public void SetState(double position, double velocity)
    {
        (_position, _velocity) = (position, velocity);
        _elapsedSteps = 0;
        _done = false;
    }

    // Normalised to ~[-1,1]: position centred on −0.3 (half-range 0.9), velocity scaled by MaxSpeed.
    private float[] Observation() => [(float)((_position + 0.3) / 0.9), (float)(_velocity / MaxSpeed)];

    public byte[] SaveState()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        var (s0, s1, s2, s3) = _rng.GetState();
        writer.Write(s0); writer.Write(s1); writer.Write(s2); writer.Write(s3);
        writer.Write(_position);
        writer.Write(_velocity);
        writer.Write(_elapsedSteps);
        writer.Write(_done);
        writer.Flush();
        return stream.ToArray();
    }

    public void RestoreState(byte[] state)
    {
        using var reader = new BinaryReader(new MemoryStream(state));
        _rng.SetState(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64());
        _position = reader.ReadDouble();
        _velocity = reader.ReadDouble();
        _elapsedSteps = reader.ReadInt32();
        _done = reader.ReadBoolean();
    }

    public string RenderString()
    {
        const int width = 41;
        int col = (int)Math.Round((_position - MinPosition) / (MaxPosition - MinPosition) * (width - 1));
        var sb = new StringBuilder();
        for (int i = 0; i < width; i++) sb.Append(i == col ? 'O' : i == width - 1 ? '|' : '_');
        return sb.ToString();
    }
}

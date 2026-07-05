using System.Text;
using MintPlayer.AI.ReinforcementLearning.Core.Environments;
using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments;

/// <summary>
/// Gymnasium MountainCar-v0 as an RL environment — a <b>public facade</b> over the single-source transpiled core
/// (<c>MountainCar/polyglot/mountaincar_solver.pg</c> → <c>PgMountainCarEnv</c>): the dynamics + normalised
/// observation live <b>once</b> in the <c>.pg</c>, shared with the browser's <c>mountaincar_solver.ts</c> (M33).
/// The facade re-adds host concerns: the <see cref="IEnvironment{T,U}"/>/<see cref="IStatefulEnvironment"/> API, the
/// start-state RNG (a seeded <see cref="Xoshiro256StarStar"/>, kept out of the single source), state (de)serialization,
/// and the throw-on-illegal-action contract. The C# dynamics stay byte-identical to the original (same
/// <c>Math.Cos</c>); the browser's <c>cos</c>/<c>tanh</c> differ by ≤1 ULP but that's harmless (argmax decision, no
/// server twin). State [position, velocity], 3 actions (push left/none/right), reward −1/step, terminated at the goal.
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

    private Xoshiro256StarStar _rng = new(0);
    private readonly PgMountainCarEnv _core;

    public MountainCarEnv(int maxEpisodeSteps = DefaultMaxEpisodeSteps, bool shapeReward = false)
    {
        _core = new PgMountainCarEnv(maxEpisodeSteps, shapeReward);
        ObservationSpace = new BoxSpace([-1f, -1f], [1f, 1f]);
        ActionSpace = new DiscreteSpace(3);
    }

    public Space<float[]> ObservationSpace { get; }
    public Space<int> ActionSpace { get; }

    public double Position => _core.position;
    public double Velocity => _core.velocity;

    public (float[] Observation, EnvInfo Info) Reset(ulong? seed = null)
    {
        if (seed.HasValue)
            _rng = new Xoshiro256StarStar(seed.Value);
        _core.reset(_rng.NextDouble(-0.6, -0.4));
        return (Observation(), EnvInfo.Empty);
    }

    public StepResult<float[]> Step(int action)
    {
        if (_core.done)
            throw new InvalidOperationException("Episode is done; call Reset() before stepping.");
        if (!ActionSpace.Contains(action))
            throw new ArgumentOutOfRangeException(nameof(action));

        _core.step(action);
        return new StepResult<float[]>(Observation(), _core.lastReward, _core.lastTerminated, _core.lastTruncated, EnvInfo.Empty);
    }

    /// <summary>Pins the physics state directly (golden-trajectory tests, demos).</summary>
    public void SetState(double position, double velocity) => _core.setState(position, velocity);

    private float[] Observation()
    {
        var core = _core.buildObservation();
        return [(float)core[0], (float)core[1]];
    }

    public byte[] SaveState()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        var (s0, s1, s2, s3) = _rng.GetState();
        writer.Write(s0); writer.Write(s1); writer.Write(s2); writer.Write(s3);
        writer.Write(_core.position);
        writer.Write(_core.velocity);
        writer.Write(_core.elapsedSteps);
        writer.Write(_core.done);
        writer.Flush();
        return stream.ToArray();
    }

    public void RestoreState(byte[] state)
    {
        using var reader = new BinaryReader(new MemoryStream(state));
        _rng.SetState(reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64(), reader.ReadUInt64());
        _core.position = reader.ReadDouble();
        _core.velocity = reader.ReadDouble();
        _core.elapsedSteps = reader.ReadInt32();
        _core.done = reader.ReadBoolean();
    }

    public string RenderString()
    {
        const int width = 41;
        int col = (int)Math.Round((_core.position - MinPosition) / (MaxPosition - MinPosition) * (width - 1));
        var sb = new StringBuilder();
        for (int i = 0; i < width; i++) sb.Append(i == col ? 'O' : i == width - 1 ? '|' : '_');
        return sb.ToString();
    }
}

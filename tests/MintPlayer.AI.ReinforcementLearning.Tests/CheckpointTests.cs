using MintPlayer.AI.ReinforcementLearning.Core.Agents.Tabular;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Numerics;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Schedules;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments;
using MintPlayer.AI.ReinforcementLearning.Environments.Game2048;
using Xunit;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

public class CheckpointTests
{
    // ---------- MLP ----------

    [Fact]
    public void Mlp_RoundTrip_IsBitwiseIdentical()
    {
        var original = new Mlp([4, 32, 16, 2], new Xoshiro256StarStar(123), Activation.Relu);

        using var stream = new MemoryStream();
        MlpCheckpoint.Save(original, stream);
        stream.Position = 0;
        var restored = MlpCheckpoint.Load(stream);

        Assert.Equal(original.Sizes, restored.Sizes);
        Assert.Equal(original.HiddenActivation, restored.HiddenActivation);
        AssertParametersBitwiseEqual(original, restored);

        // Forward pass on the restored net is bitwise identical too.
        var input = Tensor.RandomNormal(new Xoshiro256StarStar(7), 0f, 1f, 3, 4);
        using (GradMode.NoGrad())
            Assert.Equal(original.Forward(input).Data, restored.Forward(input).Data);
    }

    [Fact]
    public void Mlp_Load_RejectsWrongKind()
    {
        using var stream = new MemoryStream();
        var agent = new NTuple2048Agent();
        agent.Save(stream);
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => MlpCheckpoint.Load(stream));
    }

    // ---------- Adam ----------

    [Fact]
    public void Adam_RoundTrip_ContinuesTrainingBitwiseIdentically()
    {
        // Train net A for a few steps so Adam accumulates non-trivial moments.
        var netA = new Mlp([3, 8, 2], new Xoshiro256StarStar(5), Activation.Tanh);
        var adamA = new Adam(netA.Parameters(), learningRate: 0.01f);
        TrainSteps(netA, adamA, steps: 5);

        // Serialize net + optimizer, restore into B.
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            MlpCheckpoint.Write(netA, writer);
            AdamCheckpoint.Write(adamA, writer);
        }
        stream.Position = 0;
        Mlp netB;
        Adam adamB;
        using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            netB = MlpCheckpoint.Read(reader);
            adamB = AdamCheckpoint.Read(netB.Parameters(), reader);
        }

        // Continue training both identically: only preserved moments + step count keep them in lockstep.
        TrainSteps(netA, adamA, steps: 5);
        TrainSteps(netB, adamB, steps: 5);
        AssertParametersBitwiseEqual(netA, netB);
    }

    private static void TrainSteps(Mlp net, Adam adam, int steps)
    {
        var input = Tensor.RandomNormal(new Xoshiro256StarStar(11), 0f, 1f, 4, 3);
        var target = Tensor.RandomNormal(new Xoshiro256StarStar(13), 0f, 1f, 4, 2);
        for (int i = 0; i < steps; i++)
        {
            adam.ZeroGrad();
            net.Forward(input).MseLoss(target).Backward();
            adam.Step();
        }
    }

    private static void AssertParametersBitwiseEqual(Mlp a, Mlp b)
    {
        var pa = a.Parameters().ToArray();
        var pb = b.Parameters().ToArray();
        Assert.Equal(pa.Length, pb.Length);
        for (int i = 0; i < pa.Length; i++)
            Assert.Equal(pa[i].Data, pb[i].Data); // float[] equality = element-exact
    }

    // ---------- Tabular (JSON) ----------

    [Fact]
    public void TabularQTable_JsonRoundTrip_IsExact()
    {
        var env = new FrozenLakeEnv();
        var agent = new QLearningAgent(env.StateCount, env.ActionCount, new Xoshiro256StarStar(1)) { Gamma = 0.99 };
        TabularTrainer.Train(env, agent, new TabularTrainingOptions
        {
            Episodes = 500,
            Epsilon = new LinearSchedule(1.0, 0.1, 400),
            Alpha = new LinearSchedule(0.5, 0.1, 400),
        }, envSeed: 2);

        using var stream = new MemoryStream();
        TabularCheckpoint.Save(agent, stream);
        stream.Position = 0;
        var restored = new QLearningAgent(env.StateCount, env.ActionCount, new Xoshiro256StarStar(99));
        TabularCheckpoint.LoadInto(restored, stream);

        for (int s = 0; s < env.StateCount; s++)
            for (int a = 0; a < env.ActionCount; a++)
                Assert.Equal(agent.Q[s, a], restored.Q[s, a]); // exact double equality
    }

    [Fact]
    public void TabularCheckpoint_RejectsDimensionMismatch()
    {
        var agent = new QLearningAgent(16, 4, new Xoshiro256StarStar(1));
        using var stream = new MemoryStream();
        TabularCheckpoint.Save(agent, stream);
        stream.Position = 0;
        var wrongShape = new QLearningAgent(25, 4, new Xoshiro256StarStar(1));
        Assert.Throws<InvalidDataException>(() => TabularCheckpoint.LoadInto(wrongShape, stream));
    }

    // ---------- 2048 n-tuple ----------

    [Fact]
    public void NTuple_RoundTrip_PlaysBitwiseIdenticalGames()
    {
        var original = new NTuple2048Agent();
        var trainRng = new Xoshiro256StarStar(42);
        for (int game = 0; game < 50; game++)
            original.PlayGame(trainRng, learn: true);

        using var stream = new MemoryStream();
        original.Save(stream);
        stream.Position = 0;
        var restored = NTuple2048Agent.Load(stream);

        Assert.Equal(original.Alpha, restored.Alpha);
        var rngA = new Xoshiro256StarStar(7);
        var rngB = new Xoshiro256StarStar(7);
        for (int game = 0; game < 20; game++)
            Assert.Equal(original.PlayGame(rngA, learn: false), restored.PlayGame(rngB, learn: false));
    }

    // ---------- Environment state snapshot ----------

    [Fact]
    public void CartPole_StateRoundTrip_ContinuesBitwiseIdentically()
    {
        var envA = new CartPoleEnv();
        envA.Reset(seed: 31);
        for (int i = 0; i < 20; i++)
            envA.Step(i % 2);

        var envB = new CartPoleEnv();
        envB.RestoreState(envA.SaveState());

        // Identical physics from here on...
        for (int i = 0; i < 50; i++)
        {
            var a = envA.Step(i % 3 == 0 ? 1 : 0);
            var b = envB.Step(i % 3 == 0 ? 1 : 0);
            Assert.Equal(a.Observation, b.Observation);
            Assert.Equal(a.Terminated, b.Terminated);
            if (a.Done) break;
        }

        // ...and the captured RNG makes the NEXT seedless reset identical too.
        Assert.Equal(envA.Reset().Observation, envB.Reset().Observation);
    }

    // ---------- DQN full resume (the M7 gate) ----------

    [Fact]
    public void DqnResume_BitwiseMatchesUninterruptedRun()
    {
        const ulong masterSeed = 7;
        DqnOptions Options(int maxSteps) => new()
        {
            Hidden = [32, 32],
            MaxSteps = maxSteps,
            WarmupSteps = 500,
            TargetSyncEvery = 250,
            EvalEvery = 2_000, // exercises the eval + env-reset path before the checkpoint
            EvalEpisodes = 5,
        };

        // Run A: 4,000 steps, uninterrupted.
        var resultA = DqnTrainer.Train(new CartPoleEnv(), Options(4_000), new SeedSequence(masterSeed));

        // Run B: 2,000 steps, serialize the full state, deserialize, resume on a FRESH env to 4,000.
        var resultB1 = DqnTrainer.Train(new CartPoleEnv(), Options(2_000), new SeedSequence(masterSeed));
        using var stream = new MemoryStream();
        resultB1.State.Save(stream);
        stream.Position = 0;
        var restored = DqnTrainingState.Load(stream);
        var resultB2 = DqnTrainer.Train(new CartPoleEnv(), Options(4_000), new SeedSequence(masterSeed), resume: restored);

        AssertParametersBitwiseEqual(resultA.Network, resultB2.Network);
        AssertParametersBitwiseEqual(resultA.State.Target, resultB2.State.Target);
        Assert.Equal(resultA.FinalEvalReturn, resultB2.FinalEvalReturn);
        Assert.Equal(resultA.State.PolicyRng.GetState(), resultB2.State.PolicyRng.GetState());
        Assert.Equal(resultA.State.BufferRng.GetState(), resultB2.State.BufferRng.GetState());
        Assert.Equal(resultA.State.EnvState, resultB2.State.EnvState);
    }

    // ---------- File model store ----------

    [Fact]
    public void FileModelStore_SaveLoadListDelete()
    {
        string root = Path.Combine(Path.GetTempPath(), "rlnet-store-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            var store = new FileModelStore(root);
            Assert.False(store.Exists("cartpole", "dqn"));
            Assert.Null(store.TryOpenRead("cartpole", "dqn"));
            Assert.Empty(store.List());

            var network = new Mlp([4, 8, 2], new Xoshiro256StarStar(3), Activation.Relu);
            store.Save("cartpole", "dqn", s => MlpCheckpoint.Save(network, s));

            Assert.True(store.Exists("cartpole", "dqn"));
            Assert.Equal(new[] { ("cartpole", "dqn") }, store.List());
            using (var stream = store.TryOpenRead("cartpole", "dqn"))
            {
                Assert.NotNull(stream);
                AssertParametersBitwiseEqual(network, MlpCheckpoint.Load(stream!));
            }

            // Overwrite replaces atomically (no leftover temp files).
            var network2 = new Mlp([4, 8, 2], new Xoshiro256StarStar(4), Activation.Relu);
            store.Save("cartpole", "dqn", s => MlpCheckpoint.Save(network2, s));
            using (var stream = store.TryOpenRead("cartpole", "dqn"))
                AssertParametersBitwiseEqual(network2, MlpCheckpoint.Load(stream!));
            Assert.Empty(Directory.GetFiles(root, "*.tmp"));

            Assert.True(store.Delete("cartpole", "dqn"));
            Assert.False(store.Exists("cartpole", "dqn"));
            Assert.False(store.Delete("cartpole", "dqn"));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void FileModelStore_FailedSave_KeepsPreviousCheckpoint()
    {
        string root = Path.Combine(Path.GetTempPath(), "rlnet-store-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            var store = new FileModelStore(root);
            var network = new Mlp([2, 4, 2], new Xoshiro256StarStar(3), Activation.Relu);
            store.Save("env", "algo", s => MlpCheckpoint.Save(network, s));

            Assert.Throws<InvalidOperationException>(() =>
                store.Save("env", "algo", _ => throw new InvalidOperationException("boom")));

            using var stream = store.TryOpenRead("env", "algo");
            AssertParametersBitwiseEqual(network, MlpCheckpoint.Load(stream!));
            Assert.Empty(Directory.GetFiles(root, "*.tmp"));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void FileModelStore_RejectsInvalidIds()
    {
        var store = new FileModelStore(Path.GetTempPath());
        Assert.Throws<ArgumentException>(() => store.Exists("../escape", "dqn"));
        Assert.Throws<ArgumentException>(() => store.Exists("cartpole", "a.b"));
    }
}

using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Core.Nn;
using MintPlayer.AI.ReinforcementLearning.Core.Random;
using MintPlayer.AI.ReinforcementLearning.Core.Training;
using MintPlayer.AI.ReinforcementLearning.Environments.RubiksCube;
using RLDemo.Web.Services;

namespace MintPlayer.AI.ReinforcementLearning.Tests;

// ── Shared doubles ────────────────────────────────────────────────────────────

/// <summary>A model store held entirely in memory, so a checkpoint test costs no disk and no timestamps.</summary>
internal sealed class FakeModelStore : IModelStore
{
    private byte[]? _bytes;

    /// <summary>How many times a reader actually opened the checkpoint — the reload cadence is observed here.</summary>
    public int OpenCount { get; private set; }

    /// <summary>The last stream handed out, so a test can prove the reader disposed it.</summary>
    public TrackedStream? LastStream { get; private set; }

    public void Put(byte[] bytes) => _bytes = bytes;

    public void Clear() => _bytes = null;

    public bool Exists(string environmentId, string algorithmId) => _bytes is not null;

    public Stream? TryOpenRead(string environmentId, string algorithmId)
    {
        OpenCount++;
        if (_bytes is null) return null;
        return LastStream = new TrackedStream(_bytes);
    }

    public void Save(string environmentId, string algorithmId, Action<Stream> write)
    {
        var buffer = new MemoryStream();
        write(buffer);
        _bytes = buffer.ToArray();
    }

    public IReadOnlyList<(string EnvironmentId, string AlgorithmId)> List()
        => _bytes is null ? [] : [("env", "algo")];

    public bool Delete(string environmentId, string algorithmId)
    {
        bool had = _bytes is not null;
        _bytes = null;
        return had;
    }
}

/// <summary>A MemoryStream that remembers it was disposed — the checkpoint readers must not leak a file handle.</summary>
internal sealed class TrackedStream(byte[] bytes) : MemoryStream(bytes)
{
    public bool Disposed { get; private set; }

    protected override void Dispose(bool disposing)
    {
        Disposed = true;
        base.Dispose(disposing);
    }
}

/// <summary>A logger that swallows everything. Nothing here asserts on log text, and Console must stay untouched.</summary>
internal sealed class SilentLogger : ILogger
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

    public bool IsEnabled(LogLevel logLevel) => false;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
                            Func<TState, Exception?, string> formatter)
    { }
}

/// <summary>A startup service that records that it ran, and on which cancellation token.</summary>
internal sealed class RecordingStartupService : IModelStartupService
{
    public ManualResetEventSlim Done { get; } = new(false);

    public int Calls;

    public void Initialize(CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref Calls);
        Done.Set();
    }
}

// ── The tests ─────────────────────────────────────────────────────────────────

/// <summary>
/// The web tier's checkpoint plumbing (<c>RLDemo.Web/Services/ModelServiceInfrastructure.cs</c>), which every
/// game's model service composes but which nothing exercised directly.
/// </summary>
/// <remarks>
/// <para>The silent failure this guards is a readiness lie: <c>Status</c> is what the <c>/status</c> endpoints
/// and the <c>IsReady</c> guards report, so if a checkpoint that was never loaded could read back as
/// <see cref="ModelStatus.Ready"/>, the web would hand the game page a null model and 500 on the first click
/// instead of saying "unavailable".</para>
/// <para>The second is the refresh cadence. <c>RefreshingCheckpoint</c> is deliberately allowed to swallow a
/// corrupt or mid-write checkpoint and keep the previous net; the risk is that it swallows too much (dropping a
/// good net) or too little (an exception escaping onto a solve request). Both directions are pinned here.</para>
/// <para>Time is controlled through the constructor's <c>refresh</c> cadence, never by sleeping: a zero cadence
/// means "reload on every read" and the default cadence means "never reload during a test".</para>
/// </remarks>
public class ModelServiceInfrastructureTests
{
    private static readonly byte[] SomeCheckpoint = [1, 2, 3, 4];

    private sealed record Payload(int Length);

    private static Payload ReadPayload(Stream stream) => new((int)stream.Length);

    // ── StartupCheckpoint ──

    [Fact]
    public void Startup_reports_Loading_before_anything_is_initialized()
    {
        var store = new FakeModelStore();
        var checkpoint = new StartupCheckpoint<Payload>(store, "env", "algo", ReadPayload, new SilentLogger(), "test model");

        Assert.Equal(ModelStatus.Loading, checkpoint.Status);
        Assert.Null(checkpoint.Error);
        Assert.Equal(0, store.OpenCount); // constructing must not touch the store
    }

    [Fact]
    public void Startup_TryLoad_is_false_when_the_store_holds_no_checkpoint()
    {
        var store = new FakeModelStore();
        var checkpoint = new StartupCheckpoint<Payload>(store, "env", "algo", ReadPayload, new SilentLogger(), "test model");

        Assert.False(checkpoint.TryLoad());
        Assert.Equal(ModelStatus.Loading, checkpoint.Status); // a plain TryLoad miss is not a failure verdict
        Assert.Null(checkpoint.Error);
    }

    [Fact]
    public void Startup_Initialize_reaches_Ready_with_a_non_null_value()
    {
        var store = new FakeModelStore();
        store.Put(SomeCheckpoint);
        var checkpoint = new StartupCheckpoint<Payload>(store, "env", "algo", ReadPayload, new SilentLogger(), "test model");

        checkpoint.Initialize();

        Assert.Equal(ModelStatus.Ready, checkpoint.Status);
        Assert.Null(checkpoint.Error);
        Assert.Equal(new Payload(SomeCheckpoint.Length), checkpoint.Value);
        Assert.True(store.LastStream!.Disposed); // the store stream is a file handle in production
    }

    [Fact]
    public void Startup_Initialize_without_a_checkpoint_fails_with_an_actionable_Error()
    {
        var store = new FakeModelStore();
        var checkpoint = new StartupCheckpoint<Payload>(store, "env", "algo", ReadPayload, new SilentLogger(), "test model");

        checkpoint.Initialize();

        // The readiness lie this whole file exists for: no checkpoint must never read back as Ready.
        Assert.Equal(ModelStatus.Failed, checkpoint.Status);
        Assert.NotNull(checkpoint.Error);
        Assert.Contains("test model", checkpoint.Error);
    }

    [Fact]
    public void Startup_a_load_that_throws_does_NOT_report_Ready()
    {
        var store = new FakeModelStore();
        store.Put(SomeCheckpoint);
        var checkpoint = new StartupCheckpoint<Payload>(
            store, "env", "algo", _ => throw new InvalidDataException("corrupt"), new SilentLogger(), "test model");

        // A checkpoint that EXISTS but cannot be read must land in Status/Error, not escape. Unguarded it
        // faulted the startup BackgroundService (host down at boot on a truncated Git-LFS pointer), and if the
        // host survived, Status stayed Loading so the lazy getter re-read and re-threw on EVERY request.
        checkpoint.Initialize();

        Assert.Equal(ModelStatus.Failed, checkpoint.Status);
        Assert.Null(checkpoint.Value);
        Assert.NotNull(checkpoint.Error);
        // The message must name the real fault; "no checkpoint in the store" would send the reader to the
        // wrong problem entirely, since the file is right there.
        Assert.Contains("corrupt", checkpoint.Error);
    }

    [Fact]
    public void Startup_Value_is_null_once_the_checkpoint_is_known_missing()
    {
        var store = new FakeModelStore();
        var checkpoint = new StartupCheckpoint<Payload>(store, "env", "algo", ReadPayload, new SilentLogger(), "test model");

        checkpoint.Initialize();
        int opensAfterInitialize = store.OpenCount;

        Assert.Null(checkpoint.Value);
        // Failed is terminal: the getter must not re-hit the store on every request once the verdict is in.
        Assert.Equal(opensAfterInitialize, store.OpenCount);
    }

    [Fact]
    public void Startup_Value_loads_lazily_on_first_access()
    {
        var store = new FakeModelStore();
        store.Put(SomeCheckpoint);
        var checkpoint = new StartupCheckpoint<Payload>(store, "env", "algo", ReadPayload, new SilentLogger(), "test model");

        Assert.NotNull(checkpoint.Value);
        Assert.Equal(ModelStatus.Ready, checkpoint.Status);

        var again = checkpoint.Value;
        Assert.Equal(1, store.OpenCount); // loaded once, then served read-only
        Assert.Same(checkpoint.Value, again);
    }

    // ── RefreshingCheckpoint ──

    [Fact]
    public void Refreshing_Current_is_null_while_the_store_has_nothing()
    {
        var store = new FakeModelStore();
        var checkpoint = new RefreshingCheckpoint<Payload>(
            store, "env", "algo", ReadPayload, new SilentLogger(), "test net", refresh: TimeSpan.Zero);

        Assert.Null(checkpoint.Current);
    }

    [Fact]
    public void Refreshing_picks_up_a_checkpoint_that_appears_after_construction()
    {
        // A zero cadence means "re-read on every access" — the timestamp-free way to drive the reload path.
        var store = new FakeModelStore();
        var checkpoint = new RefreshingCheckpoint<Payload>(
            store, "env", "algo", ReadPayload, new SilentLogger(), "test net", refresh: TimeSpan.Zero);

        Assert.Null(checkpoint.Current);

        store.Put(SomeCheckpoint);

        Assert.Equal(new Payload(SomeCheckpoint.Length), checkpoint.Current);
    }

    [Fact]
    public void Refreshing_serves_the_cached_net_inside_the_cadence()
    {
        var store = new FakeModelStore();
        store.Put(SomeCheckpoint);
        // The default cadence is minutes, so nothing inside a test run can reach the second read.
        var checkpoint = new RefreshingCheckpoint<Payload>(
            store, "env", "algo", ReadPayload, new SilentLogger(), "test net");

        var first = checkpoint.Current;
        for (int i = 0; i < 5; i++) Assert.Same(first, checkpoint.Current);

        Assert.Equal(1, store.OpenCount);
    }

    [Fact]
    public void Refreshing_reloads_the_new_bytes_when_the_cadence_has_elapsed()
    {
        var store = new FakeModelStore();
        store.Put(SomeCheckpoint);
        var checkpoint = new RefreshingCheckpoint<Payload>(
            store, "env", "algo", ReadPayload, new SilentLogger(), "test net", refresh: TimeSpan.Zero);

        Assert.Equal(new Payload(4), checkpoint.Current);

        store.Put([1, 2, 3, 4, 5, 6, 7, 8]); // a longer checkpoint, as a Lab campaign would rewrite it

        Assert.Equal(new Payload(8), checkpoint.Current);
        Assert.True(store.OpenCount >= 2);
    }

    [Fact]
    public void Refreshing_swallows_a_corrupt_read_and_keeps_the_previous_net()
    {
        // The contract that keeps the cube page alive while a campaign is mid-write: a throwing load must not
        // reach the caller, and must not drop the net that was already good.
        var store = new FakeModelStore();
        store.Put(SomeCheckpoint);
        bool fail = false;
        var checkpoint = new RefreshingCheckpoint<Payload>(
            store, "env", "algo",
            stream => fail ? throw new InvalidDataException("mid-write") : ReadPayload(stream),
            new SilentLogger(), "test net", refresh: TimeSpan.Zero);

        var good = checkpoint.Current;
        Assert.NotNull(good);

        fail = true;
        var afterCorruption = checkpoint.Current; // must not throw

        Assert.Same(good, afterCorruption);
    }

    [Fact]
    public void Refreshing_never_throws_even_when_it_has_no_previous_net_to_keep()
    {
        var store = new FakeModelStore();
        store.Put(SomeCheckpoint);
        var checkpoint = new RefreshingCheckpoint<Payload>(
            store, "env", "algo", _ => throw new InvalidDataException("corrupt from the start"),
            new SilentLogger(), "test net", refresh: TimeSpan.Zero);

        Assert.Null(checkpoint.Current);
    }

    [Fact]
    public void Refreshing_runs_the_onReload_hook_once_per_successful_load()
    {
        // Cube mirrors the fresh weights onto a resident GPU forward here, so a missed hook means the GPU keeps
        // serving stale weights while the CPU getter reports the new ones.
        var store = new FakeModelStore();
        store.Put(SomeCheckpoint);
        var reloaded = new List<Payload>();
        var gate = new object();
        var checkpoint = new RefreshingCheckpoint<Payload>(
            store, "env", "algo", ReadPayload, new SilentLogger(), "test net",
            refresh: TimeSpan.Zero, onReload: reloaded.Add, gate: gate);

        _ = checkpoint.Current;
        store.Put([9, 9, 9]);
        _ = checkpoint.Current;

        Assert.Equal(2, reloaded.Count);
        Assert.Equal(new Payload(4), reloaded[0]);
        Assert.Equal(new Payload(3), reloaded[1]);
    }

    [Fact]
    public void Refreshing_does_not_run_onReload_when_the_store_is_empty()
    {
        var store = new FakeModelStore();
        int hooks = 0;
        var checkpoint = new RefreshingCheckpoint<Payload>(
            store, "env", "algo", ReadPayload, new SilentLogger(), "test net",
            refresh: TimeSpan.Zero, onReload: _ => hooks++);

        Assert.Null(checkpoint.Current);
        Assert.Equal(0, hooks);
    }

    // ── ModelStartupHostedService ──

    [Fact]
    public async Task Startup_hosted_service_initializes_every_registered_model()
    {
        var a = new RecordingStartupService();
        var b = new RecordingStartupService();
        var hosted = new ModelStartupHostedService([a, b]);

        await hosted.StartAsync(CancellationToken.None);
        try
        {
            Assert.True(a.Done.Wait(TimeSpan.FromSeconds(5)));
            Assert.True(b.Done.Wait(TimeSpan.FromSeconds(5)));
            Assert.Equal(1, a.Calls);
            Assert.Equal(1, b.Calls);
        }
        finally
        {
            await hosted.StopAsync(CancellationToken.None);
        }
    }

    [Fact]
    public async Task Startup_hosted_service_with_no_models_completes_cleanly()
    {
        var hosted = new ModelStartupHostedService([]);

        await hosted.StartAsync(CancellationToken.None);
        await hosted.StopAsync(CancellationToken.None);
    }
}

/// <summary>
/// <c>CubeModelService.Rollout</c> — the greedy no-undo rollout behind the cube page's DQN tier.
/// </summary>
/// <remarks>
/// Only the <c>static</c> method is exercised: constructing <c>CubeModelService</c> would require a concrete
/// <c>AdaptiveBackend</c>, and building the ILGPU backend under coverage instrumentation is a known hazard
/// (coverlet's hit-recording injection breaks kernel JIT). The rollout needs no backend, so it is reachable
/// with a plain CPU MLP.
/// <para>What this guards: the rollout must stay honest. It reports whether the cube ended solved, and it must
/// respect the move ceiling and the no-undo mask — a rollout that quietly undid its own last move would loop
/// and report "unsolved" on boards the agent can actually handle.</para>
/// </remarks>
public class CubeRolloutTests
{
    private static GreedyQAgent UntrainedAgent(ulong seed) => new(
        new Mlp([RubiksCubeEnv.ObservationSize, 32, RubiksCubeEnv.ActionCount],
                new Xoshiro256StarStar(seed), Activation.Relu),
        RubiksCubeEnv.ActionCount);

    [Fact]
    public void Rollout_on_a_solved_cube_makes_no_moves()
    {
        var (solved, moves) = CubeModelService.Rollout(UntrainedAgent(1), new FaceletCube());

        Assert.True(solved);
        Assert.Empty(moves);
    }

    [Fact]
    public void Rollout_leaves_the_caller_cube_untouched()
    {
        var start = new FaceletCube();
        start.Apply("R");
        start.Apply("U");

        CubeModelService.Rollout(UntrainedAgent(2), start);

        // The rollout clones; if it mutated the caller's cube the page would replay from a drifted state.
        var reference = new FaceletCube();
        reference.Apply("R");
        reference.Apply("U");
        Assert.Equal(reference.ToColorFaces()[0], start.ToColorFaces()[0]);
    }

    [Fact]
    public void Rollout_respects_the_move_ceiling_and_the_no_undo_mask()
    {
        var start = new FaceletCube();
        start.Apply(FaceletCube.ScrambleMoves(new Xoshiro256StarStar(7), 15));

        var (solved, moves) = CubeModelService.Rollout(UntrainedAgent(3), start);

        Assert.InRange(moves.Count, 0, CubeModelService.MaxMoves);
        // An untrained net will not solve a 15-move scramble; whatever it does, it must be reported honestly.
        if (!solved) Assert.Equal(CubeModelService.MaxMoves, moves.Count);

        // Every emitted move is a real quarter turn, and no move immediately undoes the previous one.
        Assert.All(moves, m => Assert.Contains(m, FaceletCube.QuarterTurnMoves));
        for (int i = 1; i < moves.Count; i++)
        {
            int previous = Array.IndexOf(FaceletCube.QuarterTurnMoves, moves[i - 1]);
            int current = Array.IndexOf(FaceletCube.QuarterTurnMoves, moves[i]);
            Assert.True(RubiksCubeEnv.ActionMask(previous)[current],
                        $"move {i} ({moves[i]}) is masked after {moves[i - 1]}");
        }
    }

    [Fact]
    public void Rollout_replaying_its_moves_reproduces_its_verdict()
    {
        var start = new FaceletCube();
        start.Apply(FaceletCube.ScrambleMoves(new Xoshiro256StarStar(21), 3));

        var (solved, moves) = CubeModelService.Rollout(UntrainedAgent(4), start);

        // The moves are what the browser replays, so they must produce the state the server claims.
        var replay = start.Clone();
        foreach (var move in moves) replay.Apply(move);
        Assert.Equal(solved, replay.IsSolved);
    }
}

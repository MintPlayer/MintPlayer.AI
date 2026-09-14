using MintPlayer.AI.ReinforcementLearning.Core.Checkpoints;
using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;
using MintPlayer.SourceGenerators.Attributes;

namespace RLDemo.Web.Services;

/// <summary>
/// Owns the Block Dude policy net and plays levels with it.
/// </summary>
/// <remarks>
/// <para><b>Server-side inference, unlike FruitCake's in-browser net.</b> The reason is the checkpoint: this net
/// is 16 MB, which is a poor thing to push to every visitor of a page whose game itself runs fine without it.
/// One request per "watch the AI" click is cheaper for the user and keeps the net where it is already known to
/// work.</para>
///
/// <para><b>Greedy first, beam only as a fallback.</b> The shipped net solves all 15 levels playing greedily —
/// one forward pass per move, no lookahead — so the normal path costs a few hundred forwards and returns in
/// well under a second. Beam search exists here for anything the greedy policy cannot do, which on present
/// evidence means levels other than the fifteen it was trained on (it scores 9% on unseen boards). A visitor who
/// adds a level should get *an* answer rather than a shrug.</para>
/// </remarks>
[Register(ServiceLifetime.Singleton, "RLDemoWebModelServices")]
public sealed class BlockDudeModelService(IModelStore store, ILogger<BlockDudeModelService> logger)
{
    public const string EnvironmentId = "blockdude";
    public const string PolicyAlgorithmId = "policy";

    /// <summary>Greedy step ceiling. The longest shipped level the net plays is 826 moves, so this is generous
    /// without letting a looping policy on some unknown board run indefinitely.</summary>
    private const int StepBudget = 3_000;

    /// <summary>Beam width for the fallback. Measured on the shipped levels at 256; wider costs time per request
    /// for a path that is only taken when the policy has already failed.</summary>
    private const int BeamWidth = 256;

    private readonly RefreshingCheckpoint<BlockDudePolicyNet> _policy = new(
        store, EnvironmentId, PolicyAlgorithmId, BlockDudePolicyNet.Load, logger, "Block Dude policy net");

    /// <summary>How a solution was produced — surfaced to the client so the UI can be honest about it.</summary>
    public enum Tier { None, Greedy, Beam }

    /// <param name="Moves">Action indices, replayable through the engine twin in the browser.</param>
    /// <param name="Tier">Which tier produced it. "Greedy" means the net played it unaided.</param>
    public readonly record struct Solution(bool Solved, IReadOnlyList<int> Moves, Tier Tier);

    public bool IsReady => _policy.Current is not null;

    /// <summary>
    /// Plays <paramref name="grid"/> and returns the moves, or an unsolved result if neither tier finished.
    /// </summary>
    /// <param name="maxSeconds">Wall-clock ceiling for the beam fallback, so one pathological board cannot tie
    /// up a request thread.</param>
    public Solution Solve(string[] grid, int maxSeconds = 15)
    {
        var net = _policy.Current;
        if (net is null) return new(false, [], Tier.None);

        var board = BlockDudeBoard.FromGrid(grid);

        var greedy = BlockDudeGreedy.Run(net, board, StepBudget, recordMoves: true);
        if (greedy.Solved && greedy.Moves is not null) return new(true, greedy.Moves, Tier.Greedy);

        var beam = BlockDudeSearch.SolveByBeam(net, board, BeamWidth, maxDepth: StepBudget,
                                               TimeSpan.FromSeconds(maxSeconds));
        return beam.Solved ? new(true, beam.Moves!, Tier.Beam) : new(false, [], Tier.None);
    }
}

using MintPlayer.AI.ReinforcementLearning.Core.Random;

namespace MintPlayer.AI.ReinforcementLearning.Environments.Game2048;

/// <summary>
/// Test-time expectimax search over the n-tuple afterstate value function. The shipped
/// <see cref="NTuple2048Agent"/> picks moves 1-ply greedily (argmax [reward + V(afterstate)]),
/// which is blind to WHERE the random tile spawns — so it will, for instance, slide the big
/// corner tile away and let a stray 2 land in the corner it just vacated. Expectimax fixes
/// that by averaging V over the spawn distribution (a 2 at p=0.9 / a 4 at p=0.1, uniform over
/// empty cells), preferring moves that stay safe in expectation. It is pure serving-time
/// strength: the trained value tables are reused unchanged, nothing is learned. On the shipped
/// net, the default 1-ply lookahead lifts the average score ~1.9× over greedy (≈44k → ≈84k) and
/// reaches the 8192 tile.
///
/// Consistency with training: the tables were learned undiscounted (TD(0), target
/// reward + V(after)) with terminal afterstate value driven to 0 — so the search sums real
/// merge rewards along each path and bottoms out in V, with no discount and terminal = 0.
///
/// Cost: a strong agent plays very long games (thousands of moves), and per-move search cost
/// grows ~super-linearly in <see cref="MaxDepth"/>, so a single full playout runs ~1 s at depth 1
/// but tens of seconds at depth 2 — for little reliable gain, since an imperfect leaf value makes
/// deeper search no better here. Depth 1 is therefore the default. At higher depths, spawn
/// branches are explored only while their cumulative probability stays at or above
/// <see cref="MinBranchProbability"/> (improbable lines cut to a 1-ply leaf), and equal boards
/// reached by different spawn orders are memoized, to keep the cost in check.
///
/// Objective note: 2048 is treated here as a HIGH-SCORE game (chase the largest tile / total
/// score), not a reach-2048-and-stop game. Expectimax maximizes expected eventual score, which
/// is exactly that goal — it keeps optimizing past 2048 toward 4096/8192/16384/…
///
/// Counterpart of the 1-ply selector in <see cref="NTuple2048Agent.ChooseMove"/>.
/// </summary>
public sealed class Expectimax2048(NTuple2048Agent agent)
{
    /// <summary>
    /// Spawn (chance) lookahead layers. <c>0</c> reproduces <see cref="NTuple2048Agent.ChooseMove"/>
    /// exactly; <c>1</c> (the default) averages over the immediate spawn — the responsive, clearly
    /// stronger serving choice. Higher values are far slower per playout (see the type remarks) with
    /// no reliable strength gain, so raise this only for offline analysis.
    /// </summary>
    public int MaxDepth { get; set; } = 1;

    /// <summary>
    /// Spawn branches whose cumulative path probability drops below this are not expanded — they
    /// are scored with a cheap 1-ply leaf instead. Lower = stronger but slower. 0 disables pruning.
    /// </summary>
    public double MinBranchProbability { get; set; } = 1e-3;

    /// <summary>
    /// Best move by expectimax; mirrors <see cref="NTuple2048Agent.ChooseMove"/> — returns the
    /// action, the immediate merge reward it earns, and its afterstate. −1 when no move is legal.
    /// </summary>
    public int ChooseMove(ReadOnlySpan<byte> board, out int reward, Span<byte> bestAfterstate)
    {
        // (packed afterstate, depth) → expected value: a transposition table scoped to this one
        // decision. Different spawn orders reach the same board, so this collapses enormous overlap.
        var memo = new Dictionary<(ulong Board, int Depth), double>();

        Span<byte> after = stackalloc byte[16];
        int bestAction = -1;
        double bestValue = double.NegativeInfinity;
        reward = 0;

        for (int action = 0; action < Board2048.ActionCount; action++)
        {
            board.CopyTo(after);
            if (!Board2048.ApplyMove(after, action, out _, out int mergedValues))
                continue;

            double value = mergedValues + ExpectedAfterSpawn(after, MaxDepth, 1.0, memo);
            if (value > bestValue)
            {
                bestValue = value;
                bestAction = action;
                reward = mergedValues;
                after.CopyTo(bestAfterstate);
            }
        }
        return bestAction;
    }

    /// <summary>Plays one game from a fresh start using expectimax selection; no learning.</summary>
    public (int Score, int MaxExponent) PlayGame(Xoshiro256StarStar rng)
    {
        Span<byte> board = stackalloc byte[16];
        Span<byte> after = stackalloc byte[16];
        board.Clear();
        Board2048.Spawn(board, rng);
        Board2048.Spawn(board, rng);

        int score = 0;
        while (Board2048.AnyMoveAvailable(board))
        {
            int action = ChooseMove(board, out int reward, after);
            if (action < 0) break;
            after.CopyTo(board);
            Board2048.Spawn(board, rng);
            score += reward;
        }
        return (score, Board2048.MaxExponent(board));
    }

    /// <summary>
    /// Chance node: expected value of an afterstate over the random spawn, expanding up to
    /// <paramref name="depth"/> further spawn layers. <paramref name="prob"/> is the cumulative
    /// probability of reaching here, used to prune unlikely branches. Memoized on
    /// (board, depth): the value is keyed without <paramref name="prob"/>, which only governs
    /// pruning deeper down — reusing a hit computed under a different prob is the standard
    /// transposition-table approximation and is immaterial to move ranking at the root.
    /// </summary>
    private double ExpectedAfterSpawn(ReadOnlySpan<byte> afterstate, int depth, double prob, Dictionary<(ulong, int), double> memo)
    {
        if (depth <= 0)
            return agent.Evaluate(afterstate);

        var key = (Pack(afterstate), depth);
        if (memo.TryGetValue(key, out double cached))
            return cached;

        int empty = 0;
        for (int i = 0; i < 16; i++)
            if (afterstate[i] == 0) empty++;

        double result;
        // A changing move that filled the last cell leaves no spawn; score the reply directly.
        if (empty == 0)
        {
            result = BestReply(afterstate, depth, prob, memo);
        }
        else
        {
            double expected = 0;
            double cellWeight = 1.0 / empty;
            Span<byte> child = stackalloc byte[16];
            afterstate.CopyTo(child);
            for (int i = 0; i < 16; i++)
            {
                if (afterstate[i] != 0) continue;
                expected += cellWeight * SpawnOutcome(child, i, tile: 1, p: 0.9, depth, prob * cellWeight * 0.9, memo);
                expected += cellWeight * SpawnOutcome(child, i, tile: 2, p: 0.1, depth, prob * cellWeight * 0.1, memo);
                child[i] = 0; // restore for the next empty cell
            }
            result = expected;
        }

        memo[key] = result;
        return result;
    }

    /// <summary>One (cell, tile) spawn: the reply value, pruned to a 1-ply leaf when improbable.</summary>
    private double SpawnOutcome(Span<byte> child, int cell, byte tile, double p, int depth, double cumulativeProb, Dictionary<(ulong, int), double> memo)
    {
        child[cell] = tile;
        // Below the threshold the branch barely shifts the average — score it with depth 0 (greedy).
        int childDepth = cumulativeProb < MinBranchProbability ? 0 : depth - 1;
        return p * BestReply(child, childDepth, cumulativeProb, memo);
    }

    /// <summary>Max node: best move from this (spawn-resolved) board; 0 when it is terminal.</summary>
    private double BestReply(ReadOnlySpan<byte> board, int depth, double prob, Dictionary<(ulong, int), double> memo)
    {
        Span<byte> after = stackalloc byte[16];
        double best = double.NegativeInfinity;
        for (int action = 0; action < Board2048.ActionCount; action++)
        {
            board.CopyTo(after);
            if (!Board2048.ApplyMove(after, action, out _, out int mergedValues))
                continue;
            double value = mergedValues + ExpectedAfterSpawn(after, depth, prob, memo);
            if (value > best) best = value;
        }
        return double.IsNegativeInfinity(best) ? 0.0 : best;
    }

    /// <summary>Packs the 16 exponents (4 bits each) into a 64-bit transposition key.</summary>
    private static ulong Pack(ReadOnlySpan<byte> board)
    {
        ulong packed = 0;
        for (int i = 0; i < 16; i++)
            packed |= (ulong)(board[i] & 0xF) << (i * 4);
        return packed;
    }
}

using MintPlayer.AI.ReinforcementLearning.Environments.BlockDude;

namespace MintPlayer.AI.ReinforcementLearning.Campaigns;

/// <summary>
/// Runs a Block Dude policy greedily to a terminal state.
/// </summary>
/// <remarks>
/// <para>Greedy, never search. Block Dude is irreversible — a block set down in the wrong place can make a level
/// unsolvable with no way back — so a policy that only succeeds with search in front of it has not learned the
/// game. Both the training gate and the shipped-level benchmark measure the policy itself, and they share this
/// code so the two numbers can never drift apart.</para>
/// </remarks>
public static class BlockDudeGreedy
{
    /// <summary>Why a rollout stopped. Anything but <see cref="Won"/> is a loss.</summary>
    public enum Ending
    {
        /// <summary>Reached the door.</summary>
        Won,
        /// <summary>The policy chose a move the rules refused, leaving the board unchanged — it is stuck.</summary>
        Refused,
        /// <summary>The policy had no legal move to offer at all.</summary>
        NoMove,
        /// <summary>Revisited states past the budget: walking in circles.</summary>
        Loop,
        /// <summary>Still playing when the step budget ran out.</summary>
        Budget,
    }

    /// <param name="DistinctStates">How many distinct board states the rollout touched. Read it against
    /// <paramref name="Steps"/>: far fewer distinct states than steps means the policy is cycling, which a
    /// <see cref="Ending.Budget"/> ending alone does not tell you.</param>
    /// <param name="Moves">The actions actually played, when the caller asked for them. Optional because the
    /// training gate runs thousands of rollouts and only needs the verdict, while a benchmark wants the line
    /// itself — to replay it, to compare its length with the human's, or to watch it in the game.</param>
    public readonly record struct Outcome(Ending Ending, int Steps, int DistinctStates,
                                          IReadOnlyList<int>? Moves = null)
    {
        public bool Solved => Ending == Ending.Won;
    }

    /// <summary>
    /// Plays greedily but never re-enters a state it has already been in: among the legal actions, takes the
    /// highest-scoring one whose successor is unvisited. NOT search — no lookahead, no backtracking, one forward
    /// pass per step exactly as <see cref="Run"/> — just a tie-break that a deterministic argmax lacks.
    /// </summary>
    /// <remarks>
    /// Separates two failure modes the plain greedy number conflates. A deterministic policy is trapped by the
    /// FIRST state it revisits, so "loops after 4 steps" can mean either "has learned nothing" or "knows where to
    /// go but has no way to break a tie". Measuring both says which. Deliberately NOT what the curriculum gate
    /// uses — the gate's stance (§8.4) is that the policy should stand on its own.
    /// </remarks>
    public static Outcome RunAvoidingRevisits(BlockDudePolicyNet net, BlockDudeBoard start, int stepBudget)
    {
        var current = start;
        var seen = new HashSet<int> { start.StateHash };

        for (int step = 0; step < stepBudget; step++)
        {
            if (current.Won) return new(Ending.Won, step, seen.Count);

            var (logits, _) = net.Evaluate(current);

            // Best-scoring legal action whose successor is somewhere new.
            var best = (Action: (BlockDudeAction?)null, Score: float.NegativeInfinity, Next: current);
            for (int a = 0; a < BlockDudeBoard.ActionCount; a++)
            {
                var action = (BlockDudeAction)a;
                if (!current.IsLegal(action)) continue;
                if (logits[a] <= best.Score) continue;

                var next = current.Apply(action);
                if (next.SameStateAs(current) || seen.Contains(next.StateHash)) continue;

                best = (action, logits[a], next);
            }

            if (best.Action is null) return new(Ending.NoMove, step, seen.Count);

            seen.Add(best.Next.StateHash);
            current = best.Next;
        }

        return current.Won ? new(Ending.Won, stepBudget, seen.Count) : new(Ending.Budget, stepBudget, seen.Count);
    }

    /// <summary>Plays <paramref name="start"/> greedily under <paramref name="net"/> until it ends.</summary>
    /// <param name="stepBudget">Hard cap on moves. The training gate derives this from the rung's optimal length;
    /// the shipped levels carry no optimal count, so their benchmark passes an explicit ceiling.</param>
    /// <param name="recordMoves">Also return the line played. Off by default — the gate runs thousands of these
    /// and only needs the verdict.</param>
    public static Outcome Run(BlockDudePolicyNet net, BlockDudeBoard start, int stepBudget, bool recordMoves = false)
    {
        var current = start;
        var seen = new HashSet<int>();
        var moves = recordMoves ? new List<int>() : null;

        for (int step = 0; step < stepBudget; step++)
        {
            if (current.Won) return new(Ending.Won, step, seen.Count, moves);

            var action = net.Greedy(current);
            if (action is null) return new(Ending.NoMove, step, seen.Count, moves);

            var next = current.Apply(action.Value);
            if (next.SameStateAs(current)) return new(Ending.Refused, step, seen.Count, moves);
            moves?.Add((int)action.Value);

            // Exit on the FIRST revisit. The original guard also required `seen.Count > stepBudget`, which can
            // never hold when at most `stepBudget` states are ever added — so it was inert and a cycling policy
            // burned the whole budget. Ending here is outcome-identical, not a relaxation: the greedy policy is a
            // deterministic function of the board, so a repeated state means the trajectory is periodic from that
            // point on and can never reach the door. It only ends such a rollout sooner.
            //
            // `StateHash` is a 32-bit hash, so a collision could in principle end a live rollout early. At these
            // trajectory lengths that is well under a tenth of a percent, and the affected rollout was already
            // being counted as unsolved.
            if (!seen.Add(next.StateHash)) return new(Ending.Loop, step, seen.Count, moves);
            current = next;
        }

        return current.Won
            ? new(Ending.Won, stepBudget, seen.Count, moves)
            : new(Ending.Budget, stepBudget, seen.Count, moves);
    }
}

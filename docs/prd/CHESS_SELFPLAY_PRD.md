# Self-play training (chess) — PRD

**Status:** Planned · 2026-07-12 · branch TBD (off `master`)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M39 · **Depends on:** the Core NN + checkpoint layer (§2/§5 of [../ARCHITECTURE.md](../ARCHITECTURE.md)), the two-headed `PolicyValueNet` (M37), the training-campaign harness (§8), and action masking (§3). Additive — no change to existing training behaviour.

## 1. Problem

The SDK trains agents in three paradigms today — model-free RL (DQN/PPO), planning over a learned value net (DAVI), and **imitation from an oracle** (the cube's Kociemba teacher, Rush Hour's BFS teacher). All three need an external signal: a reward function, a forward model, or an *exact oracle*. For a game like **chess** there is no cheap exact oracle, and a reactive policy plateaus — this repo's own history is unambiguous that **search is the lever** (Snake/FruitCake/Cube all show a capped reactive net amplified massively by inference search).

The missing paradigm is **self-play**: two players (the same improving network) play each other, and the games themselves become the training signal — the network bootstraps from nothing to strong play with no human data and no oracle. That is how a from-scratch SDK trains chess. We want to add self-play as a **first-class, reusable capability** — chess is the headline, but the machinery (a two-player game abstraction + MCTS + a self-play campaign) should be shared so the *next* two-player game plugs in by writing only its rules.

**Constraint (load-bearing):** reuse the SDK; write as little new code as possible. A three-agent investigation (2026-07-12) established that **~70% of the outer loop already exists** and the genuinely-new code is small and well-isolated (see §5).

## 2. Goal & success criteria

- **Self-play works, verifiably (the priority).** Starting from a **randomly-initialized** net, self-play training produces a net whose strength **rises monotonically** against a fixed baseline — measured first on **Connect-4** (converges in minutes on CPU; a negamax oracle gives an exact yardstick), then on **chess** (win-rate vs a random-legal baseline climbs; later, Elo vs a frozen earlier checkpoint).
- **Reuse-first.** The self-play stack reuses `PolicyValueNet` (unchanged), the soft-CE+value training step (already used by the imitation campaigns), `ITrainingCampaign`/`CampaignRunner`, the model store + checkpoint format, the M38 Lab plumbing (`AdamState`/`TrainWindow`/`PolicyGrowth`), action masking, the RNG streams, `--viz`, and the paired-seed A/B harness (→ the Elo promotion gate). New code is confined to: one game seam, MCTS, the self-play data loop, and each game's rules.
- **One new deep seam, not a shallow one.** `IZeroSumGame<TState>` in `Core/Planning` is the single new abstraction; it must pass the M38 bar (minimal, honest, consumed by *both* MCTS and the self-play trainer; Connect-4/chess/checkers/Go plug in unchanged). It is a **sibling** to `IDeterministicModel`, not an extension of it (see §3).
- **Correctness gated.** Chess legal-move generation is verified by **perft** (leaf-node counts matched to published values, depths 5–6, from startpos + Kiwipete + standard positions) *before any training*. Nothing downstream is trusted until perft passes.
- **Determinism & resume.** Self-play RNG (Dirichlet noise, move sampling) uses its own `SeedSequence` stream; a campaign resumes bitwise-consistently through the model store, like every other campaign (a `CampaignContractTests` roundtrip proves it).

**Honest non-goals for v1 (strength).** Not superhuman, not engine-strength chess. The net is **MLP-only** (no conv/attention in the backend), so a flattened board loses spatial structure — this proves the pipeline and gives *steadily-improving, fully-legal* play, but caps positional strength. And a small chess MLP does **not** clear the GPU routing threshold that helps the cube, so self-play is CPU-bound. The realistic target is *"plays fully-legal, steadily-improving chess"* — Connect-4 is where the self-improvement curve is unmistakable; chess is the headline consumer of the same rails. Conv-backend support and true engine strength are explicitly out of scope (separate workstreams).

**Other non-goals (v1).** No DQN/PPO for self-play (wrong fit — a ~4600-wide masked Q-head over terminal-only sparse reward under self-play non-stationarity); no reuse of the DQN `ReplayBuffer` (it's `(s,a,r,s′,term,mask)`-shaped; self-play wants an `(obs,π,z)` window — a plain list, like the imitation campaigns); no bitboards unless a bench says movegen is the bottleneck; no web showcase page in phase 1.

## 3. Key decision — AlphaZero-style, on one new game seam, Connect-4 first

**Design it twice** (full analysis in the investigation):

- **Design A — plain self-play over `IEnvironment`.** Model the game side-to-move, reward = terminal outcome, opponent = the current/frozen net (a small `IOpponentPolicy` seam), train with PPO/REINFORCE. *Rejected as the endpoint:* no lookahead → tactically-blind play, the exact reactive plateau the repo documents. Useful only as a plumbing pipe-test.
- **Design B — AlphaZero-style (chosen).** MCTS-guided self-play: each move runs PUCT simulations whose leaves are scored by the two-headed net; the move is sampled from the **root visit-count distribution** π; training targets are `(π, z)` where `z` is the game outcome from that position's mover. Loss = `CE(π, policyLogits) + MSE(z, tanh(value))`.

**Why B, given "minimal new code":** the chess rules engine is the *same irreducible cost* in both designs. B's only marginal cost over A is `Mcts.cs` + visit-count targets instead of an opponent seam + PPO glue — and B is the design that makes "two players play each other and the model improves" actually *true* for a tactically deep game. It also showcases the SDK's signature composition (game model + net + search), which is the repo's whole thesis.

**Why a new seam, not `IEnvironment` or `IDeterministicModel`:** `IEnvironment` is single-agent (single scalar reward, episode loop) — it can host Design A but not MCTS. `IDeterministicModel<TState>` is *close* — its non-mutating `Apply` is exactly right — but its `IsGoal` (single bool) can't express **win/loss/draw relative to the side to move**, its `ActionCount` (valid in every state) lies for chess's per-state legal moves, and it carries no observation/policy-size. Bolting those on would leak (a shallow, dishonest interface). So add a minimal sibling:

```csharp
namespace MintPlayer.AI.ReinforcementLearning.Core.Planning;

/// <summary>Terminal result RELATIVE TO THE SIDE TO MOVE in the queried state.</summary>
public enum GameResult { Ongoing, Win, Loss, Draw }

/// <summary>
/// A perfect-information, two-player, zero-sum, deterministic game — the shared shape MCTS and a self-play
/// trainer consume. Distinct from IEnvironment (no reward/episode loop) and IDeterministicModel (side-to-move
/// + win/loss/draw + per-state legal moves, not a single goal). Apply returns a FRESH successor and leaves
/// 'state' untouched (search reuses a state across moves); after Apply the side to move has flipped.
/// </summary>
public interface IZeroSumGame<TState>
{
    int PolicySize { get; }          // fixed action-index space = the net's policy-head width
    int ObservationSize { get; }
    TState Root(ulong? seed = null);
    IReadOnlyList<int> LegalMoves(TState state);
    TState Apply(TState state, int move);              // non-mutating; flips side to move
    GameResult Result(TState state);                   // terminal result for the mover, or Ongoing
    void WriteObservation(TState state, Span<float> destination); // side-to-move-relative encoding
}
```

**Why Connect-4 first:** it de-risks the *novel* machinery (MCTS soundness, value-sign backup, self-play target generation, non-collapse) on a game with trivial, fast, easily-correct rules and a cheap **negamax** oracle for an exact test — *separately* from chess's large, silent-bug-prone rules surface. Chess then lands as the seam's **second consumer**, riding the same MCTS/campaign/net/checkpoint rails unchanged. This is the same "prove the mechanism cheaply, then scale" discipline the rest of the repo follows, and each piece stays independently verifiable.

## 4. Design — the pieces

| Piece | Where | New/Reused |
|---|---|---|
| `IZeroSumGame<TState>` + `GameResult` | `Core/Planning/IZeroSumGame.cs` | **new** (deep seam) |
| `Mcts` — PUCT search: select → expand-with-priors → sign-flipping value backup; Dirichlet root noise; temperature; returns the root visit-count π | `Core/Planning/Mcts.cs` | **new** (~300–450 LOC; the core novel algorithm — no tree search exists in the repo) |
| `PolicyValueTraining.TrainStep(net, adam, obs[], policyTargets[], valueTargets[], batch)` — soft-CE + value regression, grad-clip, Adam | `Core/Training/` (generalized from `CubePolicyTraining.TrainStep`) | **new-ish** (~60 LOC; cube-imitation can also call it) |
| `SelfPlayCampaign : ITrainingCampaign, INetworkTelemetrySource` — `TrainChunk` plays K self-play games (MCTS both sides via net leaf) → `(obs,π,z)` rolling window → train → `PolicyGrowth.Maybe`; `Evaluate` = arena win-rate vs frozen best (→ Elo); `Checkpoint` = save-best on win-rate + `AdamState.Save` | `tools/…Lab/` | **new** (mirrors `CubeImitationCampaign`; reuses `AdamState`/`TrainWindow`/`PolicyGrowth`/telemetry) |
| `PolicyValueNet` — the AlphaZero net, **unchanged**; value wrapped in `tanh` at the call site for WDL in [-1,1] | `Core/Nn/PolicyValueNet.cs` | **reused** |
| `Connect4Game : IZeroSumGame<…>` + a negamax oracle (test) | `Environments/Connect4/` | **new** (small) |
| `ChessBoard` + legal movegen + draw/mate detection; move encoding (8×8×73 = 4672); `ChessGame : IZeroSumGame` / `ChessEnv`; plane observation; `ChessPolicyNet` wrapper; perft tests | `Environments/Chess/` | **new** (the large, test-heavy chunk) |
| Elo eval (`…Ab`), `--game connect4|chess` Lab dispatch, checkpoints via the model store + Git LFS | reuse patterns | **reused** |

**MCTS composition** mirrors `ValueGuidedSearch` over `IDeterministicModel`:

```csharp
public static class Mcts
{
    public sealed record Config(int Simulations = 400, float Cpuct = 1.25f,
        float DirichletAlpha = 0.3f, float RootNoiseFrac = 0.25f);

    /// <summary>Leaf evaluator: priors over PolicySize + value in [-1,1] for the mover (PolicyValueNet under NoGrad).</summary>
    public delegate (float[] Priors, float Value) Evaluate<TState>(TState state);

    /// <summary>Search from 'state'; return the root visit-count distribution over PolicySize (the training target π).</summary>
    public static float[] Search<TState>(IZeroSumGame<TState> game, TState state,
        Evaluate<TState> evaluate, Config config, Xoshiro256StarStar rng);
}
```

## 5. Reuse-vs-new ledger (rough)

**Reused verbatim (0 new lines):** `PolicyValueNet` + Adam + tape autograd + `LogSoftmax`/`Mul`/`Sum`/`Tanh`; `ITrainingCampaign`/`CampaignRunner`/`AIHost`; `IModelStore`/`FileModelStore` + `CheckpointFormat` + `PolicyValueNet.Save/Load` (generic over a `kind` string, v2 trunk-widths); `IEnvironment`/`Space`/`IActionMaskProvider`/`IStatefulEnvironment`; `Net2Net` growth; `--viz`; the A/B harness pattern; `SeedSequence` streams.

**New:** the game seam + MCTS + the `PolicyValueTraining` step + the self-play campaign (~900–1100 LOC total, most of it MCTS + the campaign) — plus each game's rules: Connect-4 (small), **chess (~1,500–2,200 LOC, rules + perft dominate)**. Total ≈ 3,000–4,500 LOC, of which the chess rules engine is the bulk and MCTS is the subtle part.

## 6. Risks

1. **Chess-rule correctness (highest).** Castling (through/into check), en passant (incl. discovered check), promotion, pins, and draw rules (50-move, threefold via position hashing, insufficient material, stalemate vs checkmate) are silent-bug territory. **Mitigation, non-optional:** perft node-count tests to depth 5–6 from standard positions matched exactly to published counts, *before* any training. Isolating this in M39.2 (chess) — after the rails are proven on Connect-4 in M39.1 — means a movegen bug can't masquerade as an MCTS/self-play bug.
2. **MCTS soundness / self-play collapse.** Wrong value-sign backup, missing Dirichlet root noise, or no temperature schedule → collapsing targets. Mitigation: unit-test MCTS on forced-win/forced-draw Connect-4 positions against negamax; gate training on rising win-rate vs a *frozen* baseline.
3. **Compute.** Self-play games are long; MCTS is hundreds of sims/move. CPU-bound (the small MLP won't clear the GPU threshold). Mitigation: Connect-4 for the convergence proof; keep chess sims modest; batched-leaf MCTS is a phase-3 lever. Set expectations: legal, improving play — not engine strength.
4. **Representation ceiling.** MLP over flattened planes has no spatial prior. Accepted for v1; conv backend is a separate workstream if positional strength becomes a goal.

## 7. Verification

- **M39.1 gate (Connect-4):** perft-free (tiny rules, unit-tested directly); MCTS unit tests vs negamax on forced positions; from random init, self-play win-rate vs negamax/random **climbs**; `CampaignContractTests` resume roundtrip (fresh → `TrainChunk` advances → `Checkpoint` → new instance `Resume`s and continues); `dotnet test --filter "Category!=Slow"` green.
- **M39.2 gate (chess):** **perft matches published counts** (startpos, Kiwipete, …) to depth 5–6 — the hard gate; move-encode/decode round-trips over all legal moves; env terminated-vs-truncated split correct; win-rate vs random-legal climbs; contract-test resume; ship `models/chess.az.ckpt` (LFS).
- Every step ends on a green build + its gate, revert-friendly, one milestone at a time.

See [PLAN.md](PLAN.md) M39 for the phased step order.

# Reusable deterministic CPU-parallel data generation (self-play + cube) — PRD

**Status:** ✅ SHIPPED 2026-07-13 (branch `m39-chess-selfplay-plan`, PR #32) — all of M41.1/M41.2/M41.3.
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M41 · **Motivated by:** chess self-play is CPU-bound (movegen) and single-threaded, so a multi-core box sits ~idle while training crawls (see the M40.4 material-shaping run: 256 sims/move, one core).

## Implementation status (what actually shipped vs. this design)
- **M41.1** `Core/Training/DeterministicParallel.cs` (commit `bc0c48f`). Two overloads: the `SeedSequence`+stream form in
  §4a (with `int stream`, not `uint`), plus a **raw-`ulong baseSeed` overload** added in M41.3 (the SeedSequence form
  delegates to it). 12+1 unit tests: bitwise parallel==sequential across DOP 1/2/4/8/16, ordering, distinct streams,
  edges, and the cube-seed-equivalence lock.
- **M41.2** parallel self-play (commit `a9fa5c3`). Per §4b, minus: **eval/arena were left sequential** (not the
  bottleneck, and they can't affect trained weights), and no separate arena RNG work was needed. Gate MET — Connect-4
  **byte-identical checkpoint** at sequential vs dop-1 vs dop-8.
- **M41.3** cube dedup (commit `bed7577`) — **done, not skipped.** Better than this doc's "outputs may change, re-verify"
  assumption: the raw-seed overload reproduces the old `roundBase + φ·(worker+1)` seeding **byte-for-byte**, so cube
  output is unchanged (locked by a test). The shared `Interlocked` solve-counter was removed (per-generator counts summed
  on the owner thread).
- **Not done:** a formal games/hour speedup measurement on chess (the DOP-invariance gate + architecture make the speedup
  structural; left to observe in a real training run).

## 1. Problem

AlphaZero self-play (`SelfPlayCampaign`, used by chess/connect4) generates its training games on **one thread** — `TrainChunk` runs `for (g…) PlayGame()` serially (`SelfPlayCampaign.cs:119`). For chess, wall-time is dominated by **legal-move generation inside MCTS** (every simulation calls `IZeroSumGame.LegalMoves`/`Apply` → `ChessGame`/`ChessRules`), which is embarrassingly parallel per game — yet all but one CPU core sits idle. This is the practical bottleneck behind "training is too slow" (more so than the GPU angle: the net is tiny and inference is batch-1, a poor GPU fit).

Meanwhile the repo **already parallelizes data generation for the Rubik's cube** — but that code is **hand-rolled and duplicated in two game-specific Lab files**, not in the reusable library. The owner's question: *why isn't this in Core, and should we refactor it so self-play (and everything else) gets it for free?*

## 2. Findings (from a 2-agent read-only investigation, 2026-07-13)

### 2a. Where CPU parallelism lives today
- **Game-specific, in the Lab (duplicated):** cube training-data generation —
  `CubeImitationCampaign.cs:71` (Kociemba-oracle self-play) and `CubeEfficientCampaign.cs:93` (teacher-free scramble-reversal). Both hand-roll the **same** idiom: `Parallel.For(0, ProcessorCount-2, …)` + **per-worker result lists** (disjoint indices) + **per-worker seeded `Xoshiro256StarStar`** + an `Interlocked` counter. Copy-pasted between the two; **not in Core**.
- **Reusable, in Core (generic):** `ManagedBackend` GEMM/row-ops (`IComputeBackend.cs:426`, MAC-thresholded, disjoint row bands — the NN math under *every* campaign); `VectorEnv.Step` (`VectorEnv.cs:67`, per-env RNG, disjoint slots, `parallel:true`); `ValueIterationTrainer` batch featurization (`ValueIterationTrainer.cs:188/218`, used by `cube-davi`).
- **No explicit parallelism:** `SelfPlayCampaign` and the DQN campaigns (`SnakeDqnCampaign`, `FruitCakeDqnCampaign`, `DqnScoreCampaign`) — their only CPU parallelism is the implicit backend GEMM.

**So: there is no reusable "parallel data-generation / episode rollout" primitive in Core.** Each Lab campaign that wants it hand-rolls the pattern; self-play never did. There is **no deliberate reason** — it's organic growth, and the hand-rolled idiom is *exactly* the deterministic pattern Core already uses elsewhere (`VectorEnv`, GEMM). It should be extracted.

### 2b. Is self-play safe to parallelize? Yes — with a per-game-RNG + ordered-merge.
- **Shared read-only net inference is SAFE concurrently.** `PolicyValueNet.Forward` → `Tensor` ops allocate fresh output buffers, never mutate inputs, and there is no shared scratch/static cache (`TensorOps.cs:16/56`, `IComputeBackend.cs:189`). The one caveat is the autograd tape: weights are `RequiresGrad` (`Modules.cs:53`), so a forward *outside* `NoGrad` would build a tape over shared weights — but the evaluator already wraps every forward in `using (GradMode.NoGrad())` (`SelfPlayCampaign.cs:286`), and `GradMode` depth is **`[ThreadStatic]`** (`Tensor.cs:143`), entered per worker thread. Batch-1 forwards also don't trip `ManagedBackend`'s `rows>=2` gate, so they won't spawn nested `Parallel.For` fighting the outer per-game loop. → **one shared read-only net snapshot suffices; no per-thread net copies.**
- **The true blockers** are (i) the shared `_window` (`List.Add`/`RemoveAt`) + non-atomic `_totalGames`/`_totalSamples` counters, and (ii) the shared mutable RNGs (`_searchRng`/`_evalRng`/`_arenaRng`) — a single advancing Xoshiro shared across concurrent games both **races** and **breaks determinism** (a game's draws depend on interleaving).
- **Bottleneck** is chess `IZeroSumGame` movegen inside MCTS (heavy, per-simulation), NOT net inference — which is exactly why per-game thread parallelism pays off and a shared net is fine.

### 2c. The reproducibility invariant — must be preserved
The repo guarantees **bitwise determinism per seed** (PLAN M25/M26; **M36 SHA256-verified** viz-vs-no-viz). Core's existing parallel primitives keep it by construction: GEMM partitions **disjoint output rows, no reduction → byte-identical at any DOP** (`IComputeBackend.cs:115`); `VectorEnv` gives **each env its own RNG → parallel == sequential** (`VectorEnv.cs:9`). The refactor MUST match this: a game's randomness and output slot must be a pure function of its **global game index**, not execution order.

## 3. Decision

**Yes, refactor.** Extract a small **reusable, deterministic CPU-parallel sample-generator into Core**, and adopt it in `SelfPlayCampaign` (new capability: parallel self-play) and the two cube Lab campaigns (dedup). This gives self-play ~N-core scaling on its chess-movegen bottleneck **while keeping bitwise reproducibility**, and removes the copy-pasted cube idiom.

## 4. Design

### 4a. The Core primitive (`Core/Training/` or `Core/Concurrency/`)
A generic, determinism-preserving parallel generator — the common shape of both cube data-gen and self-play:

```csharp
public static class DeterministicParallel
{
    /// Runs `makeItem(index, rng)` for index in [0,count), each with its OWN RNG derived from
    /// (seeds, stream, baseIndex+index) — so a given index yields identical output regardless of
    /// worker count or completion order. Results are returned in ASCENDING index order. `parallel`
    /// toggles Parallel.For vs a sequential loop; the two are bitwise-identical (VectorEnv/GEMM rule).
    public static TItem[] Generate<TItem>(
        int count, SeedSequence seeds, uint stream, long baseIndex,
        Func<int, Xoshiro256StarStar, TItem> makeItem, bool parallel, int? maxDop = null);
}
```

- **Per-item RNG:** derive as `VectorEnv`/`SeedSequence` already do (`baseSeed + index*0x9E3779B97F4A7C15UL`, `VectorEnv.cs:54`; `SeedSequence.Derive`) so item *g* is reproducible independent of scheduling.
- **Ordered result slots:** write `results[index]` (preallocated), exactly like `VecStep`'s per-index arrays — the parallel region touches only disjoint slots (no locks), and the caller consumes them in index order.
- **Bitwise guarantee** identical to `ManagedBackend(maxDegreeOfParallelism)`: the DOP/`--parallel` knob provably does not change output.

### 4b. Self-play adoption (`SelfPlayCampaign`)
- Move the single-game logic (`PlaySelfPlay`/`PlayVsRandom`) into a **pure, index-addressable** form that takes its own `Xoshiro` (for the mode coin-flip, MCTS Dirichlet, and `SelectMove`) and a **shared read-only net snapshot** (`_net`, or a `Freeze(_net)` to decouple from an in-flight train step), and **returns** its samples instead of calling `AddSample`.
- `TrainChunk` calls `DeterministicParallel.Generate(gamesPerChunk, …, makeItem: g => PlayOneGame(g, …), parallel)` → then **merges the returned per-game sample lists into `_window` in ascending game index** and only then bumps `_totalGames`/`_totalSamples`. Training (`PolicyValueTraining.TrainStep`, `Backward`, optimizer) stays on the **owner thread** after the join.
- The eval/arena loops (`ArenaVsRandom`, `ArenaVsNet`) are structurally the same independent-game loops → same primitive, per-game eval/arena RNG.
- A `--parallel`/`--dop` flag (default = cores−2, matching cube; 1 = today's behaviour). **Weights are identical at any DOP** (assert via a SHA check like M36).

### 4c. Cube dedup (secondary)
`CubeImitationCampaign`/`CubeEfficientCampaign` re-express their `Parallel.For + per-worker-list + seeded-RNG` block as `DeterministicParallel.Generate(...)`, removing the duplication. (Their `makeItem` calls `CubeOracle.LabelScramblePath` / `CubeSelfSupervised.LabelScramblePath` — the primitive is general enough; it's just "produce K labeled samples per index".)

## 5. Phases

- **M41.1 — the Core primitive + tests.** `DeterministicParallel.Generate` + unit tests proving parallel output == sequential output (bitwise), for several DOPs, on a toy `makeItem`.
- **M41.2 — parallel self-play.** Refactor `SelfPlayCampaign` generation onto it (pure per-game fn + per-game RNG + ordered merge; shared read-only net); `--parallel`/`--dop`. **Gate:** SHA256-identical trained checkpoint at dop-1 vs dop-N for a fixed seed/short run (the M36-style determinism proof), self-play contract tests green, and a wall-clock speedup on chess (report games/hour at dop-1 vs dop-N).
- **M41.3 (optional) — cube dedup.** Move the two cube campaigns onto the primitive; verify their outputs unchanged (SHA) and DOP-invariant.

## 6. Risks
1. **Reproducibility regression** — the whole point; mitigated by the per-index-RNG + ordered-merge design and a SHA dop-1-vs-N gate (M41.2).
2. **Hidden shared state in inference** — investigation found none (fresh buffers, `[ThreadStatic]` NoGrad); guard by keeping `Backward`/optimizer strictly off worker threads.
3. **Nested parallelism** (outer per-game × inner GEMM) — batch-1 self-play forwards don't trip the GEMM `rows>=2` gate, so no oversubscription; if it ever does, pin the inner backend to DOP-1 during generation (the `FruitCakeAb` precedent, `FruitCakeAb.cs:36`).
4. **Scope creep** — DQN campaigns could also adopt it later; out of scope for M41 (they're not the bottleneck).

## 7. Verification
- M41.1: parallel==sequential bitwise unit test across DOPs.
- M41.2: **SHA256(checkpoint) identical dop-1 vs dop-8** for a fixed seed + short run; `SelfPlayCampaignTests` green; measured games/hour speedup on `--game chess`.
- M41.3: cube checkpoints unchanged + DOP-invariant.

See [PLAN.md](PLAN.md) M41.

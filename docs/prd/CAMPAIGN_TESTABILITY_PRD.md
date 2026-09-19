# PRD — Making the training campaigns unit-testable (M64)

*2026-09-18 · investigated by a four-agent sweep · successor concern to `COVERAGE_90_PRD.md` §12.6*

`Campaigns` is **2,454 coverable lines at ~34%** — the largest uncovered pool in the repo, and the
reason M63 could not reach 90% (`COVERAGE_90_PRD.md` §12.6). The prompt for this work was: *the
campaigns actually train models, so let us leverage dependency injection to make them testable.*

## 1. The premise is redirected: **DI is not the blocker**

Two agents concluded this independently, and the code agrees.

- **The one-step seam already exists.** `ITrainingCampaign.TrainChunk()` is one unit of training,
  callable directly with no loop. Three campaigns are *already* driven that way by existing tests
  (`SelfPlayCampaignTests`, `CampaignContractTests`, `BlockDudeDeterminismTests`).
- **The outer loop is already clock-seamed.** `CampaignRunner(TimeProvider?)` does zero console and
  zero file IO; everything leaves through `CampaignOptions.OnEval`. `CampaignRunnerTests` drives it
  with a fake clock and a fake campaign, asserting exact cadence call-counts.
- **M46 already injected the rest**: environments, `IZeroSumGame<TState>`, options records,
  `ILogger?`, `IModelStore`, `ILadderStore`, `IPolicyValueNetBuilder`, and the Ilgpu-free
  forward/train-step factory delegates.

**What actually blocks a fast campaign test is hard-coded *constants*, and two campaigns that
nothing ever instantiates.** A second DI pass over an already-DI'd surface would be *riskier* than
the first, because the parts still un-injected are the ones left alone precisely because they are
load-bearing for determinism (§3).

| Blocker | Where | Effect |
|---|---|---|
| `const int TrainChunkIterations = 1000` | `Cube/CubeDaviCampaign.cs:70` | one `TrainChunk()` = 1000 optimizer steps; a fast Cube test is impossible |
| `const int BatchSize` (256 / 128 / 256) | BlockDudeImitation:33, BlockDudeExpertIteration:41, RushHourImitation:24 | `TrainChunk` **returns early doing nothing** below the threshold, so a tiny batch silently trains nothing |
| `CubeDaviCampaign(AdaptiveBackend …)` | `Cube/CubeDaviCampaign.cs:68` | concrete type, not `IComputeBackend` — and constructing a real one **under coverage** is itself hazardous (ILGPU × coverlet, `COVERAGE_PRD.md`) |
| `_randomEval = RushHourGenerator.Generate(…30 puzzles…)` | `RushHour/RushHourImitationCampaign.cs:60` | 30 generator + BFS calls in a **field initializer**, before a test can call anything |
| `while (labeled is null \|\| labeled.Count < 50)` | `RushHour/…:119` | unbounded rejection loop, no attempt cap |

**Zero tests instantiate `CubeDaviCampaign` (577 lines) or `BlockDudeExpertIterationCampaign` (518).**
1,095 lines that nothing has ever constructed.

## 2. The cheapest large win is not new tests — it is shrinking four existing ones

**Almost every assertion in the Slow campaign tests is already MECHANICAL.** They are Slow because
of the *budgets they were configured with*, not because of what they check. Across
`CampaignContractTests` (4), `SelfPlayCampaignTests` (3) and `DraughtsSelfPlayTests` (2), there is
**not one learning assertion**. They check: resume-returns-false-when-empty, chunk arithmetic,
metric names, no-NaN, which store ids were written, and resume-continues.

The proof is already in the tree:

| | budget | `Category=Slow`? |
|---|---|---|
| `TetrisEnvTests.Campaign_Registers_TrainsCheckpointsAndResumes` | `ChunkSteps=60, TargetSteps=120, Hidden=[32,32]` | **no** |
| `CrazyFruitsEnvTests.Campaign_Registers_TrainsCheckpointsAndResumes` | same shape | **no** |
| `CampaignContractTests` (Snake) | `ChunkSteps=1500, TargetSteps=3000, Hidden=[128,128]` | **yes** |

Identical assertions, **25× the budget**. Shrinking `CampaignContractTests` to the budget the Tetris
and CrazyFruits tests already prove affordable moves `SnakeDqnCampaign`, `FruitCakeDqnCampaign` and
`RushHourImitationCampaign` into the measured bucket for no new test code and no production change.

### The `BatchSize` trap — read this before shrinking anything

`SelfPlayCampaign.TrainChunk` gates *all* training behind `if (_window.Count >= _batchSize)`
(`SelfPlay/SelfPlayCampaign.cs:226`). Shrink `GamesPerChunk` without shrinking `BatchSize` and the
chunk generates games, trains **nothing**, and still returns a plausible game count — a green test
asserting nothing about training or the optimizer.

**This may already be biting the existing Slow test:** `SelfPlay_Plays_Checkpoints_AndResumesTheNet`
uses `GamesPerChunk = 4` against the default `BatchSize = 128`, which is roughly 80–160 Connect-4
samples — borderline, possibly zero training today. **Every shrunk self-play test must assert
`policyLoss` is non-NaN**, which is the cheapest available proof that a batch actually ran.

## 3. PRECONDITION — the determinism gate must be fixed *before* any of this

M46 established *"training behavior stays bitwise identical"* as a **hard cross-cutting gate for
every milestone** (`DI_CAMPAIGNS_PRD.md:63-66`). Two findings, both verified:

**1. The gate does not run in CI.** Both workflows run `--filter "Category!=Slow"`, and every
checkpoint-SHA determinism test is `[Trait("Category", "Slow")]`
(`SelfPlayCampaignTests.cs:95`, `DraughtsSelfPlayTests.cs:57`). **A change that breaks
bitwise-identity produces a fully green PR today.** The gate is local-and-manual by construction,
and nothing says so where someone would look. There is no checked-in hashing script either — the
procedure is "run the Slow tests, and hash by hand for anything they miss, then paste the result
into the PRD."

**2. The gate is blind to uniform drift.** `RunAndHashCheckpoint` asserts only that the three arms
agree with *each other*:

```csharp
Assert.Equal(sequential, parallelDop1);
Assert.Equal(sequential, parallelDop8);
```

All three are computed by the post-change code, so a refactor that shifts the seed fan-out
**uniformly** passes all three. It catches parallelism bugs — which is what M41 built it for — but
not the failure mode a wiring change actually causes.

**The fix is cheap and entirely test-side:**

- **M64.0a** — extend the two `RunAndHashCheckpoint` helpers to compare against a **checked-in hash
  literal**, generated on the base commit. Extract the duplicated helper while doing it.
- **M64.0b** — add a checkpoint-hash test for the **DQN family**, which has none at all today (only
  a forward-pass fingerprint and resume tests) — and which is exactly the family carrying the
  generated-ctor and env-lifetime hazards below.
- **M64.0c** — run the determinism tests in CI as a **separate, uninstrumented job**. This avoids
  M63.4's failure mode entirely: that revert was caused by a *performance-comparison* assertion
  (`BatchedGreedySolve_IsFasterThanPerSuccessor`), and determinism tests are not performance
  assertions.

### The three real hazards (not the ones the prompt assumed)

Construction **order** is mostly safe: `SeedSequence.Derive(i)` is a pure function of
`(masterSeed, i)`, not a sequential draw, so reordering `CreateRng` calls changes nothing. The
genuine hazards are identity-and-wiring, and all three compile cleanly and pass CI:

1. **Environment instance lifetime.** Envs hold a persistent xoshiro and `Reset()` re-seeds *only
   when passed a seed*, so an env's RNG is one continuous stream across the whole run. Campaigns are
   registered `AddSingleton` with envs **captured in the closure** — that capture is load-bearing.
   Make an env transient, or hand a campaign a second instance, and every episode changes.
2. **`[Inject]` generated parameter order.** `SnakeDqnCampaign` declares
   `[Inject] SnakeEnv evalEnv` and inherits `[Inject] IEnvironment<float[],int> trainEnv`; the
   generator emits **subclass fields first**. Add, remove or reorder an `[Inject]` field and the
   ctor silently reorders — a positional call site still compiles when the types are compatible,
   train and eval envs swap, and training runs on the eval grid with no error anywhere. The
   registration file already carries a warning comment about this.
3. **Stream-index collision.** `SelfPlayCampaign.cs:153-155` takes `RngStreams.Buffer` for the
   shuffle RNG specifically so it cannot collide with `Policy`. Nothing detects a new consumer
   claiming an already-used index.

**Plus one subtlety nobody had written down:** the DQN trainer's periodic eval runs on the **same
env instance** as training and mutates its RNG, then calls unseeded `Reset()`. So `EvalEvery`,
`EvalEpisodes` and chunk size are *inside the training byte stream*. Changing chunk size changes
trained weights while looking like a pure cleanup — which directly constrains §2's shrinking work:
**shrinking a DQN contract test changes its trained bytes, and that is expected and fine, but it
must not be confused with a determinism regression.**

## 4. Milestones

- **M64.0 — the gate** (a/b/c above). **Precondition, not a step.**
- **M64.1 — shrink, no new code.** `CampaignContractTests` ×4 to the Tetris/CrazyFruits budget, drop
  `Category=Slow`. Shrink the self-play lifecycle test *and* its `BatchSize`, with a non-NaN
  `policyLoss` assertion. Keep **one** wide-config DOP-invariance run as a production canary.
- **M64.2 — serialization round-trips** (Tier A). `CampaignProgressState` and
  `BlockDudeTrainingState` (196 lines, **zero coverage**): every field round-trips, RNG-count
  mismatch is refused, truncated streams degrade to null, fingerprint/observation-size mismatches
  are refused. Pure binary positional IO — a field added to `Save` but not `TryLoad` reads every
  later field as garbage, invisibly, until a long run resumes wrong.
- **M64.3 — the DQN spine** (Tier B). Save-best (**a worse eval must not overwrite the deployable
  net** — nothing tests this today), warm-start without `dqn-state`, the exact-cap chunk arithmetic,
  and metric order.
- **M64.4 — the cheapest real campaign** (Tier C). `SelfPlayCampaign<Connect4State>` at
  `Hidden=8, Simulations=1, GamesPerChunk=1, BatchSize=16` — single-digit ms uninstrumented.
  Asserts the `az-progress` sidecar round-trip, which the existing Slow test never checks.
- **M64.5 — Campaigns helpers** (Tier D). `PolicyGrowth.Maybe`, `SupervisedTraining`,
  `PolicyValueNetBuilders` kind tags, `FileLadderStore` naming, `CubePolicyTraining.Shuffle`.
- **M64.6 — BlockDude lifecycle without `TrainChunk`** (Tier E). Resume/checkpoint/fingerprint on
  526 lines with zero lifecycle coverage, deliberately never calling `TrainChunk` so no oracle runs.
- **M64.7 — the two constants**, only if M64.1–6 have not already met the target:
  `CubeDaviSettings.ChunkIterations`, `BatchSize` onto the options records, `IComputeBackend` on
  `CubeDaviCampaign`, a level filter on expert iteration, lazy `_randomEval`. **Production changes —
  each one re-runs M64.0's gate.**

**Ordering is deliberate: everything through M64.6 is test-only.** No construction order changes, no
registration changes, so no exposure to §3's hazards. The production changes are last and are the
only ones that need the gate re-run.

## 5. Expected outcome

~670 physical lines newly covered directly, plus ~750–800 moved from Slow-only into the measured
bucket ≈ **620–660 coverable ≈ Campaigns 34% → high 50s**, which is roughly **+4 points repo-wide
(70.8% → ~75%)**.

The remainder is concentrated and identifiable: `CubeDaviCampaign` 577,
`BlockDudeExpertIterationCampaign` 518, and the training halves of the imitation campaigns.

## 6. Ceremony warnings — things that move the number without meaning anything

- **`CampaignRunner` tests buy ZERO Campaigns coverage.** It lives in **Core**. Driving it with a
  fake campaign never loads the Campaigns assembly. Say this out loud before someone chases the
  percentage with runner tests. (Worth adding the four uncovered runner branches anyway — but bank
  them as Core.)
- **More DI-resolution tests.** `CampaignRegistrationTests` already resolves every campaign; adding
  more touches constructors and nothing else.
- **Calling `INetworkTelemetrySource` members per campaign** to cover `catch { return null; }`.
- **Repeating "no metric is NaN" on every campaign.** One per family is plenty.
- **A one-chunk `CubeDaviCampaign` test** would be both slow and shallow. The right move there is to
  extract its pure decision logic (frontier advance ratio, auto-widen stall rule, depth cap) into a
  static — the way `BlockDudeCurriculum.Advance` already was — and unit-test that in microseconds.

## 7. What must stay Slow

1. **`Davi_LearnsToSolveShallowCubes_TeacherFree`** — ≥80% solve rate after 6000 DAVI iterations.
   **The only genuine learning assertion in the entire campaign test set.** A 200-iteration version
   asserts nothing; it would pass on an untrained net about as often as a trained one.
2. **`BatchedGreedySolve_IsFasterThanPerSuccessor`** — a timing comparison, and the measured proof
   that instrumentation does not slow two code paths equally (10.3s → 45s, *and it fails*). It has
   no correctness content that `BatchedGreedySolve_MatchesPerSuccessorSolve` does not already own.
   **Best: move it out of the test suite into a benchmark.**
3. **Any future "win rate > X" / "food > N" threshold.** None exist in the campaign tests today —
   strength gates live in the Lab, and that is correct. Written down here so it stays that way: a
   threshold on a learned metric belongs in a Lab gate. A shrunk version of such a test is not a
   faster test, it is a coin flip with a green tick.
4. **One wide-config DOP-invariance run**, as a production canary.
5. **Lab `--eval-only` smokes** for the cube family (GPU stand-up + Kociemba warmup).

---

## 8. Outcome — M64 as built *(2026-09-18)*

| | result |
|---|---|
| Repo coverage | **70.72% → 74.18%** |
| **Campaigns** | **34.3% → 54.0%** |
| Tests | 771 → **841**, all passing |
| Fast bucket | 213s → **147s** (under the §10a budget) |
| Determinism gate | did not exist → **4 tests, green, in CI** |

- **M64.0 — the gate** ✅. `DeterminismGateTests`, `Category=Determinism`, run as a separate
  **uninstrumented** CI step in both workflows. See §8.1 — the first CI run of this gate found a real
  defect in the gate's own design.
- **M64.1 — shrink** ✅. Six tests moved Slow → measured, assertions unchanged.
- **M64.2 — serialization** ✅. 14 tests over `CampaignProgressState` and `BlockDudeTrainingState`
  (196 lines, previously zero coverage).
- **M64.3 — DQN spine** ✅. 8 tests. **Save-best was entirely untested** and its failure mode is
  silent: a noisy bad eval overwrites the deployable net and the web app serves it.
- **M64.4 — cheapest real campaign** ✅. 4 tests on the `az-progress` sidecar round-trip, which the
  existing lifecycle test never asserted.
- **M64.5 — helpers** ✅. 12 tests; `FileLadderStore`'s filename contract had no direct coverage at
  all (the ladder tests use an in-memory double).
- **M64.6 — BlockDude lifecycle** ✅. 8 tests on 526 lines, deliberately never calling `TrainChunk`.
- **M64.7 — the constants** 🟡 **half done, deliberately.** The three `BatchSize` constants are now
  options properties (shipped values kept as defaults; determinism gate re-run and green). The
  `CubeDaviCampaign` `AdaptiveBackend → IComputeBackend` seam is **NOT** done: that campaign is
  GPU-mandatory and **ILGPU × coverlet is a known incompatibility here** — coverlet's injected
  `RecordHit` calls make ILGPU's runtime kernel compilation throw, which cost 37 CI failures once
  already. Constructing a real `AdaptiveBackend` under coverage is the hazard, so that seam needs its
  own milestone with the GPU question answered first, not a drive-by change.

### 8.1 The gate caught a defect in itself, first run

The checked-in **checkpoint-byte** hash for the DQN spine **failed on Linux CI while passing on
Windows**. Self-play agreed on both. The gate was right and the design was wrong: **trained bytes are
not bit-identical across platform / JIT / SDK**, and the repo never claimed they were — its
determinism tests had only ever run on one machine, so nobody had found out.

Fixed by pinning what is actually invariant: **the xoshiro stream state** after a fixed run. Integer
arithmetic, identical everywhere — *and a better gate for the purpose*, because a uniform shift in the
seed fan-out (a new consumer claiming an existing `RngStreams` index, a construction order that
re-seeds an env, an `[Inject]` field reorder swapping train and eval) moves those states exactly, on
every platform. That is the failure mode the literal exists to catch and the one dop-invariance
structurally cannot see, since all of its arms are computed by the post-change code.

Byte comparisons remain only where both sides are computed in the **same environment**
(dop-invariance, same-process reproducibility), which is where float determinism does hold.

**Recorded as a standing fact:** DQN training is reproducible *within* a platform but **not bitwise
across platforms**. Anyone comparing checkpoints from two machines should know that before concluding
something regressed. The cause is `TensorPrimitives.Tanh/Exp/Log` (transcendentals are not
IEEE-specified and dispatch on ISA), `TensorPrimitives.Dot` (vectorized reduction order),
`TensorPrimitives.MultiplyAdd` (FMA contraction) and `MathF.Pow` in Adam's bias correction. The GEMM's
`Parallel.For` is **not** implicated — it partitions disjoint output rows.

Making checkpoints portable is achievable but would mean owning the numerics: hand-written
`Tanh`/`Exp`/`Log` from IEEE-basic ops, fixed-order `Dot`, explicit `Math.FusedMultiplyAdd`, `Pow` by
repeated multiplication. **Not recommended for this repo**: `cube-davi` and `cube-policy` are
GPU-mandatory and two *different GPUs* also disagree (reduction order depends on SM count), so it
would buy portability for the eight CPU-only games and nothing for the cube family. If ever wanted,
the scoped version — a bit-portable backend used only by gates and checkpoint tooling, never by
training — is worth far more than the total one.

### 8.2 Coverage vs assertion value — an honest split

M64.3/.4/.6 added 20 tests and **+5 covered lines**. Those paths were already *executed* by the
contract tests; what the new tests add is **assertion value** — save-best, the progress round-trip and
the fingerprint guard now have regression guards where they had none. The *number* moved from M64.1
(shrink) and M64.2/.5 (genuinely new surface). Both kinds of test are worth having; conflating them
is how a coverage percentage stops meaning anything.

### 8.3 90% is out of reach, with arithmetic

Covering **100% of the remaining Campaigns lines** reaches **81.5%**. The gap to 90% is +2,434 lines
against 3,974 uncovered in total, so it additionally requires most of `tools/Lab` (1,344) and
`Environments` (1,048). M63's §10.4 decision — exclude `tools/**`, move the target, or treat 90% as a
multi-milestone arc — is now unavoidable rather than optional.

### 8.4 Process failures worth keeping

- The M64.1 tests were timed **without `--collect`**, then used to justify moving them into the
  instrumented bucket — the exact error `COVERAGE_90_PRD.md` §12.7 was written to prevent, repeated
  one milestone later.
- A **449s** wall clock was reported as a result when it was contention from concurrent background
  jobs; the clean figure was 292s. Re-measure before reporting.
- Both guesses about *where* that time went were also wrong (the shrunk tests measured 15.2s in
  isolation, the determinism gate 0.7s). The real cost was three pre-existing BlockDude gate-board
  tests at 223s/209s/199s. **Isolated measurement understates**: FruitCake is 15.2s alone but 34s
  inside the full run, because instrumented cost depends on cache contention.
- To meet the 180s budget the gate-board **sampling constants** were cut (8→3 boards per stage, 8→2
  on starvation, 7→3 stages on the dead-end measurement). Every assertion is unchanged and coverage
  is bit-identical, so this removed sampling rather than code paths — but it is a real reduction in
  gate strength, and each site records the old value and when to raise it back.

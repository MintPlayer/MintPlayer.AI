# Reduce per-game boilerplate — campaigns, web services, frontend — PRD

**Status:** B0–B5 complete (incl. B3 `CliArgs` + B5 P7) · 2026-07-12 · branch `m38-reduce-boilerplate-plan`, [PR #31](https://github.com/MintPlayer/MintPlayer.AI/pull/31) · **supersedes the stale [PR #27](https://github.com/MintPlayer/MintPlayer.AI/pull/27)** · verified live (web host + Playwright) — see PLAN M38. (Only B5 P8, a shallow director-loop wrapper, intentionally left.)
**Owner:** Pieterjan
**Milestone:** [PLAN.md](PLAN.md) M38 · **Depends on:** nothing new — a behaviour-preserving refactor across the Lab training harness (§8 of [../ARCHITECTURE.md](../ARCHITECTURE.md)), the web model-service layer (§7), and the Angular playground (§10).

## 1. Problem

Adding each game has left behind **near-identical copy-paste** in three layers. The duplication is not
harmful *today*, but every copy is a place a fix has to land N times, and several copies have **already
drifted** — one copy has a bug its siblings don't. This is exactly the change-amplification / unknown-unknowns
cost Ousterhout warns about.

PR #27 ("reduce per-game boilerplate") took a first cut at this on 2026-07-10, but it is now **stale and
conflicting**: M36 (network visualizer / `INetworkTelemetrySource`) and M37 (progressive net growth /
`DqnGrowth`) landed afterward and edited precisely the campaign method bodies and fields that PR #27 deletes
and relocates. A straight rebase would silently drop net-growth. This PRD **re-scopes** the effort: it keeps
the parts of PR #27 that still apply verbatim, re-cuts the campaign refactor to *absorb* the new M36/M37
duplication rather than fight it, and adds the further duplication a three-agent audit surfaced.

**What "boilerplate" means here** (audited 2026-07-12, three parallel agents; counts verified against `master`):

### Lab / training campaigns (`tools/…Lab/`)
- **DQN campaign twins.** `SnakeDqnCampaign.cs` (213 L) and `FruitCakeDqnCampaign.cs` (236 L) are the *same*
  score-maximizing Double+Dueling DQN campaign. `TrainChunk` is byte-identical; warm-start/resume, the
  save-best gate, the eval loop, and the telemetry block are all parallel. FruitCake's only real extras: a
  noisy `StateId` line and the plain→noisy / `GrowInput` warm-net adaptation.
- **`INetworkTelemetrySource` duplicated 6×** (M36). Every campaign — Snake, FruitCake, RushHourImitation,
  CubeImitation, CubeEfficient, CubeDavi — carries a ~30-line explicit-interface block
  (`SnapshotParameters`/`Sample`/`SampleIo`/`SampleActivations`). The two DQN copies are ~90% identical twins.
- **Net-growth wiring** (M37) repeated per campaign: `grow`/`growEvery` ctor params, a `_growRng` field, and a
  `DqnGrowth.Maybe(...)` / `PolicyGrowth.Maybe(...)` line in `TrainChunk`.
- **CLI flag parsing hand-rolled 6×.** Every `*Lab.cs` re-derives the `i+1 < Length` bounds guard and
  `args[++i]` walk for the shared flags (`--hours`/`--data`/`--seed`/`--lr`/`--eval-only`, plus
  `--grow`/`--grow-every` in five). **Live inconsistency:** culture handling is ad-hoc — `double`/`float`
  mostly pass `InvariantCulture`, `int`/`long`/`ulong` inconsistently do not. A latent locale-dependent bug.
- **Host-bootstrap + `runner.Run` tail duplicated in all 6 labs** (~12 L each): `AIHost.CreateBuilder` →
  resolve store/runner → CSV path → new campaign → `VizLauncher.TryStart` → `runner.Run` → `WaitForViewer`.
  `WaitForViewer` currently lives in `SnakeLab` and is called cross-type from every other lab (a smell).
- **Imitation/policy lifecycle plumbing 3×** (RushHour, CubeImitation, CubeEfficient): Adam load/save, net
  resume-or-init, window-mean fields+reset, and the growth one-liner are identical. (Their `Evaluate` and
  data-generation *are* genuinely different algorithms — those must stay bespoke.)
- **Trivia:** identical one-line `Log` in all 6 campaigns; growth-RNG seed constant differs by family.

### Web model-service layer (`src/RLDemo.Web/`)
- **The cadence-refresh checkpoint getter — 4 copies** (the most bug-prone): "re-read on a cadence,
  double-checked-locked, swallow a corrupt/mid-write read and keep the previous value, optional post-load
  hook." CubeModelService ×3 (policy/value/efficient) + RushHourModelService ×1. This is the copy PR #27
  correctly targeted; **still 4 verbatim copies on `master`** (`TryOpenRead` appears 7× across the 3 services).
- **Model-service startup/readiness quartet — 3 copies:** `Status`/`Error` snapshot + lazy agent getter +
  `TryLoadFromStore` + `Initialize` ("load, else `Status=Failed` + warn"). Cube/Game2048/RushHour.
- **`RushHourController.TryBuildPuzzle` duplicated verbatim in `RushHourDeckStore`** (~26 L; a comment already
  admits the mirror). **Two validators of one contract will drift** — the deck store could accept a board the
  solver rejects.
- **Controller `/status` + 503 gating — 3 copies;** and `Game2048Controller` needlessly re-declares
  `Status2048Response` where Cube+RushHour share `StatusResponse`.

### Angular playground (`src/RLDemo.Web/ClientApp/`)
- **Component `pollStatus()` self-rescheduling poller — 3 copies, with a real divergence bug:** `cube.ts` has
  a `try/catch`; `game-2048.ts` and `rush-hour.ts` **do not** → an uncaught rejection on a backend blip.
- **`*-api.ts` `status()` + 503→`loading` classification — 3 copies.**
- **Watch-mode wake-lock + `visibilitychange` scaffolding — 3 copies** (snake, mountaincar, fruit-cake);
  `onVisibilityChange()` is byte-identical.
- **Director watch-loop skeleton — 2 copies** (snake, mountaincar; fruit-cake's rAF loop differs — leave it).
- **Atomic temp-file write — 2 copies** (`GalleryStore`, `RushHourDeckStore`; trivial).

## 2. Goal & success criteria

**Reduce the duplication above into deep, well-named modules — with zero behaviour change — and fix the two
latent bugs the duplication has already caused.**

- **Behaviour-preserving (the hard gate).** Training stays **bitwise-identical**: a `--game snake` / `fruitcake`
  run (and the imitation/policy campaigns) produces a **SHA256-equal** checkpoint before and after. The web API
  contract tests and the campaign/checkpoint test suites stay green. Only log wording may change.
- **Deep, not shallow.** Every extraction must hide more than it exposes (Ousterhout): a `RefreshingCheckpoint<T>`
  exposes `.Current` and hides the lock + cadence + keep-previous-on-corrupt; a `CliArgs` reader hides the
  bounds-check + culture + `++i`. We explicitly **do not** create a shallow `ModelService<TAgent>` base or a
  shared *solve*-controller base — the solve endpoints are genuinely different algorithms/DTOs and stay bespoke
  (PR #27 got this call right).
- **Fix the drift bugs found in the audit** as part of the relevant step: the missing `try/catch` in two
  frontend pollers, and the duplicated Rush Hour puzzle validator that can drift out of sync with the solver.
- **Absorb M36/M37, don't regress them.** The re-cut `DqnScoreCampaign` base **owns** net-growth and
  implements `INetworkTelemetrySource` itself, so both DQN games lose their growth wiring and telemetry twin —
  *more* reduction than PR #27, with growth preserved.
- **Net line reduction** across the touched files, with the shared modules documented (interface comment stating
  what each promises and hides).

**Non-goals.** No new features, no API/DTO changes visible to clients, no change to checkpoint formats or the
wire protocol. No merging of the three imitation/policy *campaigns* (only their plumbing is extracted). No
touching the `.pg` single-source FruitCake path. No frontend visual change.

## 3. Design — the shared modules

Grouped by the concern each hides. Each is a **deep module**: small surface, meaningful implementation hidden.

### Lab
1. **`DqnScoreCampaign` (abstract base)** — the score-maximizing DQN spine. Owns: seeds/state/warm-net fields,
   the save-best gate, `Resume`/`TrainChunk`/`IsComplete`/`Evaluate`/`Checkpoint`/`Dispose`, **net-growth**
   (`grow`/`growEvery` + `_growRng` + `DqnGrowth.Maybe` inside its `TrainChunk`), **and
   `INetworkTelemetrySource`** (`NetKind`, `SnapshotParameters`, `Sample`, `SampleIo`, `SampleActivations`,
   reading `State?.Online ?? WarmNet`). Subclasses supply only: `Environment`, `TrainEnv`, `BaseOptions`,
   `EvaluateNet`, `ObservationSize`/`InputLabels`/`OutputLabels`, and (FruitCake) an `AdaptWarmNet` hook for
   plain→noisy + `GrowInput`. *This is the re-cut of PR #27 change 1, enlarged to fold in the M36 telemetry
   twin and the M37 growth wiring.*
2. **`CliArgs` value-reader + `CommonLabArgs.Parse`** — `CliArgs` hides the `i+1<Length` guard, `++i` walk, and
   `InvariantCulture` for every scalar type (fixes the locale inconsistency in one place). `CommonLabArgs`
   parses the shared 6 flags into a record; each lab layers its game-specific flags on top. **Not** a
   declarative parser (would leak on the 46-flag `cube-davi`).
3. **`LabHost.Run(dataDir, csvName, hours, evalOnly, useGpu, build: …)`** — owns DI bootstrap + optional
   `AddGpuBackend` + CSV path + `VizLauncher.TryStart` + `runner.Run` + viewer-lifetime (`WaitForViewer` moves
   *into* it, off `SnakeLab`).
4. **`SupervisedNetState` + `WindowMean`** — extracts *only* the plumbing shared by the 3 imitation/policy
   campaigns (net+Adam load-or-init/save, growth one-liner; rolling-mean helper). The campaigns keep their own
   `Evaluate`/data-gen. **Explicitly not a shared campaign base.**
5. **Trivia:** `LabLog.Line` (the one-liner), `GrowthRng(seed)` (family-seeded).

### Web (backend)
6. **`RefreshingCheckpoint<T>`** — `(store, envId, algoId, load, refresh, onReload?)` exposing just `.Current`.
   Hides TTL + double-checked lock + keep-previous-on-corrupt (defines the corrupt-read error out of
   existence). Cube's GPU resident-forward rebuild rides the `onReload` hook. Collapses the 4 getters to a
   field + one-line property. *(PR #27's `RefreshingModel<T>`, renamed for clarity.)*
7. **`StartupCheckpoint<T>`** — the load-at-startup readiness holder (`Status`/`Error` + single-place
   `Initialize`). Applied by **composition**, not a base class (a `ModelService<T>` base would be shallow/leaky).
   Covers the same concern as #6, so ship them as a pair.
8. **`RushHourPuzzleDto.TryBuild(...)`** on the Environments layer — one validator; controller + deck store
   both call it. Kills the drift bug.
9. **`ControllerBase.ModelStatusResult(...)` extension** + **delete `Status2048Response`** (use the shared
   `StatusResponse`). Small; solve endpoints stay bespoke.

### Web (frontend)
10. **`pollModelStatus(api, set, intervalMs)`** — one poller with the `try/catch` built in (fixes the 2048 /
    rush-hour divergence). `fetchStatus(path)` + `classifySolve(...)` for the `*-api.ts` `status()`/503 union
    (keep each game's typed `Solve*Result`).
11. **`WatchWakeLock`** helper/directive driven by the component `mode` signal — hides the sentinel +
    `visibilitychange` bookkeeping (not just a wrapper around acquire/release).
12. **`runDirectorLoop({tick, step, onFrame, onDone})`** for the 2 `setInterval` directors (kept minimal — only
    2 call sites). **`AtomicFile.Write`** for the 2 temp-write copies (trivial, backend).

### Checkpoints doc
13. **`Core/Checkpoints/README.md`** — the "when to use which checkpoint type" decision table from PR #27
    (pure docs, still accurate). Land verbatim.

## 4. Deliberately left alone

- **Per-game *solve* controllers + DTOs** (beam search / policy-A\* / expectimax) — different algorithms,
  different validation. A shared base would be a leaky, shallow abstraction. (Same call PR #27 made.)
- **The 3 imitation/policy *campaigns* as wholes** — only plumbing (§3.4) is extracted; `Evaluate`/data-gen
  stay bespoke.
- **`CubeDaviLab`'s 46-flag + appsettings block**, `CampaignRunner`, `DqnGrowth`/`PolicyGrowth`,
  `CubePolicyTraining`, `CubeViz`, `VizLauncher`, `VizServer` — already deep/good.
- **fruit-cake's rAF watch loop** — genuinely different from the `setInterval` directors.

## 5. Risks

- **Silent behaviour change is the whole risk.** Mitigation: the SHA256 bitwise-identical checkpoint gate per
  DQN game (the M36 methodology already exists), plus the existing campaign/checkpoint/web-API test suites, run
  after **each** step. Any per-step diff in a checkpoint = stop and reconcile.
- **The telemetry-in-base move must read the same fields the 6× copies read.** Verify `--viz` still renders each
  game live (dev-only manual check) after the base absorbs telemetry.
- **Order matters.** Land the low-risk, self-contained pieces (docs, bug-fix extracts, web checkpoint holders)
  before the campaign re-cut, so a regression is bisectable.

## 6. Verification

- `dotnet build` clean (0 warn / 0 err) after every step.
- `dotnet test --filter "Category!=Slow"` green (campaign contract/runner, checkpoint round-trip, web API/service
  suites).
- **Bitwise gate:** a short fixed-seed `--game snake` and `--game fruitcake` run yields a SHA256-equal
  `*.dqn.ckpt` + `*.dqn-state.ckpt` vs a pre-refactor baseline; likewise a short imitation/policy run after §3.4.
- **`--viz` smoke** (dev-only): each game still streams a live net after the telemetry move.
- Net **negative** line diff with the shared modules documented.

See [PLAN.md](PLAN.md) M38 for the phased, revert-friendly step order.

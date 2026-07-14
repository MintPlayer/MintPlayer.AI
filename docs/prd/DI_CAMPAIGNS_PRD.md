# Dependency-injectable campaigns & games — PRD

**Status:** DRAFT — investigated 2026-07-14 (3-agent analysis: architecture map, source-generator study, test-landscape audit); no code changed yet.
**Milestone:** [PLAN.md](PLAN.md) M46
**Depends on:** M25/M26 (AIHost + `CampaignRunner` DI seam), `MintPlayer.SourceGenerators` 10.20.0 (already dogfooded on `CampaignRunner`).

## 1. Problem

The owner's ask: (1) all projects use the `AIHost.CreateBuilder` / `LabHost.Run` convention, but the
`*Campaign` and `*Game` classes are still pure instance classes outside the container — is that by design,
or can they become DI services? (2) adopt the `[Inject]` / `[Register]` source generators from
MintPlayer.Dotnet.Tools to simplify registration; (3) once refactored into units, write proper unit tests.

**Verdict from the investigation: partly by design, mostly changeable.** The hosting/runtime layer
(`TimeProvider`, `IModelStore`, `CampaignRunner`, `AdaptiveBackend`) is already DI; randomness, time, the
checkpoint store, the compute backend and the net architecture are already behind injectable seams — that
part is deliberate and correct (it's what makes the SHA256 checkpoint-determinism tests pass). What is *not*
by design, just accreted:

1. **Campaigns are `internal` classes inside the Lab exe** (`tools/…Lab/`), hand-`new`ed in the
   `Func<IServiceProvider, ITrainingCampaign> build` lambda each `<Game>Lab` passes to `LabHost.Run`
   (`LabHost.cs:21`). Tests reach them via an `extern alias Lab` + `InternalsVisibleTo` hack
   (needed because the exe's generated `Program` collides with `RLDemo.Web`'s under
   `WebApplicationFactory`) — campaigns are not first-class library types.
2. **Campaign constructors take positional CLI primitives** (`ulong seed, int chunkSteps, long targetSteps,
   float learningRate, …` — `SnakeDqnCampaign` takes 14 of them), not options objects.
   `SelfPlayOptions` is the only options record; it's still built inline from `CliArgs`.
3. **The DQN campaigns hard-wire their environment**: `SnakeDqnCampaign.cs:18` does
   `private readonly SnakeEnv _env = new(trainGrid, stepPenalty, safeMask);` in a field initializer.
   (By contrast `SelfPlayCampaign` already takes `IZeroSumGame<TState> game` as a ctor param —
   the self-play family is fully injectable today.)
4. **The self-play difficulty ladder does raw `File.*`/`Directory.*` I/O** (`SelfPlayCampaign.cs:538-671`),
   bypassing the `IModelStore` seam the rest of the checkpoint path uses — ladder promotion is untestable
   without a real disk.
5. **Campaign logging is a static `Console.WriteLine`** — harmless, but unassertable in tests.

Games need almost nothing: `ChessGame` / `Connect4Game` are stateless, parameterless
`IZeroSumGame<TState>` implementations — registering them is trivial.

## 2. What is by design and must NOT change

These look like DI violations but are deliberate; the refactor must leave them alone:

- **`Backend.Current`** (`IComputeBackend.cs:110`) — a settable process-global compute backend all
  autograd `Tensor` ops route through. Deliberately global (single-device per op; multi-GPU goes through
  resident forward objects that bypass it). Not a scoped service.
- **`GradMode.NoGrad()`** — thread-local static toggle.
- **Static `WriteObservation`/`BuildObservation`** — the train==serve guarantee.
- **Polyglot-generated cores** (`PgChessState`, `PgSnakeEnv`, `PgConvNet`, … from the single-source `.pg`
  files, ~2,240 lines) — generated C#/TS, DI-free by construction and *pure*, so they need no dependencies.
  The hand-written facades (`ChessGame`, `SnakeEnv`, `FruitCakeEnv`) are the injectable surface; they keep
  `new`ing the generated core internally. Don't push DI below the facade.
- **GPU layering via factories** — `SelfPlayCampaign` takes `forwardFactory`/`trainStepFactory`/`backend`
  (defaulting to CPU autograd) precisely so the generic campaign never references an `Ilgpu` type
  (`ChessLab.cs:116-141` builds them from `AdaptiveBackend`). The DI design keeps GPU-resident pieces
  behind the Core interfaces (`IPolicyValueForward`, `IPolicyValueTrainStep`, `IComputeBackend`).
- **Seeded RNG discipline** — `SeedSequence` → `RngStreams`, `Xoshiro256StarStar`; `System.Random` is
  banned in the RL path. Campaigns take `ulong seed`; envs seed from `Reset(seed)`. Already perfect.
- **`TimeProvider` / `IModelStore` / options records** — already injected; the existing contract tests
  (`CampaignRunnerTests` with `FakeTime`, `SelfPlayCampaignTests` with `Connect4Game` + temp stores)
  are the template to generalize, not replace.

**Hard cross-cutting gate for every milestone:** training behavior stays *bitwise identical* — the
SHA256 checkpoint-determinism tests (`SelfPlayCampaignTests.RunAndHashCheckpoint`,
`DeterministicParallelTests`) must pass unchanged, and the full suite (~360 tests) stays green.

## 3. The source generators (MintPlayer.Dotnet.Tools)

Two nuget.org packages, both **10.20.0**, both required:

- **`MintPlayer.SourceGenerators.Attributes`** — `[Inject]`, `[Register]`, `[PostConstruct]`,
  `[Config]`, `[ConnectionString]`, `[Options]`, `[RegisterFactory]` (netstandard2.0; only runtime dep
  is `Microsoft.Extensions.DependencyInjection.Abstractions`).
- **`MintPlayer.SourceGenerators`** — the incremental Roslyn generators (plain `PackageReference`;
  self-installs as analyzer — this repo already references it, see `CampaignRunner.cs:45`).

Semantics that matter here:

- **`[Inject]`** on fields / get-only properties **generates the constructor** into a `partial` class,
  assigning each member from a same-named param. Nullable member ⇒ optional param. Base-ctor chaining is
  automatic (base's `[Inject]` members, or its largest ctor, become forwarded params). Generics fully
  supported. If a matching ctor already exists, generation is silently skipped.
- **`[Register(ServiceLifetime.X, "MethodNameHint")]`** on a class (self or `typeof(IService)` first arg)
  emits one `public static class DependencyInjectionExtensionMethods` per assembly with a chained
  `services.Add<Hint>()` extension. Open generics supported. Assembly-level overload registers
  third-party types. `[RegisterFactory]` on a static method registers via factory.
- **Limits relevant to this repo:** no *keyed* services (the `--game <name>` → campaign mapping needs a
  hand-rolled registry or plain `switch`); `[Inject]` classes must be `partial`; `[Config]`/`[Options]`
  need the Configuration/Options runtime packages if we use them (not required for plain `[Inject]`).

The repo already dogfoods `[Register]` (`CampaignRunner` → generated `AddReinforcementLearningCore()`,
called from `AIHost.AddReinforcementLearning`, documented in `ARCHITECTURE.md` §11). This PRD extends
that pattern rather than introducing anything new.

## 4. Design

### 4.1 Design it twice — where does construction move?

**Option A — full container resolution:** register every campaign as an `ITrainingCampaign` service
(keyed by game name), bind CLI flags into `IOptions<T>`, and have `LabHost.Run` resolve by `--game`.
Rejected as the *first* step: the generator has no keyed services, CLI→options binding for 7 campaigns
× ~14 flags each is a big bang, and `SelfPlayCampaign<TState>` is generic over the game state — the
container would need one closed registration per game anyway.

**Option B (chosen) — campaigns become public, DI-*constructible* library types; each Lab registers its
game's services and the `build` lambda shrinks to a resolution call.** The
`Func<IServiceProvider, ITrainingCampaign> build` parameter of `LabHost.Run` stays (it is a good seam —
a factory *is* the DI-idiomatic way to combine runtime CLI values with container services), but what it
does changes from "hand-construct everything" to "resolve services, pass options". Per-game registration
extensions (`AddChessSelfPlay()`, `AddSnakeDqn()`, …) are `[Register]`-generated where the shape fits and
hand-written where it doesn't (keyed/generic cases). This is incremental, testable per game, and never
breaks the determinism gate.

### 4.2 Target shape

- **New class library `src/MintPlayer.AI.ReinforcementLearning.Campaigns`** referencing Core +
  Environments + Ilgpu. All 8 campaign types (`SelfPlayCampaign<TState>`, `DqnScoreCampaign`,
  `SnakeDqnCampaign`, `FruitCakeDqnCampaign`, `CubeImitationCampaign`, `RushHourImitationCampaign`,
  `CubeEfficientCampaign`, `CubeDaviCampaign`) move there and become `public`. The Lab exe keeps CLI
  parsing, GPU factory wiring, `VizLauncher`, and the per-game `Run(args)` glue.
  Layout (owner, 2026-07-14): **per-game subfolders in both projects** — Campaigns gets
  `SelfPlay/ Cube/ Snake/ FruitCake/ RushHour/ Shared/`, the Lab gets `Chess/ Connect4/ Cube/ FruitCake/
  RushHour/ Snake/` with the shared host glue (`Program`, `LabHost`, `CliArgs`, `CampaignCli`, viz) at its
  root. Folders are organizational only: the library keeps ONE flat `…Campaigns` namespace so consumers
  don't chase per-game usings.
  Rationale for one project (not per-game): campaigns share `DqnScoreCampaign`/telemetry plumbing, and
  the two cube GPU campaigns take `AdaptiveBackend` directly — a single lib referencing Ilgpu is the
  honest dependency graph. (`SelfPlayCampaign` itself still must not use `Ilgpu` types — enforced by
  its factory params, unchanged.)
- **Options records per campaign family** (`SnakeDqnOptions`, `FruitCakeDqnOptions`, `CubeImitationOptions`,
  `RushHourImitationOptions`, `CubeEfficientOptions`; `SelfPlayOptions` and `CubeDaviSettings` already
  exist): the positional-primitive ctors collapse to `(deps…, XxxOptions options)`. Options are plain
  records built from `CliArgs` in the Lab — no `IConfiguration` binding needed (the Lab is flag-driven,
  not appsettings-driven).
- **Environment injection for the DQN family:** `DqnScoreCampaign` ctor gains the train/eval
  `IEnvironment<float[], int>` instances (or a factory when the campaign needs fresh instances);
  subclasses stop `new`ing envs in field initializers. Self-play already takes its game — unchanged.
- **Games registered as singletons:** `ChessGame`, `Connect4Game` get
  `[Register(typeof(IZeroSumGame<ChessState>), ServiceLifetime.Singleton, …)]`-style registrations
  (attributes live on the classes in Environments; that project must reference the two generator
  packages — verify no NuGet-consumer friction since Environments ships to nuget.org: the attributes
  package becomes a public dependency, acceptable, it's tiny and Apache-2.0).
- **`[Inject]` adoption where it pays:** RLDemo.Web model services (`RushHourModelService`,
  `Game2048ModelService`, `CubeModelService` — currently hand-written ctors taking
  `IModelStore`/`AdaptiveBackend`/`ILogger<T>`) become `partial` with `[Inject]` members +
  `[Register]`, replacing the hand-list in `Program.cs:39-47`. Campaign classes adopt `[Inject]` only
  where the generated ctor matches what we'd write by hand (the options param + seams); if a campaign's
  ctor needs logic, keep it hand-written — the generator skips generation when a ctor exists, so mixing
  is safe.
- **Ladder persistence behind a seam:** route `PromoteTier`/`WriteManifest`/`LoadLadderState`
  (`SelfPlayCampaign.cs:538-671`) through `IModelStore` — extending it with the few missing operations
  (enumerate/delete/move by key) rather than adding a parallel `ILadderStore`, unless the extension
  distorts `IModelStore`'s abstraction (decide at implementation; deep-module bias says one store).
- **Logging:** campaigns take an optional `ILogger` (defaulting to a console logger in the Lab so
  today's output format is preserved); the static `Log(...)` helpers delegate to it.

### 4.3 What stays out

Polyglot cores and facades' internals; `Backend.Current` / `GradMode` / static observation writers;
`RLDemo.Console` (a demo scratchpad — not worth DI-ifying; optionally delete-or-document later);
web-side training (web stays load-only); keyed-service support upstream in the generator (nice-to-have,
tracked as an upstream FR, not a blocker).

## 5. Milestones

- **M46.1 — Campaigns become a public library.** Create
  `src/MintPlayer.AI.ReinforcementLearning.Campaigns`; move the 8 campaigns + supporting types out of
  the Lab exe; make them `public`; retire `extern alias Lab` + `InternalsVisibleTo` in the test project
  (also resolves the `Program`-collision hack). Lab keeps CLI/GPU/viz glue. **Gate:** full suite green
  with the aliases gone; checkpoint SHA tests bitwise-identical; Lab + Web build clean.
- **M46.2 — Options records + injected environments.** Introduce the per-campaign options records;
  collapse positional-primitive ctors; `DqnScoreCampaign` and subclasses receive their envs via ctor.
  **Gate:** a DQN campaign runs against a stub `IEnvironment` in a unit test; SHA tests unchanged.
- **M46.3 — Source-generated registration end-to-end.** `[Register]` on games + campaigns (closed
  registrations per game where generic), `[Inject]`+`[Register]` on the web model services replacing the
  hand-list in `Program.cs`; per-game `Add<Game>Campaign()` extensions; each Lab's `build` lambda shrinks
  to resolve-and-go. **Gate:** container smoke test — every registration resolvable on both the CPU and
  GPU (`AddGpuBackend`) paths; web integration tests green.
- **M46.4 — Ladder persistence through the store seam + `ILogger`.** No raw `File.*`/`Directory.*` left
  in campaigns. **Gate:** ladder promotion unit-tested against an in-memory store (promote/manifest/resume
  round-trip), no disk touched.
- **M46.5 — Unit-test the new seams.** Following existing conventions (`<Subject>Tests`,
  `Subject_Condition_Expectation`, `[Trait("Category","Slow")]` for anything heavy): DQN-campaign contract
  tests via stub env; CliArgs→options mapping tests; DI smoke tests; ladder tests from M46.4. **Gate:**
  campaigns/games testable without disk, GPU, or `extern alias`; suite green.

Sequencing note: M46.1 → M46.2 are ordered; M46.3–M46.5 can interleave per game. Every milestone is
individually shippable and individually gated on bitwise-identical training.

## 6. Open questions

1. `IModelStore` extension vs. separate `ILadderStore` — decide when the ladder operations are listed
   concretely (M46.4).
2. Does the Campaigns library ship to nuget.org alongside Core/Environments (SDK north star says
   probably yes — campaigns are the reusable harness), or stay repo-internal at first? Default: publish
   once M46.3 lands.
3. Upstream FR for keyed-service support in `MintPlayer.SourceGenerators` (would let the `--game`
   registry be generated too) — file after M46.3 shows the hand-rolled shape.

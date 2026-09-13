# Adding a new game end-to-end

> ## ⚠️ Read this first — the default is CLIENT-SIDE
>
> The six-layer checklist below describes the **server-authoritative** path (env → model service → controller →
> `*-api.ts`). That was the original design and it is now the **exception**, kept for models that genuinely
> cannot run in a browser — today only **Rush Hour, 2048 and Cube**.
>
> **Every game added since M32 is fully client-side**, and that is what you should reach for: Snake, MountainCar,
> FruitCake, Crazy Fruits, Tetris, Chess, Draughts, Block Dude and Lunar Lockout. They have **no controller and
> no model service at all**, and cost nothing per viewer.
>
> ### The client-side checklist
> 1. **One `.pg` single source** — `src/…Environments/<Game>/polyglot/<game>_solver.pg`: rules,
>    `buildObservation`, the net forward pass, and any scripted or search tiers. Use `constructor(...)`, never
>    `init(...)`. The richest example is `Tetris/polyglot/tetris_solver.pg`.
> 2. **Route the TS twin** — add an `include` entry to the repo-root `pgconfig.json` pointing at
>    `src/RLDemo.Web/ClientApp/src/app/<game-dir>/<game>_solver`. `dotnet build` emits both C# and TS. The
>    `*_solver.ts` twins are **gitignored build outputs — never edit them**.
> 3. **C# facade + env** beside the `.pg` (`<Game>Board.cs`, `<Game>Env.cs`), delegating rules to the generated
>    core. Observation built by a **static** method so training and serving are byte-identical.
> 4. **Campaign** in `Campaigns/<Game>/`, registered in `CampaignServiceCollectionExtensions`, plus a Lab entry
>    and a `--game <name>` line in `Lab/Program.cs`. Persist progress counters and RNG state — see
>    `CampaignProgressState`, added in M58 after three campaigns were found to be silently replaying data on
>    every restart.
> 5. **Level content**, if any: one canonical JSON under `<Game>/levels/`, embedded for training and copied into
>    `wwwroot/levels/` by the `CopyLevelPacks` target, so the browser reads the exact bytes the campaign trains
>    against. Generate it with a committed script in `tools/`.
> 6. **Ship the weights** to `src/RLDemo.Web/wwwroot/models/` (LFS), with a `<game>-net.ts` `.ckpt` parser and a
>    **stale-checkpoint guard** — input width ≠ observation width must fall back to a scripted tier, never
>    half-load. (`PolicyValueNet.Load` now throws on a shape mismatch; before M58 it silently left the tail of
>    layer 0 at random init.)
> 7. **Page**: component + renderer + `.scss`, a lazy route in `app.routes.ts`, a nav link in `app.html`, a home
>    card in `home/home.ts`. Canvas loop outside Angular's zone, plain `fetch`, `touch-action: none` on any
>    canvas you drag on.
> 8. **Tests** in `tests/…Tests/`: engine rules, env contract, C#-vs-generated parity, and a `.ckpt` byte
>    reference for the TS parser.

The mechanical, file-by-file checklist for the **server-authoritative** path, reverse-engineered from
**Rush Hour** / **2048** / **Cube** (investigated 2026-06-15). Cross-refs: `prd/PRD.md` §7 + §7.1 (interaction
models), `prd/PLAN.md` M8–M10 (web slices) + M22 (MountainCar/Snake).

For a new game `X` (lowercase env id `xgame`, PascalCase `XGame`), pick the **interaction principle** first
(PRD §7.1): **A — compute-and-return** (HTTP, like Cube/2048/RushHour/Snake) or **B — live control stream**
(WebSocket, like MountainCar). Then work the six layers.

> **Note:** the live WebSocket stream (principle B) was **retired in M32/M33**. Every "watch AI" mode now runs
> in the browser. Principle B is documented here for historical reading of the code, not as a choice to make.

## Load-bearing conventions
- **Model-store filename:** `FileModelStore.PathOf` maps `(environmentId, algorithmId)` → `<root>/<envId>.<algoId>.ckpt`.
  The same `(env, algo)` key must match across the service, the console trainer, and the shipped `models/*.ckpt`.
- **Seed flow:** `Program.cs` copies every `models/*.ckpt` (the `SeedModelsDirectory`) into `/data` at startup if
  absent. Drop the trained checkpoint in repo `models/` (tracked via **Git LFS** — `*.ckpt`) and it ships to fresh
  clones + the Docker image. CI must `git lfs pull` before `docker build` (workflow checks out `lfs: true`).
- **Warmup:** every `ITrainableModelService` is run once at startup (off the request path) by
  `ModelTrainingHostedService`; each first tries `TryLoadFromStore`, else trains + saves.
- **503-while-training** and **`{ error }`-400** are the two contracts the Angular `*-api.ts` switches on — keep them.

## 1. Environment — `src/MintPlayer.AI.ReinforcementLearning.Environments/XGame/XGameEnv.cs`
`public sealed class XGameEnv : IEnvironment<float[],int>` (+ `IActionMaskProvider` if some moves are illegal;
+ `IStatefulEnvironment` for bitwise-resumable training — cheap, recommended). `BoxSpace` obs / `DiscreteSpace`
actions in the ctor; seeded `Xoshiro256StarStar` owned by the env; `Reset(ulong? seed)` reseeds only when a seed
is given; `Step` throws if `_done` / on an illegal action; keep **terminated vs truncated** distinct. Add the pure
game logic + (if NN-trained) any oracle/generator/policy-net/search beside it. No DI needed (plain library).

## 2. Model service — `src/RLDemo.Web/Services/XGameModelService.cs`
`sealed class XGameModelService(IModelStore store, ILogger<…> logger) : IModelStartupService`. Constants
`EnvironmentId="xgame"`, `AlgorithmId="dqn"|"ppo"|…`. `TryLoadFromStore()` (lazy, locked) → deserialize +
build the agent; `Initialize(ct)` → load the shipped checkpoint, or set `Status=Failed` if absent. **The web never
trains** (PRD §14 / M26): produce the checkpoint on a dev machine (the Lab campaign — §8 — or the Console, §6) and
commit it to `models/` via Git LFS. Status fields + `Error`. (Optional refreshing secondary net like
`CubeModelService.PolicyNet`/`ValueNet`.)

## 3. Controller / WS handler — `src/RLDemo.Web/Controllers/XGameController.cs`
**Principle A:** `[ApiController][Route("api/xgame")]`, inject `(XGameModelService model, GalleryStore gallery)`.
`[HttpGet("status")]` (touch `_ = model.Agent` to lazy-load); `[HttpPost("solve")]` → validate (400 `{error}`),
`if (agent is null) return StatusCode(503, Status())`, run rollout, `gallery.Add("xgame", summary, request, response)`,
return the full trajectory DTO. **Principle B:** a WebSocket endpoint (`app.UseWebSockets()` + a handler that owns
an env + agent and streams `(state, action, done)` frames) instead of `POST /solve`.

## 4. DI + startup — `src/RLDemo.Web/Program.cs`
`AddSingleton<XGameModelService>()` **and** `AddSingleton<IModelStartupService>(sp => sp.GetRequiredService<XGameModelService>())`
(concrete for the controller, interface for the startup load via `ModelStartupHostedService`). Principle B also adds
`app.UseWebSockets()` + maps the WS route.
Seed-copy + hosted service are game-agnostic — no change.

## 5. Frontend — `src/RLDemo.Web/ClientApp/src/app/x-game/`
`x-game-api.ts` (a `fetch` wrapper + a `{kind:'solved'|'invalid'|'training'}` union, or a WS client for B),
`x-game.ts`/`.html`/`.scss` (standalone signals component; DOM grid like 2048 or `<canvas>`), optional
`x-game-logic.ts` (pure client engine for human play / client-side animation). Register: add a lazy route in
`app.routes.ts`, a nav link in `app.html`, a card in `home/home.ts`, and a `gameLabel` case (+ optional color) in
`gallery/gallery.ts`. **Do not run `ng serve`/`ng build`** — the host runs the embedded dev server (`UseAngularCliServer`).

## 6. Console (offline seed-checkpoint production) — `src/RLDemo.Console/Program.cs`
Add `"xgame"` to `knownSections`, the env `using`, and an `if (ShouldRun("xgame")) { … }` block (DQN: `TryLoadMlp`/
`DqnTrainer.Train`/`SaveMlp("xgame","dqn",(Mlp)result.Network)`; custom agent: `store.Save("xgame","…", s => agent.Save(s))`).
Run `RLDemo.Console xgame --save --data ../../models` to produce the seed `.ckpt`; commit it (LFS).

## 7. Tests — `tests/MintPlayer.AI.ReinforcementLearning.Tests/XGameApiTests.cs`
Mirror `RushHourApiTests`/`CubeApiTests`: spin up the host in the `Testing` environment (skips SPA + warmup),
control the store directly, assert the solve/status/503/400 contracts. Add env-level + (if applicable) gate tests.

## 8. Long training — the campaign harness (`tools/…Lab/`, PLAN M25)
For multi-hour/resumable training (vs. the web's one-shot `EnsureModel`), write an **`ITrainingCampaign`** instead of
a bespoke loop. The shared **`CampaignRunner`** owns the wall-clock budget, eval/checkpoint cadence and resume; you
implement only the game-specific parts:

```csharp
internal sealed class XGameCampaign(...) : ITrainingCampaign
{
    public string Environment => "xgame";
    public bool Resume(IModelStore store) { /* load net+optimizer+state, or start fresh; return whether resumed */ }
    public long TrainChunk()    { /* train one chunk; return cumulative progress (steps/samples) */ }
    public CampaignEval Evaluate() { /* return CampaignMetrics + a one-line summary (solve-rate OR mean return) */ }
    public void Checkpoint(IModelStore store) { /* persist the deployable net + full resume state */ }
    public bool IsComplete => false;                       // score-maximizing: run to the time budget
    // public bool IsComplete => samples >= target;        // goal-reaching: optional hard stop
    public void Dispose() { }
}
```

There is **one** interface for both paradigms (see PRD §14): the only difference is what `Evaluate` reports
(solve-rate vs mean return) and whether `IsComplete` ever fires — no per-paradigm base class. Wire a `--game xgame`
entry in the Lab that resolves the runtime from DI and runs the campaign:

```csharp
var builder = AIHost.CreateBuilder(dataDir);
builder.Services.AddGpuBackend();                          // only if the net is large enough to win on GPU
using var host = builder.Build();
host.Services.GetRequiredService<CampaignRunner>().Run(
    new XGameCampaign(...), host.Services.GetRequiredService<IModelStore>(),
    new CampaignOptions { Duration = TimeSpan.FromHours(h), OnEval = CampaignCli.ConsoleAndCsv(csvPath) });
```

Keep all console/CSV IO in the `OnEval` callback (`CampaignCli`), not in the campaign — the runner does no IO itself.
Add a Slow contract test (`CampaignContractTests`): fresh → `TrainChunk` advances → `Checkpoint` → a fresh instance
`Resume`s and continues rather than restarting.

# Adding a new game end-to-end

The mechanical, file-by-file checklist for adding a game to the playground, reverse-engineered from
**Rush Hour** / **2048** / **Cube** (investigated 2026-06-15). Cross-refs: `PRD.md` §7 + §7.1 (interaction
models), `PLAN.md` M8–M10 (web slices) + M22 (MountainCar/Snake) + M23 (Pendulum/SAC).

For a new game `X` (lowercase env id `xgame`, PascalCase `XGame`), pick the **interaction principle** first
(PRD §7.1): **A — compute-and-return** (HTTP, like Cube/2048/RushHour/Snake) or **B — live control stream**
(WebSocket, like MountainCar/Pendulum). Then work the six layers.

**Continuous-action games** (a real-valued action, like Pendulum) differ only at a few seams: the env is
`IEnvironment<float[],float[]>` with a **`BoxSpace` action space** (not `DiscreteSpace`), the agent is a
`ContinuousPolicyAgent` over a Gaussian policy net (greedy = tanh(mean), rescaled to the box bounds), trained with
**SAC** (`SacTrainer`), and the WS `EpisodeStreamer` is used with `TAct = float[]` (pass a zero-vector
`resetAction`). Serve-time checkpoint is the actor `Mlp` via `MlpCheckpoint` (critics/temperature are training-only).
Human play maps ←/→ to a continuous torque rather than a discrete button.

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
`sealed class XGameModelService(IModelStore store, ILogger<…> logger) : ITrainableModelService`. Constants
`EnvironmentId="xgame"`, `AlgorithmId="dqn"|"ppo"|…`. `TryLoadFromStore()` (lazy, locked) → deserialize +
build the agent; `EnsureModel(ct)` → load-or-train-and-save; status fields + `Error`. (Optional refreshing
secondary net like `CubeModelService.PolicyNet`/`ValueNet`.)

## 3. Controller / WS handler — `src/RLDemo.Web/Controllers/XGameController.cs`
**Principle A:** `[ApiController][Route("api/xgame")]`, inject `(XGameModelService model, GalleryStore gallery)`.
`[HttpGet("status")]` (touch `_ = model.Agent` to lazy-load); `[HttpPost("solve")]` → validate (400 `{error}`),
`if (agent is null) return StatusCode(503, Status())`, run rollout, `gallery.Add("xgame", summary, request, response)`,
return the full trajectory DTO. **Principle B:** a WebSocket endpoint (`app.UseWebSockets()` + a handler that owns
an env + agent and streams `(state, action, done)` frames) instead of `POST /solve`.

## 4. DI + startup — `src/RLDemo.Web/Program.cs`
`AddSingleton<XGameModelService>()` **and** `AddSingleton<ITrainableModelService>(sp => sp.GetRequiredService<XGameModelService>())`
(concrete for the controller, interface for warmup). Principle B also adds `app.UseWebSockets()` + maps the WS route.
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

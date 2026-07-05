# Client-side AI for Snake & MountainCar (PRD & Plan) — M33

> **Goal (user's steer):** do for the remaining WebSocket-AI games what M32 did for FruitCake — move the env
> **dynamics** + **observation** + **net forward** (+ any serving "wrapper" like masking) into a MintPlayer.Polyglot
> `.pg` where feasible, ship the trained **checkpoint** as a browser asset, run the AI **client-side**, and delete
> the server serving path. Per-viewer server inference/streaming → **zero**.

- **Status:** ✅ **IMPLEMENTED — both games client-side, 2026-07-05.** Snake + MountainCar AI now run entirely in
  the browser (single-source `.pg` env+obs+net, C# facades, browser directors, server path + `EpisodeStreamer`
  retired). Verified host + Playwright (Snake ate 40; MountainCar reaches the flag), 0 console errors, 0 `/api/*`
  AI calls. Branch `snake-clientside-ai`. See PLAN.md M33 for commit refs.
- **Author:** Pieterjan (with Claude Code)
- **Precedent:** `FRUITCAKE_CLIENT_SIDE_AI_PRD.md` (M32, shipped) — the pattern this replicates.

---

## 1. Scope — exactly two games

The only games whose AI "watch" mode holds an open WebSocket are **Snake** (`/api/snake/live`) and **MountainCar**
(`/api/mountaincar/live`), both driven by the shared `EpisodeStreamer`. The other three are **request/response REST**
batch solvers with no per-viewer socket, so they need nothing here:
- **2048** — `POST /api/2048/solve` (expectimax, one response).
- **RushHour** — `POST /api/rushhour/solve` (policy-guided A*, one response).
- **Cube** — `POST /api/cube/solve*` (beam search, one response).

So M33 = **Snake + MountainCar**, as **two independent PRs** (they share only `EpisodeStreamer` and the checkpoint
parser). **Snake first** (clean, high-value, lowest new-code); MountainCar second (has a real complication — §5).

---

## 2. What we confirmed (3-agent investigation + Polyglot 0.2.0 probes, 2026-07-05)

### Polyglot 0.3.0 (verified 2026-07-05 — supersedes the 0.2.0 notes below)
**0.3.0 fixes all of #11** (verified by probe): **transcendentals added** (`cos`/`sin`/`tan`/`exp`/`log`/`pow`/`tanh`
— emit `System.Math.Cos`/`Math.cos` etc.); **module imports now LINK** (TS emits `import {…} from "./nn"`, C# no
longer inlines → a shared `nn.pg` is viable, no CS0101); **`i32()` TS wrap fixed** (`Math.trunc(x)`, no `|0`). #9
stays fixed. **Consequence: MountainCar's transcendental fork DISSOLVES** — `cos`/`tanh` can now be written directly
in a `.pg`; the C#↔TS sub-ULP drift is harmless (no server twin post-cutover, argmax-robust), so MountainCar gets the
same uniform full-Polyglot treatment as Snake (no poly-approx). §5's Option A is now the clean default. The
Environments `.csproj` is bumped 0.1.4 → **0.3.0** (FruitCake TS codegen verified content-identical, 22 tests green).

### ⚠️ BLOCKER discovered (MintPlayer.Polyglot#14) — multi-`.pg` prelude collision
Adding a **second** `.pg` (Snake) to the Environments project fails to compile: **every transpiled `.pg` emits its
own copy of the prelude** (`Option`/`Some`/`None` records + `PolyglotProgram`), so two generated `.cs` in one
assembly hit **CS0101 duplicate** (+ CS8863). FruitCake only worked as the *sole* `.pg`. This contradicts the
package targets' own §4.5 comment ("linked build → each type defined once → no CS0101") — the *import* linking works,
but the auto-emitted prelude isn't deduped. **`snake_solver.pg` is written, type-checks, and generates correct
C#/TS individually**, but is **parked** (excluded via `PolyglotFile Remove` in the csproj) until the fix. Given the
0.1.4→0.2.0→0.3.0 turnaround, waiting for a prelude-dedup fix is preferred over a per-game-project workaround. Once
fixed: delete the `Remove`, then continue SN1–SN6 (facade, parity tests, parser reuse, director, retire server).

### Polyglot 0.2.0 (historical)
- **All three M32 codegen bugs are FIXED** (issue MintPlayer.Polyglot#9, closed): `i32()/f64()` casts now lower to
  `(int)x` / `Math.trunc(x)|0` / `(double)n`; `var x: T? = null` emits a typed decl; 2+ nested-generic params parse.
  ⇒ **new `.pg` files can use casts, nullable locals, and nested generics freely** (no M32 workarounds needed).
- **Integer `/` and `%` codegen correctly** (`a/b` → `(a/b|0)` in TS; `a%b` → `(a%b|0)`) — Snake's grid index math
  (`head/Size`, `head%Size`, non-negative) is clean.
- **std.math has NO transcendentals.** Only `abs`, `floor`, `sqrt` (+ `min`/`max`) exist; **`cos`, `sin`, `exp`,
  `pow`, `tanh` are absent** — consistent with "only `+ − × ÷ √` are byte-identical across .NET and JS." This is the
  crux for MountainCar (§5).
- **User `.pg`→`.pg` imports DO work** (`import { X } from "./nn"` and `from "nn"` both check + build) — but via
  **inlining**: the importer's output gets a *full copy* of the imported symbols (no cross-module reference emitted).
  In the MSBuild "glob every `**/*.pg` → compile all into one assembly" setup, a standalone library `.pg` would then
  be defined **twice** (once from its own transpile, once inlined into the importer) → CS0101 duplicate type. A shared
  `nn.pg` is therefore feasible only if the library file is **excluded from the C# transpile glob** (and not committed
  as a standalone TS). ⇒ **For two games, duplicate the net-forward per game** — Snake copies `PgDuelingNet` verbatim
  (~70 lines), MountainCar gets a small `PgMlpNet`; the tiny duplication is simpler than the build-config work a
  shared module needs. (Revisit a shared `nn.pg` only if a third+ net-bearing game appears.)

### Checkpoint parser generalization
The `.ckpt` envelope (`CheckpointFormat`) is shared: `RLNC` magic + 7-bit-length kind string + int32 version, then
length-prefixed int/float/bool blocks. Two payload shapes:
- **`dueling-q`** (Snake) — identical to FruitCake. **`fruitcake-net.ts` parses it unchanged.**
- **`mlp`** (MountainCar) — `Ints Sizes` + `byte HiddenActivation` + per-layer `(W, b)` floats.
⇒ Lift the shared header/primitive readers out of `fruitcake-net.ts` into a **shared `ckpt.ts`** (`parseDuelingQNet`
+ a new `parseMlp`); each game's loader hands the parsed tensors to *its* generated net class.

### EpisodeStreamer
`src/RLDemo.Web/Services/EpisodeStreamer.cs` is used **only** by Snake + MountainCar → **deletable** once both
migrate (delete in whichever PR lands second).

---

## 3. Snake — full Polyglot client-side port (the clean one)

**Facts** (`Environments/Snake/SnakeEnv.cs`; net `snake.dqn.ckpt`):
- **Observation = 177** (a 9×9×2 egocentric patch = 162, + 15 scalars). *(Note: the "8-ray obs" memory is stale —
  it's the 177-wide patch.)* Pure integer/`+−×÷` math → **byte-identity is free**.
- **Net = plain `DuelingQNet` 177 → [256,256] → 4**, checkpoint kind `dueling-q` v1 → **reuse `PgDuelingNet.forward`
  and the existing parser verbatim.**
- **No search** — one greedy, **masked** Q-step per tick. Much simpler than FruitCake.
- **The "wrapper" that must move to the browser = the action mask** (`CurrentActionMask` + `ReachableFreeSpace`):
  (a) forbid the 180° reversal onto the neck; (b) the **anti-self-trap shield** — a **BFS flood-fill** keeping only
  moves whose reachable free space ≥ snake length (never all-false). The net was trained *with* this shield
  (`safeMask:true`), so serving must reproduce it exactly.
- **Collections to re-express** (the main porting effort): `SnakeEnv` uses `LinkedList<int>` (body), `HashSet<int>`
  (occupancy), `Queue<int>` (BFS) — Polyglot's `std.collections` is `List<T>` only. Re-express as: body =
  `List<i32>` used as a deque (add-front / remove-last); occupancy = `List<bool>` of size `Cells`; BFS = `List<i32>`
  + head cursor, `seen` = `List<bool>`. All flat, all integer.
- **Food RNG** → `Math.random` client-side (nondeterministic, fine for a watch view; no need to port Xoshiro).
- **Rendering** is a CSS-grid of `<div>`s (not canvas). Director `toFrame()` → `{body, food, foodEaten, done,
  length}`, so the existing `render()` is unchanged. **Discrete tick** — reuse a `setInterval`/timer (like human
  mode), not a RAF physics loop.

### Snake plan (SN0–SN6) — ✅ DONE 2026-07-05
> **Shipped (branch `snake-clientside-ai`).** `snake_solver.pg` single-sources the dynamics + 177-dim observation +
> action mask (incl. the flood-fill shield) + `PgSnakeNet`; C# `SnakeEnv` is now a facade over it (9 SnakeEnvTests
> green). Net shipped at `ClientApp/public/snake-net.ckpt` (+ `snake-net.ts` parser); `SnakeDirector` runs the AI in
> the browser; server path (controller/service/`snake-api.ts`/stale net/`SnakeApiTests`) deleted. Verified host +
> Playwright: the AI plays (ate 40, length ~35), 0 console errors, weights from `/snake-net.ckpt`, **0 `/api/snake`
> calls**. Flood-fill was rewritten BFS→relaxation to dodge a Polyglot evolving-any TS7022 (net-agnostic; MintPlayer.Polyglot#14 was the multi-.pg prelude fix in 0.3.1).
- **SN0 — `snake_solver.pg` scaffold + env dynamics.** Port `SnakeEnv` step (move/grow/collision/food, tail-follow
  ordering) into `PgSnakeEnv` over flat `List` collections; bump the Environments `.csproj` Polyglot ref to **0.2.0**.
  Gate: builds; a C# facade delegates; parity test vs `SnakeEnv` step outcomes.
- **SN1 — observation in the `.pg`.** `buildObservation() -> f64[177]` (9×9×2 patch + 15 scalars + the flood-fill
  free-space features). Gate: reproduces `SnakeEnv.Observation()` exactly (integer math → exact, not tolerance).
- **SN2 — action mask in the `.pg`.** `currentActionMask() -> [bool;4]` (reversal + flood-fill shield). Gate: matches
  `SnakeEnv.CurrentActionMask()` on played boards.
- **SN3 — net forward.** Copy `PgDuelingNet` into `snake_solver.pg` (verbatim from FruitCake). Gate: argmax-exact vs
  SDK `DuelingQNet.Forward` (reuse the M32 parity-test pattern).
- **SN4 — checkpoint delivery + shared parser.** Copy `models/snake.dqn.ckpt` → `ClientApp/public/snake-net.ckpt`
  (LFS); refactor `fruitcake-net.ts` → shared `ckpt.ts` (`parseDuelingQNet` unchanged). Gate: browser parses it →
  177/[256,256]/4.
- **SN5 — client director + rewire watch mode.** `snake-director.ts`: masked greedy step over `PgSnakeEnv` +
  `PgDuelingNet`, `toFrame()` for the existing renderer; rewrite `snake.ts` watch mode to drive it (drop
  `snake-api.ts`). Reconcile human-play `snake-logic.ts` onto the single-source `PgSnakeEnv` (eliminate the twin).
  Gate: host + Playwright — Snake AI plays in-browser, 0 console errors, 0 `/api/snake` calls.
- **SN6 — retire server path.** Delete `SnakeController`, `SnakeModelService` (+ `Program.cs` regs), `snake-api.ts`,
  stale `models/snake.dqn.ckpt`; update `SnakeApiTests`. Gate: build + tests green; Playwright watch still works.

---

## 4. Reusable M32 machinery (both games)

Build-time C# transpile via `MintPlayer.Polyglot.MSBuild` (bump **0.1.4 → 0.2.0**); committed generated `.ts`
(regenerate on Windows with `polyglot build … --target typescript`); the internal-core + public-facade pattern +
`InternalsVisibleTo` for tests; **`.ckpt` committed directly to `ClientApp/public/` via LFS** (M32 did *not* use an
MSBuild copy — the asset is committed); the async-load client **director**; and the **Polyglot-C# == SDK-C#
equivalence tests** (+ C#↔TS byte-identity free for integer/`+−×÷√` code).

---

## 5. MountainCar — the transcendental fork (needs a decision)

MountainCar is tiny (env ≈ 15 lines; net = `Mlp [2,64,64,3]`), but it has **two transcendentals on the client path**,
and **neither can be written in a `.pg`** (std.math has no `cos`/`tanh`):
1. **`cos(3·position)`** in the env dynamics (every step).
2. **`tanh`** — the PPO actor's hidden activation (`Mlp`, `Activation.Tanh`). Serving = `argmax` over 3 logits.

(The observation itself — `(pos+0.3)/0.9`, `vel/MaxSpeed` — is pure `+−×÷`, so the *net input* is byte-safe; the
checkpoint is kind `mlp`, needing a new `parseMlp` + a `PgMlpNet` forward.)

> **UPDATE (Polyglot 0.3.0):** `cos` and `tanh` are now in std.math, so both CAN be written in a `.pg`. They are not
> byte-identical across C#/TS (~1 ULP), but that no longer matters (no server twin after cutover; argmax-robust). So
> **Option A is now the clean default — no polynomial approximations needed.** The fork below is retained for record.

So MountainCar **cannot** get FruitCake's "byte-identical single source" for free. Two viable approaches:

- **Option A — uniform Polyglot, with byte-identical polynomial approximations of `cos` and `tanh`** (pure `+−×÷`).
  *Pro:* true single-source + C#↔TS byte-identity, matching the uniform goal. *Con:* the approximations change the
  numbers vs the trained float32 `Math.Cos`/`Math.Tanh`, so it needs a **validation gate** (argmax unchanged across a
  position/velocity sweep vs the real net) — feasible if the approximations are accurate (~1e-6), but real work and a
  small risk of needing a re-eval/retrain.
- **Option B — pragmatic client-side port, no Polyglot for the transcendental parts (RECOMMENDED).** Reuse the
  existing hand-written `mountaincar-logic.ts` dynamics (it already has `Math.cos`), hand-write the ~30-line
  `Mlp`+`tanh` forward in TS (JS `Math.tanh` — a ULP off .NET, argmax-robust), add `parseMlp`, ship the ckpt, delete
  the socket. *Pro:* achieves the actual goal (zero server cost, client-side AI) with minimal work and **no net
  perturbation**. *Con:* keeps a ~15-line C#/TS env twin (but MountainCar dynamics are stable and never change, so
  drift risk is negligible), and doesn't "single-source via Polyglot."

**Recommendation:** **Option B.** For a 15-line env whose twin never changes, Polyglot single-sourcing buys almost
nothing, while the transcendentals actively fight it. The cost win (removing the socket) is identical either way.
Reserve Option A only if *uniform Polyglot single-source* is a hard requirement — in which case gate it behind an
approximation-accuracy spike (**MC0**) before committing.

### MountainCar plan (MC0–MC5) — ✅ DONE 2026-07-05 (uniform Polyglot, per the 0.3.0 update above)
> **Shipped.** `mountaincar_solver.pg` single-sources the dynamics (`cos`) + normalised obs + `PgMlpNet` (Tanh MLP,
> argmax); C# `MountainCarEnv` is a facade (6 tests green, incl. the Gymnasium golden). Net at
> `ClientApp/public/mountaincar-net.ckpt` + `mountaincar-net.ts` (`parseMlp`); `MountainCarDirector` runs it;
> server path + `EpisodeStreamer` deleted; `app.UseWebSockets()` removed (no server WS left). Verified in-browser.
- **MC0 (only if Option A) — approximation spike.** Implement `cos`/`tanh` in pure `+−×÷`; verify argmax matches the
  real net across a state sweep. Pass → proceed Polyglot (mirror Snake's SN plan). Fail → fall back to Option B.
- **MC1 — `parseMlp` in the shared `ckpt.ts`** (Sizes + activation byte + per-layer W/b).
- **MC2 — TS `Mlp` forward** (`mountaincar-net.ts`: linear + tanh per hidden layer, linear head, argmax). Gate:
  matches SDK `Mlp.Forward`/`PolicyAgent.Act` argmax on a state sweep.
- **MC3 — checkpoint delivery.** Copy `models/mountaincar.ppo.ckpt` → `ClientApp/public/mountaincar-net.ckpt` (LFS).
- **MC4 — client director + rewire.** `mountaincar-director.ts`: obs → net → argmax → step (reuse `mountaincar-logic.ts`
  dynamics) → render; rewrite `mountaincar.ts` watch mode (drop `mountaincar-api.ts`). Gate: Playwright — plays,
  0 console errors, 0 `/api/mountaincar` calls.
- **MC5 — retire server path + delete `EpisodeStreamer`.** Delete `MountainCarController`, `MountainCarModelService`
  (+ regs), `mountaincar-api.ts`, `EpisodeStreamer.cs` (now unused), stale `models/mountaincar.ppo.ckpt`; update
  `MountainCarApiTests`. Gate: build + tests green; Playwright watch works.

---

## 6. Risks

| Risk | Game | Mitigation |
|---|---|---|
| Action-mask shield (flood-fill) must be reproduced exactly in the browser | Snake | Port `CurrentActionMask` + `ReachableFreeSpace` to the `.pg`; equivalence-test vs `SnakeEnv`. Integer math → exact. |
| Collection re-expression (LinkedList/HashSet/Queue → List) introduces a bug | Snake | Parity test the ported step + obs + mask against `SnakeEnv` on many played boards. |
| `cos`/`tanh` can't be in a `.pg` → no byte-identity | MountainCar | Option B (recommended) sidesteps it (hand-port, argmax-robust); Option A gates on an accuracy spike. |
| Net perturbation from approximations | MountainCar (Opt A only) | MC0 spike validates argmax parity before committing. |
| Deploy: LFS asset must reach the SPA build | both | Same as M32 (verified): `playground-docker.yml` `lfs:true` + Dockerfile `COPY ClientApp/` before `npm build`; Angular serves `public/**`. |

---

## 7. Non-goals
- 2048 / RushHour / Cube (request/response, no socket).
- Retraining Snake or MountainCar (ship the existing checkpoints; both play well today).
- A shared `nn.pg` (user `.pg` imports unsupported — duplicate per game).

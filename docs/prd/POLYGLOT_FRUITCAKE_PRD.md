# Single-Source FruitCake Physics via MintPlayer.Polyglot (PRD & Plan)

> FruitCake's circle-physics solver is **duplicated by hand** in C# (`FruitCakeWorld`, training + serving) and
> TypeScript (`fruit-cake-physics.ts`, human play), kept in sync only by discipline and comments. `FRUITCAKE_AI_PRD.md`
> §4.8 concluded a single source "isn't cleanly viable via transpilation" — *because no maintained C#↔TS transpiler
> existed.* **That premise is now false:** MintPlayer.Polyglot v0.1.0 ships, and the FruitCake solver is literally its
> north-star conformance sample. This PRD re-opens §4.8 and lays out a phased, low-risk adoption.

- **Status:** Draft v1.0 · 2026-07-04 (3-agent investigation complete; not started)
- **Author:** Pieterjan (with Claude Code)
- **Depends on:** MintPlayer.Polyglot 0.1.0 (`C:\Repos\MintPlayer.Polyglot`; NuGet `MintPlayer.Polyglot.MSBuild`,
  the win-x64 CLI, VS Code ext); the FruitCake physics (`FruitCakeWorld.cs` / `fruit-cake-physics.ts`); companion to
  `FRUITCAKE_AI_PRD.md` §4.8 (the sync contract both solver files cite).

---

## 1. The opportunity

The two solvers are **near-1:1 ports** — identical constants (`PixelsPerMeter 64`, `Gravity 9.8·64`, `Restitution 0.1`,
`Friction 0.3`, `VelocityIterations 12`, `Slop 0.5`, `AngularDamping 0.995`, board 620×850), identical method shapes
(`Step`/`step`, `BuildContacts`, `ResolveVelocity`, `ApplyImpulse`, `CorrectPosition`, `FlushMerges`), identical
catalog radii/points. `FruitCakeWorld.cs:204` even says "ported 1:1 from fruit-cake-physics.ts." Every physics change
must be made **twice, correctly, or the game silently drifts** (training/human physics diverge → the demo stops being
"the same game"). Polyglot exists to delete exactly this class of duplication, and FruitCake is its motivating sample
(`fruitcake_sketch.pg` names this repo) with a conformance gate proving byte-identical C#/TS output.

## 2. What the investigation confirmed (3 agents, 2026-07-04)

**The boundary is clean.** Both solver cores are pure deterministic math with **zero transcendental calls** — only
`+ − × ÷`, `√`, `min`/`max`, and the `π` constant (in `mass = π·r²`, which largely cancels through `invMass` in every
impulse ratio). Rotation (`angle`/`angularVel`/torque/`ω×r`) is **pure arithmetic — no `sin`/`cos`/`atan2`** anywhere in
either core; all trig lives in the render/audio/FX glue. This lands squarely inside Polyglot's byte-identical-safe op set.

**Convertible core vs host glue:**
- **Core (→ `.pg`):** all of `FruitCakeWorld.cs` physics (integration, `BuildContacts`, impulse resolution, position
  correction, merges, `SettleAfterDrop`, `Clone`, danger/eject queries) + `FruitCatalog.cs` (tiers/radii/points/merge
  rule). Mirror = `fruit-cake-physics.ts` `FruitWorld` (~260 lines) + the geometry/scoring half of `fruit-cake-fruits.ts`.
- **Glue (stays native, per platform):** RNG (`Xoshiro256StarStar`), `SaveState`/`RestoreState` (binary I/O), the
  `IEnvironment`/`IStatefulEnvironment` wrapper, reward shaping + `BuildObservation`, the WebSocket/DTO controller — and
  on the TS side the Angular component, Canvas rendering, Web Audio, effects, localStorage, the rAF loop.

**Polyglot v0.1.0 fits the math but imposes packaging constraints** (see §5).

## 3. Output-directory configuration (the open question)

- **TypeScript — configurable.** The CLI `--out <dir>` flag controls it (default: beside the input). Point it at
  `src/RLDemo.Web/ClientApp/src/app/fruit-cake/`. No config-file field; it's per-invocation.
- **C# — *not* configurable, by design.** The `MintPlayer.Polyglot.MSBuild` target hardcodes output to
  `obj/<cfg>/<tfm>/polyglot/*.cs` (`PolyglotOutDir` is set with **no `Condition`**, so a project override is clobbered)
  and injects it straight into `@(Compile)`. The generated `.cs` is an **intermediate compiled into the assembly**
  (the Grpc.Tools pattern) — you don't own the file location, you consume the *types* in-assembly. **Implication:** the
  `.pg` must live in **`MintPlayer.AI.ReinforcementLearning.Environments`** (the assembly that owns the physics), not a
  standalone project.
- `pgconfig.json` configures `root`/`lib`/`targets`/`forbiddenIdentifiers`/`dependencies` — **no output-path key**.

## 4. The precision decision (float32 vs float64)

The twins are **not byte-identical today**: C# uses `float` (float32/`MathF`), TS uses `number` (float64). They stay
*structurally* identical by hand, and the trained net (rotation-off, float32 physics) transfers fine. Two ways to unify:

- **PG-pilot choice — `f32` in the `.pg` (recommended first).** Polyglot emits C# `float` and TS `number` (JS has no
  float32), i.e. it **reproduces today's exact per-target behaviour** — C# float32 (the trained net stays valid, **no
  retrain**), TS float64 (human play unchanged). We gain the single source; we keep the current, working precision split.
  Byte-identity across targets is *not* achieved (JS can't do float32) — but we don't have it today and don't need it
  (the net never touches the TS solver; serving is server-authoritative C#).
- **Optional later upgrade — `f64` everywhere.** Moving the C# solver to `double` would make C#/TS **byte-identical**
  (Polyglot's guarantee, at matched f64 width) — but it changes training physics (float32→double) and requires
  **re-validating/retraining** the shipped net. Deferred; only worth it if exact human/AI parity ever becomes a goal.

## 5. Hard constraints to design around (v0.1.0 maturity, not correctness)

| # | Constraint | Impact here | Mitigation |
|---|---|---|---|
| C1 | **Generated C# types are `internal`** (no public-emission switch yet) | `FruitCakeWorld`/`FruitBody` are `public` and consumed by **RLDemo.Web, Tests, Lab** — 3 assemblies beyond the owner | `.pg` lives in Environments; expose via a thin **public hand-written facade** over the generated internal solver, and/or `InternalsVisibleTo`. Cleanest fix = Polyglot ships **public emission** (its own roadmap, P11). |
| C2 | **No npm/TS build-integration package; CLI is win-x64 only** | CI/deploy is Linux — can't run the CLI there to emit `.ts` | **Commit the generated `.ts`** (regenerate locally via `--watch`); CI consumes it like any source. Unblocks fully when Polyglot ships the **npm sibling + Linux CLI** (P11 remainder). |
| C3 | Byte-identity only for `+ − × ÷ √` at matched width; no `sin/cos/exp/pow` in std | Core already complies (§2); **any future** solver change must not reach for transcendentals | Keep the `.pg` within the safe op set; pin `π` to a literal. |
| C4 | Std surface is small: `List<T>` only, no Dict/Set | Core uses `List<>` + a `(a,b)` tuple + `removeAll` lambda — all supported | None needed; confirmed against the sketch. |
| C5 | Host interop (canvas/DI/render) only via `expect`/`actual` + `extern` | The `.pg` must be **pure solver**; host drives it | Keep the boundary at §2; no host APIs in the `.pg`. |
| C6 | Multi-`.pg` projects need a single import root (Option-prelude dup) | Only one `.pg` planned (the solver) | Non-issue at one file. |
| **C7** | TS emitter emitted a runnable *script*, not an importable ES module (no `export`s) | Blocked PG2 (Angular `import` + `isolatedModules`) | ✅ **RESOLVED in Polyglot 0.1.4** (export emission). Confirmed: the real 0.1.4 TS runs the full adapter consumer AND type-checks clean under Angular-strict. *A full sweep found `export` was the SOLE gap — no further hiccups.* Lesson recorded: verify importability, not just script execution. |

## 6. Goals & Non-Goals

### Goals
- **One source of truth** for the FruitCake solver, eliminating the hand-sync drift risk between training and human play.
- **Dogfood Polyglot** on a real second consumer (beyond its own repo) — the strategic win for the SDK-of-SDKs story.
- Zero regression: the trained net stays valid (PG-pilot uses `f32`), human play unchanged, all FruitCake tests green.

### Non-Goals
- Byte-identical C#/TS physics (the `f64` upgrade in §4 — deferred; not needed under server-authoritative serving).
- Porting host glue (rendering/audio/Angular/env/RNG/serialization) — stays native by design.
- Blocking on Polyglot feature work: PG0 (validation) needs nothing new; PG1/PG2 have working (if imperfect)
  workarounds and improve when Polyglot ships public-emission + the npm/Linux CLI.

## 7. Phased plan

- **PG0 — Validation pilot.** ✅ **PASSED (on Polyglot 0.1.1).** Ported the real solver to `fruitcake_solver.pg`
  (f64 — `f32` proved impractical, §4), generated C# + TS, ran both. **0.1.0 surfaced a blocker** (a codegen precedence
  bug → generated TS went all-NaN; §10) — that was root-caused and **fixed in Polyglot 0.1.1**. On 0.1.1 the pilot is
  clean: **byte-identical C#↔TS and NaN-free** across the 7-drop trace *and* a 28-drop varied cascade with a
  float-state checksum (`bodies=8 scored=63 fsum=39291` on both). So **cross-target byte-identity DOES hold** for this
  solver (my earlier "chaos-fragile" note was the NaN bug, not real chaos — corrected). **PG1/PG2 are unblocked.**
- **PG1 — C# cutover.** ✅ **DONE (2026-07-04).** The hand-written `FruitCakeWorld` physics is replaced by a thin
  **public facade** over the transpiled internal core (`PgFruitCakeWorld`, generated & **committed** — not built at
  runtime, so Linux CI is unaffected). `FruitCakeWorld`/`FruitBody` keep their exact public API (float view over the
  f64 core), so no consumer changed; host-only helpers (danger/eject/PileHeight/Clone/LoadBody) live in the facade,
  and env Save/Restore delegates to `FruitCakeWorld.Write/ReadBodies` at full double precision (keeps bitwise resume).
  Physics moved **float32 → double**. *Gates met:* full solution builds; 17 FruitCake + parity tests green (the parity
  test now checks the facade delegates faithfully to the generated core); **net re-validated on double physics** —
  30-game eval: greedy 895/tier 8.57, net+search 2317/tier 10.33/**watermelon 33%**, all within seed noise of the
  float32 baselines ⇒ **no retrain needed.**
- **PG2 — TS cutover.** ✅ *Unblocked by Polyglot 0.1.4 (export emission, C7) and implemented 2026-07-05.* The generated
  TS core (`fruitcake_solver.ts`, committed in `ClientApp/.../fruit-cake/`) is imported by a thin `FruitWorld` facade
  (`fruit-cake-physics.ts`): same public API (so game/render/component unchanged), `onMerged` exact from
  `core.lastMerges`, `mergeBorn`/age via a side-table keyed on the core body, `onLanded` a host-side approximation,
  rotation on. Adapter + core type-check clean under Angular-strict flags; C# stays on the same `.pg` (0.1.4).
  **✅ Gate met — browser-verified** (host + Playwright, 2026-07-05): drops/stack/settle work; a forced same-tier
  overlap fired `onMerged` with the exact event `{src 2 → res 3, 3 pts}` and the merge product was tagged
  `mergeBorn`/age 0; natural 6-drop play merged (score 10, still playing); rendering correct; **0 console
  errors/warnings**. Only `onLanded` (audio thud) is the documented host-side approximation, not browser-verifiable.
  **The AI stays server-side** (net inference + depth-3 search are C#-only), so the watch-AI WebSocket remains
  regardless of PG2. **PG2 DONE → FruitCake physics is fully single-source (one `.pg` → C# training/serving + TS human-play).**
- **PG3 — byte-identity.** ✅ *Effectively DONE as a side-effect of PG1/PG2.* `f32` proved impractical, so the `.pg`
  was `f64` from the start → generated C# is `double` and TS is `number`/float64 → **C# and TS are byte-identical**
  (verified at PG0: 28-drop cascade + float checksum identical). No separate upgrade needed.
- **PG4 — ❌ SUPERSEDED by M32 (`FRUITCAKE_CLIENT_SIDE_AI_PRD.md`).** The premise below ("the AI can't run in the
  browser") was wrong — the 4-agent analysis (2026-07-05) showed the net (mean weights, noise off at inference) +
  the RNG-free depth-3 search are fully portable to TS. **M32 runs the *whole* AI client-side**, which removes the
  server compute, the bandwidth, **and** the WebSocket entirely — strictly better than the column-only-streaming idea
  described here. Kept for the record:
- **PG4 — (Follow-up, NOT built) watch-AI bandwidth optimization — *enabled by PG3's byte-identity*.** Today the
  watch-AI WebSocket streams **every intra-drop frame** (all fruit positions) — unchanged by this PR; server load +
  bandwidth are as before. Because the browser now runs **byte-identical** physics to the C# server, watch-AI *could*
  instead stream only the AI's **chosen column** per drop (`{column, currentTier, nextTier}`, a few bytes) and let the
  Angular watch-mode run the shared solver locally to animate + settle — cutting the per-frame streaming bandwidth.
  **Keeps the WebSocket** (the server still computes the move). **Does NOT reduce the biggest *server compute*** — the
  depth-3 forward search + net inference stay C#-only (the AI can't run in the browser). Scope: change
  `FruitCakeController`'s streamer + the Angular watch-mode consumer. Worth doing only if watch-AI bandwidth matters.

**Recommended first session:** PG0 only — it's the honest, reversible proof that the single source reproduces the twin,
and it surfaces any real friction before touching production paths.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Internal-by-default breaks cross-assembly serving (C1) | Facade/`InternalsVisibleTo` for the pilot; push Polyglot public-emission for the clean fix. |
| Linux CI can't run the win-x64 CLI (C2) | Commit generated `.ts`; regenerate locally. Revisit when the npm/Linux CLI ships. |
| Generated solver subtly ≠ hand twin | PG0 differential test vs the existing twin as oracle — cutover (PG1/PG2) is gated on it. |
| float32 vs float64 confusion | PG-pilot pins `f32` = today's behaviour, net stays valid; `f64`/byte-identity is an explicit, separate PG3. |
| Scope creep into host glue | Boundary fixed at §2; `.pg` is pure solver, driven by native host code. |
| Polyglot v0.1.0 churn | Pin the exact NuGet/CLI version; the FruitCake sketch is conformance-gated in the Polyglot repo, so regressions there are caught upstream. |

## 10. PG0 findings (2026-07-04) — empirical

Ran the pilot end-to-end with the bundled CLI (`…/mintplayer.polyglot.msbuild/0.1.0/tools/win-x64/polyglot.exe`,
no install needed). Artifacts + minimal repro: **`polyglot-pilot/`** (`fruitcake_solver.pg`, `REPRO.md`).

- **Porting works.** The real solver (`FruitCakeWorld` + `FruitCatalog`, 1-based 11 tiers) ports to a clean,
  **type-checking** `.pg`; both C# and TS emit and run.
- **Precision:** `f32` is impractical (explicit `(f32)` cast required on *every* float literal) → use **`f64`** (as
  the sketch does). Consequence stands: generated C# is `double`, so adoption moves the C# solver float32→double
  (PG1 net A/B covers it).
- **🚩 Blocker — generated TS goes NaN.** With the same type-checked `.pg`, the **generated C# runs correct physics**
  (`nan=0` every drop) but the **generated TS has NaN in every body from drop 1** (`nan == body count`). Integer
  outputs coincidentally match for 6 scripted drops then diverge at drop 7. The upstream 6-tier sketch does **not**
  trigger this, so the real catalog/params/paths expose a **TS codegen (or §3.D-class numerical) divergence** that
  must be fixed **in the Polyglot repo**. ⇒ **cross-target byte-identity is not a usable property here** (it was never
  required — §4.8 — but the TS output being *wrong* is a hard blocker for PG2).
- **Minor rough edges:** the CLI doesn't create `--out` dir; `i64` return lowers to TS `BigInt(Math.trunc(x))` which
  throws on NaN/Inf; generated C# types are `internal` (C1).

**Revised recommendation:** PG0's value (single source, one solver) is real and the C# side is sound — but **PG2 (TS
cutover) is blocked** on root-causing the TS-NaN divergence in Polyglot. Because that fix belongs in the Polyglot repo
(and per the owner's note, cross-repo build work shouldn't be driven from the other side), the honest next step is a
**Polyglot-side bug fix using `polyglot-pilot/` as the minimal repro**, then resume at PG1/PG2. Do *not* cut over
FruitCake to the generated solver until the TS path is clean.

### ✅ RESOLVED in Polyglot 0.1.1 (2026-07-04)
The precedence bug is fixed: `a.invMass + (b?.invMass ?? 0.0)` now emits **with** parentheses in both targets, the
minimal repro prints `null=5` on both, and the full solver is **byte-identical and NaN-free** (7-drop trace + 28-drop
varied cascade + float checksum). **Byte-identity holds** — so the earlier "not for this solver / chaos-fragile" caveat
is retracted. **PG1 (C# cutover) and PG2 (TS cutover) are unblocked.** Remaining constraints for the cutover are the
packaging ones in §5 (C1 internal-by-default C#, C2 no npm/Linux CLI) plus the §4 precision note (C# solver moves
float32→double, gated by the PG1 net A/B).

## 9. Sources
- Internal (this repo): `FRUITCAKE_AI_PRD.md` §4.8 (the now-reopened "single source" question), `FruitCakeWorld.cs`,
  `fruit-cake-physics.ts`, `fruit-cake-fruits.ts`, the FruitCake test/A-B tooling.
- MintPlayer.Polyglot (`C:\Repos\MintPlayer.Polyglot`): `SPEC.md` §3.A/§3.D, `POLYGLOT_PRD.md`, `PLAN.md`/`CLAUDE.md`
  (internal-types + npm-sibling limits), `MintPlayer.Polyglot.MSBuild.targets` (fixed `PolyglotOutDir`), the CLI
  `--out`, `docs/lang/samples/fruitcake_sketch.pg` + `tests/conformance/programs/fruitcake.pg` (the conformance gate).

# Polyglot bug handoff — incremental re-transpile emits a duplicate prelude / non-`partial` PolyglotProgram

**Found:** 2026-07-10, during Snake M34 (adding look-ahead search to `snake_solver.pg`).
**Polyglot:** `MintPlayer.Polyglot.MSBuild` **0.3.1** — **still present in 0.5.3** (verified 2026-07-12; see the update at
the bottom). This is an **MSBuild `.targets` bug, not a CLI/version bug**, so bumping the package does not fix it.
**Polyglot source:** `C:\Repos\MintPlayer.Polyglot` (fix lives in `build/MintPlayer.Polyglot.MSBuild.targets`).
**Severity:** breaks *incremental* dev builds after editing a `.pg` in a **multi-`.pg` project**; a **clean build always
succeeds**, so CI / fresh clones are unaffected — but it's confusing and easy to misdiagnose.

> **⚠️ 2026-07-12 update at the bottom of this file** sharpens the root cause (MSBuild *partial-incremental build*
> passes a **subset** of `.pg` to the CLI) and gives a **verified** `.targets` fix. Note the original "clean the `--out`
> dir" idea in *Fix ideas* below is **insufficient and can be harmful** on its own — read the update first.

## Symptom

In a project with **2+ `.pg` files** built by the glob-all MSBuild target (here: `fruitcake_solver.pg`,
`mountaincar_solver.pg`, `snake_solver.pg`), after **editing one `.pg` and rebuilding**, the C# transpile of one solver
unit is intermittently emitted in **"standalone" mode** — it inlines the `Option`/`Some`/`None` prelude (which also lives
in the shared `__polyglot_prelude.cs`) and declares a **non-`partial`** `static class PolyglotProgram` (the other units
emit `static partial class PolyglotProgram` and reference the shared prelude). The C# compile then fails:

```
snake_solver.cs(NNN): error CS0260: Missing partial modifier on declaration of type 'PolyglotProgram';
                      another partial declaration of this type exists
__polyglot_prelude.cs(3): error CS0101: '<global namespace>' already contains a definition for 'Option'  (Some, None)
__polyglot_prelude.cs(4): error CS8863: Only a single partial type declaration may have a parameter list
```

## Root cause (what I actually pinned down)

It is **not** about the source `.pg` content. I initially suspected top-level `record` declarations, but ruled that out:
with the records removed, an *incremental* rebuild still failed, while *clean* builds pass. The differentiator is
**clean vs incremental**, driven by **stale files in the output directory**:

- The MSBuild target (`build/MintPlayer.Polyglot.MSBuild.targets`, target `PolyglotTranspile`) transpiles the **whole
  `.pg` set in ONE CLI invocation** into `$(IntermediateOutputPath)polyglot\`, and the CLI **does not clean that
  directory first**. A 2+-`.pg` project also emits a source-less `__polyglot_prelude.cs` there.
- On re-transpile, a **pre-existing `__polyglot_prelude.cs`** (from the previous build) is present in `--out`, and the
  CLI emits one solver unit in standalone mode + re-emits the prelude → the duplicate/`partial` collision above.
- The target's incremental gate (`Inputs=@(PolyglotFile);$(PolyglotTool)`,
  `Outputs=…%(Filename).cs`) makes it **worse**: once a bad `snake_solver.cs` is written, its mtime is newer than the
  `.pg`, so the next build **skips the transpile and recompiles the bad file** — the failure sticks until the generated
  `.cs` is deleted.

### Reproduce
1. Clean-build a 2+-`.pg` project → succeeds; `obj/<cfg>/net10.0/polyglot/` has `__polyglot_prelude.cs` + one `.cs` per
   `.pg`, all `static partial class PolyglotProgram`.
2. Edit any one `.pg`; rebuild → intermittently one unit comes out standalone → `CS0260`/`CS0101`/`CS8863`.
3. `rm obj/**/polyglot/*.cs` and rebuild → succeeds again. (Verified: 3/3 fresh builds correct & deterministic.)

## Fix ideas (for the Polyglot maintainer)

- **Clean the `--out` directory** (or at least overwrite/rewrite `__polyglot_prelude.cs` deterministically) at the start
  of each `build` invocation, so a stale prelude can't flip a unit to standalone.
- Make the standalone-vs-library decision **deterministic and project-scoped**: in a multi-file invocation, *never* emit a
  unit standalone — always shared prelude + all-`partial` `PolyglotProgram`.
- MSBuild side: add `__polyglot_prelude.cs` to the target's `Outputs`, or make `PolyglotTranspile` delete prior generated
  `.cs` before running, so the incremental gate can't cache a bad transpile.

## Impact on this repo

None at ship time — clean/CI builds are correct. During iterative `.pg` editing, if you hit the errors above:
`rm src/MintPlayer.AI.ReinforcementLearning.Environments/obj/*/net10.0/polyglot/*.cs` then rebuild.

Note: the M34 search *does* use a top-level `record PgSnakeBeamNode` (a typed frontier element — required to avoid a
separate TS `any`-inference failure; see `SNAKE_SEARCH_PRD.md` §9). Records build fine on a clean build (FruitCake uses
them too); this incremental-staleness bug is unrelated to records. If you hit the errors above during iterative `.pg`
editing: `rm src/MintPlayer.AI.ReinforcementLearning.Environments/obj/*/net10.0/polyglot/*.cs` then rebuild.

---

## UPDATE 2026-07-12 — precise root cause + a verified fix (bug persists through 0.5.3)

Re-encountered during **Chess M40.1** (adding a 4th solver, `chess_solver.pg`). Bumping the package **0.3.1 → 0.5.3**
did **not** fix it — an incremental touch of one `.pg` still produces `CS0101`/`CS0260`/`CS8863`. That rules out the
CLI: **the bug is in the MSBuild `.targets`, and it is version-independent.**

### The actual trigger (sharper than "stale files in the out dir")

`PolyglotTranspile` declares a **per-file** output map:

```xml
Inputs="@(PolyglotFile);$(PolyglotTool)"
Outputs="@(PolyglotFile->'$(PolyglotOutDir)%(Filename).cs')"
```

Because both `Inputs` and `Outputs` are item transforms of the same `@(PolyglotFile)`, MSBuild does a **partial
incremental build**: when only *one* `.pg` is newer than its `.cs`, MSBuild runs the target with `@(PolyglotFile)`
**filtered to just that one file**. The `Exec` then invokes the CLI with a **single** source:

```
polyglot build "…/chess_solver.pg" --target csharp --out "…/polyglot/."
```

From the CLI's point of view that is a **single-module** build, so it (correctly, for a lone module) emits
`chess_solver.cs` in **standalone** mode — inline `Option`/`Some`/`None` + a **non-`partial`** `PolyglotProgram`. That
standalone unit then collides with the **other** solvers' `.cs` + the shared `__polyglot_prelude.cs` still sitting in
`obj/` from the prior full build → `CS0260`/`CS0101`/`CS8863`. The stale prelude is a *symptom*, not the cause; the
cause is **MSBuild handing the CLI a subset**. (`rm *.cs` "fixes" it only because it makes *all* outputs missing, which
forces MSBuild to re-run with the **full** set.)

### Why the earlier "clean the `--out` dir" idea is not enough — and is harmful alone

If the CLI cleaned `--out` at the start of every build but MSBuild still invoked it with a **subset**, then a
single-file incremental build would **delete the other solvers' generated `.cs`** and emit only the changed one →
"type `PgFruitCakeWorld` not found" etc. The out-dir cleaning must be paired with **always invoking the CLI with the
full set**. The real fix is therefore MSBuild-side.

### Verified fix (MSBuild `.targets`) — force all-or-nothing with a single stamp output

Replace `PolyglotTranspile`'s per-file `Outputs` with a **single stamp file** (a static path, *not* an item transform),
and clean the out dir so a removed/renamed `.pg` leaves no orphan `.cs`. A single static `Outputs` makes the target
**un-batchable**, so any staleness re-runs it with the **complete** `@(PolyglotFile)`:

```xml
<Target Name="PolyglotTranspile"
        BeforeTargets="CoreCompile"
        DependsOnTargets="_PolyglotVerifyTool"
        Condition="'@(PolyglotFile)' != ''"
        Inputs="@(PolyglotFile);$(PolyglotTool)"
        Outputs="$(PolyglotOutDir)__polyglot.stamp">   <!-- was: @(PolyglotFile->'…%(Filename).cs') -->
  <PropertyGroup> … (unchanged _PolyglotLibArg / _PolyglotAccessArg) … </PropertyGroup>
  <RemoveDir Directories="$(PolyglotOutDir)" />          <!-- no orphan .cs from a deleted .pg -->
  <MakeDir Directories="$(PolyglotOutDir)" />
  <Exec Command="… build @(PolyglotFile->'&quot;%(FullPath)&quot;', ' ') --target csharp --out &quot;$(PolyglotOutDir).&quot; …" />
  <Touch Files="$(PolyglotOutDir)__polyglot.stamp" AlwaysCreate="true" />
</Target>
```

`_PolyglotAddGenerated` already globs `$(PolyglotOutDir)*.cs`, so the `.stamp` is not compiled; add it to `FileWrites`
too so `dotnet clean` removes it. A no-op rebuild stays up-to-date (stamp newer than every `.pg` and the tool); any
`.pg`/tool change re-runs the full set. **This is the recommended upstream change.** A defensive
`--clean-out`/idempotent-prelude on the CLI is a fine belt-and-braces addition but is not sufficient on its own (see
above).

### Consumer-side mitigation already shipped (so this repo no longer needs the `rm`)

Until the `.targets` ships the fix, `MintPlayer.AI.ReinforcementLearning.Environments.csproj` carries a
`_PolyglotForceFullRetranspile` target that reproduces the effect without editing the package: it wipes
`$(IntermediateOutputPath)polyglot\` whenever any `.pg` is newer than the shared `__polyglot_prelude.cs`, forcing
`PolyglotTranspile` to regenerate the whole set in one CLI call. Verified: the incremental touch that reliably broke now
rebuilds clean in Release **and** Debug; no-op rebuilds stay up-to-date; full test suite 362/362, chess perft 25/25.
When the upstream `.targets` fix lands, that local target can be deleted.

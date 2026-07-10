# Polyglot bug handoff — incremental re-transpile emits a duplicate prelude / non-`partial` PolyglotProgram

**Found:** 2026-07-10, during Snake M34 (adding look-ahead search to `snake_solver.pg`).
**Polyglot:** `MintPlayer.Polyglot.MSBuild` **0.3.1** (transpiler `tools/win-x64/polyglot.exe` 0.3.1).
**Polyglot source:** `C:\Repos\MintPlayer.Polyglot`
**Severity:** breaks *incremental* dev builds after editing a `.pg` in a **multi-`.pg` project**; a **clean build always
succeeds**, so CI / fresh clones are unaffected — but it's confusing and easy to misdiagnose.

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

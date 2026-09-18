# PRD — Coverage 60% → 90%, and teaching coverage to speak Polyglot (M63)

*2026-09-17 · branch `m63-coverage-90` · see `PLAN.md` M63 for milestone status*

Successor to [`COVERAGE_PRD.md`](COVERAGE_PRD.md) (M56), which built the collection + upload
pipeline and deliberately left two things on the table: the `.pg` sources measure as nothing, and
there is no frontend coverage at all. This PRD picks up both, and raises the number.

## 1. Goal

Three outcomes, in priority order:

1. **The 9 `.pg` solvers become measurable**, with each `.pg` line carrying the *union* of hits from
   its generated C# and its generated TypeScript.
2. **Repo-wide line coverage reaches 90%** on the figure the badge renders.
3. **The denominator becomes honest** — what it measures is the shipped library surface, and every
   exclusion is a written decision rather than an accident of which project the test assembly
   happens to reference.

Non-goals are in §9.

## 2. Where the 60% actually comes from

Investigated 2026-09-17 by four parallel agents (baseline audit, Polyglot compiler feasibility,
coverlet mechanics, tooling survey). Measured facts:

| Project | files | lines | In the report today? |
|---|---|---|---|
| `src/…Core` | 72 | 7,447 | yes — **best covered** |
| `src/…Environments` | 62 | 9,292 | yes (hand-written C# only) |
| `src/…Campaigns` | 35 | 5,580 | yes — **weak**, and its tests are mostly `Category=Slow` |
| `src/…Ilgpu` | 7 | 2,128 | **no** — assembly `Exclude`, must stay |
| `src/…Ilgpu.Hosting` | 1 | 23 | yes |
| `src/…Hosting` | 1 | 42 | yes |
| `src/RLDemo.Web` | 19 | 1,356 | yes — partial |
| `src/RLDemo.Console` | 1 | 631 | **no** — never referenced by the test project |
| `tools/…Lab` | 27 | 3,352 | yes — **near-zero covered**, only `CliArgs` is tested |
| `tools/…Bench` | 1 | 205 | no — not referenced |
| **generated C# from `.pg`** | 10 | **8,466** (per config) | **no** — `ExcludeByFile` drops all of `obj/` |
| **`.pg` sources** | 9 | **6,828** | **no** — nothing measures them |
| `ClientApp` TypeScript | 73 | — | **no** — zero tests exist |

Test suite: 95 test files, 622 `[Fact]` + 30 `[Theory]` × 121 `[InlineData]`. **27 tests carry
`[Trait("Category","Slow")]` across 16 files and are filtered out of every CI run**
(`--filter "Category!=Slow"`), so the badge is explicitly fast-bucket coverage.

**The four levers, ranked by uncovered lines in the denominator:**

1. `tools/Lab` (~3.2k lines) — instrumented only because the test project references it for
   `CliArgs`. Pure denominator padding.
2. `Campaigns` (~5.6k) — long training loops; the tests that *do* exercise them are `Category=Slow`
   and therefore never counted.
3. The Kociemba block in Environments (~1.7k: `K_CubieCube` 838, `K_CoordCubeBuildTables` 461,
   `K_SearchRunTime` 438) — largely untested.
4. `RLDemo.Web` Services (`CubeModelService` 158, `ModelServiceInfrastructure` 131).

**The counter-intuitive one:** bringing the `.pg` solvers *in* should **raise** the percentage, not
dilute it. Those 8,466 generated lines are among the most heavily exercised code in the repo — the
parity suites and perft gates hammer them — and today they contribute to neither numerator nor
denominator.

## 3. The key finding: `#line` makes the merge script unnecessary (C# side)

The original idea in the request was a line-mapping sidecar plus a post-processing tool that
collapses generated-file coverage onto `.pg` lines. **For the C# half that tool is not needed**,
because .NET already has the mechanism and coverlet already honours it.

**Verified empirically** (throwaway xunit + class-library probe, coverlet.collector 10.0.1,
scratchpad only — the repo suite was never run). Given generated C# containing
`#line 100 "C:\pgsrc\polyglot\tetris_solver.pg"`, the emitted cobertura contained:

```xml
<sources><source>C:/</source></sources>
<class name="PgProbe.Solver" filename="pgsrc/polyglot/tetris_solver.pg" line-rate="0.3846">
  <lines>
    <line number="101" hits="1" />
    <line number="103" hits="1" branch="True" condition-coverage="50% (1/2)" />
    <line number="104" hits="0" />
```

Four consequences, all observed rather than reasoned:

- **The reported filename is the `.pg` path**, and **line numbers are remapped into `.pg`
  numbering** (`#line 100` before the method produced hits at 101–108). Roslyn writes `#line` into
  the PDB sequence points; coverlet reads them verbatim.
- **`ExcludeByFile` matches the PDB-recorded path, not the physical file.** A second run with
  `<ExcludeByFile>**/lib/**/*.cs</ExcludeByFile>` left the remapped class *intact* and dropped only
  the non-remapped methods. So the repo's existing `**/obj/**/*.cs` rule **stops applying by
  itself** once the generated C# claims a `.pg` origin — no runsettings change is required.
- **One physical `.cs` splits into multiple `<class>` entries keyed by filename**, so partial
  remapping (mapped bodies + unmapped scaffolding) is well-defined.
- `.pg` files **are** in `git ls-files` (verified), which is exactly what coverage.mintplayer.com
  needs to resolve a path.

**Gotcha for the spikes:** the first probe attempt put the code in the test assembly and produced
`lines-valid="0"` — coverlet's `IncludeTestAssembly` defaults to false. Code under test must live in
a separate assembly.

## 4. The union comes free from the existing server

`COVERAGE_PRD.md` §2 records that the MintPlayer coverage service **merges multiple reports under
one `(repo, sha, runId, runAttempt)` with max semantics**. That is precisely the per-line union the
request asks for. So the architecture is:

```
dotnet test ──► coverage.cobertura.xml ──┐   (C# hits, .pg-attributed via #line)
                                          ├──► upload (2 files) ──► server max-merge ──► badge
vitest      ──► lcov.info / cobertura ───┘   (TS hits, .pg-attributed via v3 source map)
```

No `pg-union.xml`, no bespoke unioner, no ReportGenerator in the hot path. Two reports, both already
speaking `.pg` line numbers, merged by infrastructure that exists.

**Semantic caveat, stated deliberately:** union means *"this logic is tested somewhere"*, not
*"tested in both targets"*. That is defensible **here specifically** because the repo's parity
suites (`tools/cf_parity.mjs`, `tetris_parity.mjs`, and the C#-side parity tests) assert the two
emissions agree bitwise on identical inputs. A line proven correct in C# is therefore meaningfully
covered for TS. **If the parity gates are ever dropped, this union stops being honest** — that
dependency is recorded here so it is not rediscovered the hard way.

## 5. What Polyglot has to grow

Polyglot (`C:\Repos\MintPlayer.Polyglot`, v0.9.9) is **C++, 17,946 lines across 51 files**. Backends
are *data* — declarative JSON plugin manifests (`plugins/{csharp,typescript}/polyglot-plugin.json`)
interpreted by one class. Modifying the compiler is sanctioned (owner, 2026-08-26).

Both prerequisites for source mapping **already hold** — this is what makes the work moderate rather
than a rewrite:

- **Positions survive to the emitter.** `SourcePos {line, col, fileId}` is a base-class field on
  both `ir::Expr` (`ir.hpp:27-33`) and `ir::Stmt` (`ir.hpp:277-282`), set through every constructor;
  `lower.cpp` propagates it at **72 sites with zero default-constructed `SourcePos{}`**. *(Verified
  directly in this session — one investigating agent reported the opposite, that no IR node carries
  a location. That report is wrong; it would have made this a whole-IR rewrite.)*
- **There is exactly one line-writing chokepoint.** `EmitterBase::line()`
  (`emitter_base.cpp:1598`), 27 call sites, all in that file. Every other line producer
  (`openBlock`, `closeBlock`, `blockBody`, `headBlock`, `runDeclRule`) routes through it.
  `curPos_` can be set in `EmitterBase::emitStmt` (`emitter_base.cpp:1683`).

Work items, with the four real obstacles called out:

| Item | Where | Size |
|---|---|---|
| Set `curPos_` in `emitStmt`; emit a directive from `line()` **on every output line** — `#line N` when a position is known, `#line hidden` when not (see S1: emitting only on *change* causes drift that silently marks innocent `.pg` lines covered) | `emitter_base.cpp:1598,1683` | small |
| **Obstacle 1** — thread a real `SourceMap` through `compile()` (today passes `nullptr`, so every build-path token is stamped `fileId = 0`; `#line` needs real filenames) | `compiler.cpp:629,353` | small |
| **Obstacle 2** — `inlineBlock` (`emitter_base.cpp:1643`) flattens `\n` → space in a scratch buffer, so directives emitted inside it would be **spliced into the middle of a code line and break compilation** — suppression there is mandatory, not cosmetic. `line()` is also sometimes handed strings containing `\n`. *(The prelude-prepend shift is no longer an obstacle: per-line absolute directives have nothing to shift — see S1.)* | `emitter_base.cpp:1594,1643` | small |
| **Obstacle 3** — scaffolding with no `.pg` origin (prelude, `partial class PolyglotProgram`, `__polyglot_prelude.cs`) needs `#line hidden` / `#line default` or it is misattributed | `emitter_base.cpp` | small |
| Add `SourcePos` to the 8 IR decl structs + populate from AST `pos`/`namePos` (maps *signature* lines; bodies already map) | `ir.hpp:459+`, `lower.cpp` | medium, optional |
| Widen `Backend::emit` / `EmitResult` / `ModuleFile` to carry mappings (TS only) | `backend.hpp`, `polyglot.hpp:61-77` | small |
| Base64-VLQ v3 source-map writer + sibling `.map` write (TS only; ~80 new lines, repo has its own JSON writer, no third-party dep) | new; `Cli/src/main.cpp:167-241` | medium |
| `--origin-info` CLI flag / `pgconfig.json` `"originInfo"` / MSBuild `PolyglotOriginInfo` + `.targets` `<Exec>` — **shipped in [Polyglot#70](https://github.com/MintPlayer/MintPlayer.Polyglot/pull/70)**; one flag drives both the C# `#line` sink and the TS source map | `main.cpp`, `.targets` | small |
| **Obstacle 4 — the biggest cost item** — the conformance suite compares emitted output **byte-for-byte**, so inserting `#line` churns a large fixture set. Must be flag-gated **off by default**. | `tests/**` | **medium–large** |

Shipping: land in Polyglot → tag `v0.10.0` (release workflow auto-pushes to nuget.org) → bump
`MintPlayer.Polyglot.MSBuild` from `0.9.9` in the Environments `.csproj`. For iteration before
publishing, the `PolyglotTool` MSBuild property can point at a locally built CLI.

## 6. Spikes

Each spike is cheap, answers one question that would otherwise be discovered late, and has a
written pass criterion. **Run targeted slices only — never the full suite.**

### S1 — Many-to-one collapse ✅ **RUN 2026-09-17 — PASSED, and changed the design**

`#line` maps ~8,466 generated C# lines onto 6,828 `.pg` lines, so **multiple sequence points land on
the same `(document, line)`**. Unknown: does coverlet emit duplicate `<line number>` elements, sum
them, take the max, or emit last-writer-wins?

**Results** (scratchpad probe, coverlet.collector 10.0.1, `lib/Collapse.cs` + `lib/Drift.cs`):

| Case | Shape | Observed |
|---|---|---|
| A | 4 sequence points on `.pg` line 200, three executed, one not | `<line number="200" hits="3"/>` — **one element; covered if *any* contributor ran** |
| B | 3 sequence points on line 300, none executed | `<line number="300" hits="0"/>` |
| C | loop body, 2 statements × 5 iterations on line 400 | `hits="10"` — **hits SUM, they do not max** |
| D | two *different methods* sharing line 500, one called | `hits="1"` — merged into one element |

So the core semantics are exactly what we need: one `<line>` per number, no duplicates, and
**"covered if any contributing generated line was hit"** comes for free. No normalisation tool.

**But the probe surfaced a defect the naive design would have shipped.** Case E emitted `#line 600`
before a 4-line `if` block and left the braces to drift:

```
line 600 hits 1   ← real
line 601 hits 1   ← PHANTOM (the `{`)
line 602 hits 1   ← PHANTOM (the body)
line 603 hits 1   ← PHANTOM (the `}`)
line 610 hits 1   ← real
```

Lines 601–603 are *different, innocent source lines* in a real `.pg`, silently marked covered —
**misattribution that inflates the number**. Case F wrapped the scaffolding in `#line hidden` and
reported **only 700 and 710**. `#line hidden` suppresses phantoms completely.

**Design consequence (folded into §5):** the emitter must emit a directive for **every** output
line — `#line N` where a source position is known, `#line hidden` where it is not (braces,
scaffolding, prelude). Emitting only on *change* is not merely less precise, it is wrong. A welcome
side effect: because every line carries an absolute directive, the prelude-prepend shift problem
(§5 obstacle 2) disappears entirely — there is nothing left to shift.

### S2 — `#line` end-to-end on one real solver

Build Polyglot locally with `#line` behind a flag, transpile **`mountaincar_solver.pg` only** (152
lines, the smallest), point `PolyglotTool` at the local CLI, run a targeted parity slice with
coverage. **Pass:** `coverage.cobertura.xml` contains a class whose `filename` ends
`MountainCar/polyglot/mountaincar_solver.pg`, the C# still compiles, and the existing parity test
still passes. **Also record:** whether compiler *errors* in generated code now report against the
`.pg` (expected, arguably desirable) and whether `#line hidden` correctly suppresses the prelude.

### S3 — Does the number actually go up?

With S2 green, transpile all 9 solvers and run the fast bucket locally. **Pass:** total line rate
does not *drop*. This is the falsifiable version of §2's claim that the `.pg` cores are well tested.
If it drops, the `.pg` bodies are less exercised than believed and §8's milestone ordering changes.

### S4 — Vitest from zero on one spec

Add `vitest` + `@vitest/coverage-v8`, a `test` target in `angular.json` via
`@angular/build:unit-test`, and **one** trivial spec against a hand-written TS file. **Pass:**
`lcov.info` with `DA:<line>,<hits>` records appears. This de-risks the Angular 22 builder wiring
before any `.pg` mapping is involved. (`tsconfig.spec.json` already declares
`"types": ["vitest/globals"]` — dead scaffolding today, vitest is not installed.)

### S5 — Chained remap `.pg → .ts → .js`

The weak link. `@vitest/coverage-v8` collects on the built bundle and remaps to `.ts` via the
bundle's map; getting to `.pg` needs that composed with the Polyglot map. **Pass:** a spec
exercising the generated `mountaincar_solver.ts` produces lcov keyed on
`…/polyglot/mountaincar_solver.pg`. **If this fails**, fall back to `@vitest/coverage-istanbul` with
an explicit `inputSourceMap`; if *that* fails, the TS half ships as plain `.ts` coverage and the
`.pg` union is C#-only — §4 still works, it just has one input. **S5 failing must not block M63.**

### S6 — Server acceptance of a `.pg`-keyed report

Before wiring CI, confirm coverage.mintplayer.com resolves a `.pg` path through its `git ls-files`
suffix matching and merges two reports naming the same `.pg`. **Pass:** the build reaches `Complete`
and the `.pg` file is browsable server-side with per-line gutters. Cheapest as a throwaway branch
push. **Fail:** fall back to local ReportGenerator merge (§7) and upload one pre-merged cobertura.

## 7. Fallback if the server route fails (S6 red)

ReportGenerator 5.5.x is **not** in this repo today, but it is the escape hatch:

- `CoverageReportParser.CreateProducer` sniffs each input independently, so
  `-reports:"coverage/**/*.cobertura.xml;ClientApp/coverage/lcov.info"` merges mixed formats in one
  run.
- `CodeFile.Merge` **sums hits per line and takes `Math.Max` of line visit status, keyed on file
  path** — the same union semantics, computed locally.
- It is extension-agnostic (`LocalFileReader.LoadFile` is just `File.Exists` + read-as-text), so
  `.pg` renders with a full hit gutter, no syntax highlighting.
- `-reporttypes:Cobertura;MarkdownSummaryGithub;HtmlInline;Badges`.

Only adopt this if S6 is red — it adds a tool and a CI step that the server route makes redundant.

## 8. Milestones

- **M63.1 — Spikes S1 ✅ + S4.** The two independent risk probes (collapse semantics; vitest wiring).
  Both are scratchpad/local; neither touches the Polyglot compiler. Gate: both pass criteria
  recorded in this file, including negative results.
- **M63.2 — Polyglot `#line` (flag-gated).** §5 items 1–4 + the CLI flag, conformance fixtures kept
  byte-identical with the flag off. Spike S2. Gate: S2 pass + Polyglot's own suite green.
- **M63.3 — Adopt in MintPlayer.AI.** Tag `v0.10.0`, bump the `PackageReference`, enable the flag
  via `pgconfig.json`/`.targets`. Spikes S3 + S6. Gate: fast-bucket total does not drop; a `.pg`
  file is browsable on the service.
- **M63.4 — The denominator decisions.** Implement whatever §10 resolves (Lab, Console, Slow
  bucket). Gate: every exclusion has a one-line written rationale in `coverlet.runsettings`.
- **M63.5 — C# coverage push to 90%.** Tests against the ranked gaps from §2: Campaigns, Kociemba,
  RLDemo.Web Services. Gate: fast-bucket line rate ≥ 90%.
- **M63.6 — Frontend tests + TS `.pg` mapping.** Vitest suite, CI step, source-map emission, spike
  S5. Gate: `lcov.info` uploads and merges; S5 result recorded either way.
- **M63.7 — README/badge note + PLAN.md entry.**

Per the repo's batching rule, the **full test suite runs once** at the end of M63.5, not per
milestone; intermediate milestones verify by targeted slice + type-check.

## 9. Out of scope / genuinely not being done

- **Column-level / branch coverage on `.pg`.** Would mean threading positions through the expression
  rule interpreter, which returns bare `std::string` — a materially bigger compiler job. Line
  granularity is enough for coverage.
- **Coverage for `Category=Slow`.** Multi-minute training and perft gates; running them per-PR would
  swamp CI. (Whether *some* of them should be reclassified is §10, not this.)
- **Codecov / Coveralls migration.** Both would work (`.pg` is in git, so path resolution succeeds;
  a non-standard extension costs only the language-aware viewer), but the existing service already
  does multi-report merging. No new vendor.
- **Instrumenting `src/…Ilgpu`.** ILGPU JIT-compiles kernels from their IL at runtime and coverlet's
  injected `RecordHit` calls make that throw `ILGPU.InternalCompilerException` — this cost 37 CI
  failures once already. The assembly `Exclude` stays.
- **A debugger/editor story for `#line`.** Stepping will land in `.pg`, which no editor renders with
  C# semantics. Noted, not solved here.

## 10. Open decisions for the owner

These change the shape of the work and are **not** being decided unilaterally:

1. **`tools/Lab` (~3.2k lines, near-zero covered).** It is in the denominator only because the test
   project references it to test `CliArgs`. Three options: (a) exclude `tools/**` as dev tooling and
   state that coverage measures the shipped library surface; (b) genuinely test it; (c) leave it and
   absorb the drag. **Recommendation: (a)** — it is the single largest lever and the most defensible
   scoping statement, but it *is* a metric-definition change and should be the owner's call, made
   explicitly rather than absorbed into a refactor.
2. **The `Category=Slow` filter.** Several Campaigns tests exist but never count. Introduce a
   `Category=Medium` bucket for ones that run in seconds rather than minutes? That converts existing
   tests into coverage at near-zero cost — likely the cheapest single step toward 90%.
3. **`src/RLDemo.Console` (631 lines)** is invisible because nothing references it. Bring it in
   (honest, lowers the number short-term) or declare it out of scope alongside `tools/**`?
4. **Is 90% the right target *after* the denominator is fixed?** If §10.1 and §10.3 both resolve to
   "exclude", 90% of a smaller, shipped-code-only denominator is a meaningfully stricter bar than
   90% of today's. Worth re-baselining after M63.4 rather than steering by a number set before the
   denominator was known.

## 11. Corrections to existing docs landing in this PR

- `COVERAGE_PRD.md` §3.1 states the `.pg` sources "stay effectively measured through nothing —
  accepted: coverage speaks C#/TS file language". M63 supersedes that; the paragraph needs a pointer
  here rather than deletion (the reasoning was correct for M56's scope).
- `COVERAGE_PRD.md` §6 lists "Angular test coverage — there is no `ng test` in CI to instrument" as
  out of scope. Superseded by M63.6.
- The `ExcludeByFile` comment in `coverlet.runsettings` explains that `obj/` is excluded because the
  paths are unresolvable against `git ls-files`. Once `#line` ships, that rule **silently stops
  applying to the Polyglot output** (§3) while still applying to source-generator output. The
  comment becomes actively misleading and must be rewritten to say so.

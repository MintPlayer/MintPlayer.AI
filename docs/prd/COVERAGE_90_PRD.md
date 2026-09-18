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

### S2 — `#line` end-to-end on a real solver ✅ **RUN 2026-09-18 — PASSED, superseded in scope**

The compiler work shipped as [Polyglot#70](https://github.com/MintPlayer/MintPlayer.Polyglot/pull/70)
(`v0.10.0`), so this ran against the released package on **all nine** solvers rather than only
`mountaincar`. Observed on `PolyglotOriginInfo=true`:

- **8,466 directives over 8,466 generated lines** — 7,841 positioned + 625 `#line hidden`, i.e.
  exactly one per line, which is the invariant that prevents drift.
- All nine `.pg` files named with absolute, forward-slashed paths; **multi-file attribution works**,
  so the `fileId = 0` prerequisite (Polyglot `compiler.cpp`) is genuinely fixed rather than masked
  by single-file testing.
- The report's `<sources>` root is `C:/Repos/MintPlayer.AI/` with **repo-relative** `filename`
  values — the shape the coverage service needs (feeds S6).
- The polyglot `obj/` paths are **gone** from the report. The one remaining `obj/` entry is the
  pre-existing `Inject.g.cs` leak that `COVERAGE_PRD.md` already documents, unchanged.
- 707/707 tests pass with directives on — no behavioural change.
- The option-tagged stamp works in this repo: `obj/Release/net10.0/polyglot/__polyglot-origin.stamp`.

### S3 — Does the number actually go up? ✅ **RUN 2026-09-18 — PASSED (+8.25 pp), with a cost finding**

A/B on the full fast bucket (`Category!=Slow`, Release, 707 tests, Polyglot 0.10.0):

| | origin info OFF | origin info ON | Δ |
|---|---|---|---|
| Line rate | **59.86%** | **68.11%** | **+8.25 pp** |
| Covered / valid | 6,965 / 11,636 | 10,458 / 15,355 | +3,493 / +3,719 |
| Tests | 707 pass | 707 pass | no behavioural change |
| Duration | 3m 08s | **15m 55s** | **5×** |

The OFF run reproduces the published badge to two decimals, which is what makes the A/B
trustworthy. §2's claim is confirmed and then some — the nine solvers enter at **3,493/3,719 =
93.9% covered**, i.e. they were depressing the number purely by being invisible.

Per-solver, and this is actionable on its own: `blockdude` 99.4%, `crazyfruits` 99.2%, `lunarlockout`
98.9%, `chess` 98.0%, `fruitcake` 97.5%, `tetris` 97.3%, `draughts` 93.2%, **`snake` 75.4%**,
**`mountaincar` 58.0%**. The last two are the only `.pg` files worth writing tests against.

**The cost finding: instrumenting the solvers costs 5× wall-clock.** They are the hot path — the
parity and perft suites push millions of simulation steps through them — so every inner-loop line
now takes a `RecordHit` call it previously skipped by being excluded. 3m → 16m on the *fast* bucket
is a CI-budget problem, not a rounding error. Mitigation measured as **S3c** (coverlet `SingleHit`,
which stops recording after a line's first hit instead of counting every pass); see §6a.

Remaining gap to the 90% goal: 13,819 covered needed vs 10,458 today = **+3,361 lines**, which is
what M63.4/M63.5 have to find.

### S4 — Vitest from zero on one spec ✅ **RUN 2026-09-18 — PASSED**

Add `vitest` + `@vitest/coverage-v8`, a `test` target in `angular.json` via
`@angular/build:unit-test`, and one spec. **Pass:** `lcov.info` with `DA:<line>,<hits>` records.

**Result:** 7/7 tests pass, `coverage/ClientApp/lcov.info` with **54 `DA:` records**, plus
`coverage-final.json` (istanbul, with statement column ranges — kept deliberately, it is what S5
needs; lcov is line-only and cannot be remapped faithfully).

Notes for anyone redoing this:

- **`@angular/build` 22.0.6 declares `peer vitest@^4.0.8`.** Pinning `^3.2.4` fails `npm install`
  with ERESOLVE. Read the peer range rather than assuming the current vitest major.
- The first spec was written against a *generated* file (`mountaincar_solver.ts`) rather than an
  Angular component, on purpose: the solver twins are pure functions, so they need no TestBed, no
  DOM and no fixtures — the right place to start a suite that must exist from zero, and the exact
  surface the `.pg` union needs covered.
- `coverageInclude` is scoped to `src/app/**/*_solver.ts`. The rest of the SPA has no tests; including
  it would add denominator and nothing else.

### S5 — Chained remap `.pg → .ts → .js`

The weak link. `@vitest/coverage-v8` collects on the built bundle and remaps to `.ts` via the
bundle's map; getting to `.pg` needs that composed with the Polyglot map.

**Result 2026-09-18: ACHIEVED, but not the way the plan assumed.**

The clean route was tried first and **structurally cannot work**: a Vite plugin returning the
Polyglot `.ts.map` from a `load` hook, so v8 would compose `.pg → .ts → bundle` by itself. The
config file loads (`Using Vitest configuration file: …vitest.config.mts`) but the plugin never
fires, because **`@angular/build:unit-test` pre-builds the app with esbuild before Vitest starts** —
the log shows `Building… → Application bundle generation complete` ahead of the run, so the `.ts`
never passes through Vite's transform hooks at all. Confirmed by istanbul's output carrying no
`inputSourceMap`. The plugin was removed rather than left in place looking functional.

So the composition is done after the fact by **`tools/pg_coverage_remap.mjs`** (~150 lines, no
dependencies — base64-VLQ decoded inline): it reads istanbul's `coverage-final.json`, maps each
statement's start line through the sibling `*_solver.ts.map`, and writes a cobertura keyed on the
`.pg`. Several `.ts` statements collapse onto one `.pg` line (one-line loops, one-line bodies), and
it takes the **max** — covered if any contributing statement ran, matching both the C# side and the
server's merge.

**Measured:** `mountaincar_solver.ts` → `mountaincar_solver.pg`, 39/58 statements hit, **37/42 `.pg`
lines (88.10%)**. Cross-checked line by line against the source: every mapped line is a real
executable statement (15 `clampF`, 87–95 ctor, 100–103 `reset`, 108–111 `setState`, 115–132 `step`,
137–140 `buildObservation`) with **no blank lines, comments or out-of-range numbers**, and the five
misses are exactly `chooseAction` (145–150), which the spec deliberately never calls. Zero
misattribution.

**Consequence:** §4's two-input union is real — the C# report and this one both key on the same
`.pg` paths, and the service merges them with max semantics.

### S6 — Server acceptance of a `.pg`-keyed report 🟡 **half-answered; needs a push**

Local half **confirmed** (S2): the report carries `<source>C:/Repos/MintPlayer.AI/</source>` with
repo-relative `filename` values ending `.pg`, and those paths are in `git ls-files` — which is
exactly what the service suffix-matches against. Nothing in the report shape should defeat it.

Still unverified because it cannot be checked locally: that the service *actually* resolves a `.pg`
extension, renders it, and merges two reports naming the same `.pg`. **Pass:** the build reaches
`Complete` and the `.pg` file is browsable with per-line gutters. Cheapest as a throwaway branch
push. **Fail:** fall back to local ReportGenerator merge (§7) and upload one pre-merged cobertura.

### S3c — Does `SingleHit` recover the 5× slowdown? *(added 2026-09-18, forced by S3)*

S3 measured the fast bucket going 3m08s → 15m55s once the solvers are instrumented, because they
are the hot path. Coverlet's `<SingleHit>true</SingleHit>` stops recording a line after its first
hit instead of counting every pass, which is precisely the hot-loop cost. **Pass:** wall-clock
returns to roughly the OFF baseline *and* the line rate is unchanged (hit *counts* drop to 1, but
covered/not-covered — all this repo and the server's max-merge consume — must not move).
**Fail:** the 5× is intrinsic, and the choice becomes explicit: pay it, or collect coverage on a
reduced filter and accept a partial number.

**Result: PASSED — 15m55s → 5m07s, line rate bit-identical (10,458/15,355).** Mechanism, measured on
an isolated 20M-iteration probe: coverlet injects `ldc.i4` + `call RecordHit` per *sequence point*,
and `RecordHit` is an `Interlocked.Increment`. Under xUnit's parallel execution several threads hit
the same `HitsArray` slots, so each atomic forces exclusive ownership of that cache line away from
the other cores.

| probe (20M iterations) | 1 thread | 4 threads |
|---|---|---|
| coverage off | 100 ms | 48 ms |
| coverlet default | 990 ms (9.9×) | **3411 ms (71×)** |
| coverlet `SingleHit` | 186 ms (1.9×) | 111 ms (2.3×) |

Note the multi-hit 4-thread case is *slower in absolute terms* than 1 thread — a textbook
false-sharing signature, and the real reason the regression was 5× rather than ~1.3×. `SingleHit`
keeps the call and replaces the atomic with a plain read + predicted branch, hence its ~1.9× floor.

### S3e — Would Microsoft's `Code Coverage` collector be faster? ❌ **RUN 2026-09-18 — REJECTED**

An isolated probe suggested MS `Code Coverage` (basic-block probes, non-atomic byte flags, dynamic
instrumentation) would run at ~1.0–1.5× uninstrumented against coverlet-SingleHit's 1.9–2.3×, i.e.
roughly 2× better. **It did not survive contact with the real suite.**

| collector | fast bucket wall clock | line rate |
|---|---|---|
| coverlet + `SingleHit` | **4m 39s** | 68.11% (10,458/15,355) |
| MS `Code Coverage;Format=cobertura` | **5m 57s** | 68.29% (10,445/15,295) |

Fidelity was fine — it honours `#line` and attributed all nine `.pg` files — but it is **28% slower
here**, and its cobertura emits absolute backslash paths with an empty `<sources>` element, which
would additionally break the server's `git ls-files` suffix matching without a normalisation pass.
Recorded so the idea is not re-proposed: the probe-level extrapolation was simply wrong at suite
scale. **Coverlet + `SingleHit` stays.**

### S3f — Class split: the free parallelism win ✅ **RUN 2026-09-18 — 279s → 192s, nothing traded**

xUnit's unit of parallelism is the test *collection*, which by default is the class; tests inside one
class run strictly serially. `BlockDudeGateBoardsTests` held 188.2s + 81.4s + 2.4s = **272s of serial
work against a 279s wall clock for all 707 tests** — it *was* the critical path, and the observed
"parallelism of 3.0 on 8 cores" was a consequence of it, not a scheduler weakness. Given that
structure the best achievable was `max(272, 835.8/8) = 272s`, so xUnit was already at ~98% of
optimal: **the runner had nothing left to give, and switching to NUnit/MSTest/TUnit would buy only
what splitting the class buys, at the cost of migrating 707 tests.** (Playwright is browser
automation and cannot run these tests at all.)

Splitting the three tests into three classes **in the same file** (the collection is keyed on the
class, not the file) lets them run concurrently. No assertion touched, no constant changed, no
sampling lost. Expected bound afterwards: `max(188.2, 835.8/8) = 188.2s`.

**Result: 192s wall / 3m02s reported, line rate bit-identical at 10,458/15,355, 707/707 pass.** Full `.pg` coverage now costs ~4s over the 188s baseline that measured none of it. This meets §10a and retires the need for both the sampling-constant cuts and the solver fixes.

**The ceiling this exposed remains worth recording.** Amdahl's floor for the suite is the longest
*single* test, so while `EveryGateBoardIsOneTheExactOracleCanFullyLabel` takes 188s, **no runner,
scheduler or core count can get below ~3m08s.** Reaching the ~3-minute bar therefore requires that
one test to get cheaper — via its sampling constants, or by making the oracle itself faster (§10b).

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

## 10. Denominator decisions — **settled by the owner 2026-09-18**

1. **`tools/Lab` (~3.2k lines): TEST IT.** Not excluded. The owner declined the "exclude `tools/**`
   as dev tooling" recommendation, so Lab stays in the denominator and M63.5 must genuinely cover
   it. This is the largest single block of work in the milestone — `VizServer.cs` 479,
   `Tetris/TetrisLab.cs` 372, `CrazyFruits/CrazyFruitsLab.cs` 297, `BlockDude/BlockDudeLevelBench.cs`
   277. Consequence to plan for: much of Lab is `HttpListener`/WebSocket/console-rendering glue, so
   expect a seam-extraction pass (pull the testable logic out from behind the IO) rather than
   straight unit tests against the current shapes.
2. **`Category=Slow`: ADD A `Medium` BUCKET.** Reclassify the Slow tests that run in seconds rather
   than minutes and include them in the coverage run, converting existing tests into coverage at
   near-zero cost. Needs a measurement pass first — the 27 `Category=Slow` traits across 16 files
   have never been individually timed.
3. **`src/RLDemo.Console` (631 lines): OUT OF SCOPE.** Stays unreferenced and therefore absent from
   the report. It is a demo CLI, not shipped library surface. *(Note the deliberate asymmetry with
   §10.1: Console is out, Lab is in and gets tested. The distinguishing line is that Lab is already
   referenced by the test project and Console is not.)* To be recorded in `coverlet.runsettings` so
   the exclusion is a written decision rather than an accident of project references.
4. **Re-baseline 90% after M63.4.** Still open, deliberately. With §10.1 resolving to "test it"
   rather than "exclude", the denominator stays large (15,355 lines today), so 90% means +3,361
   covered lines — a materially bigger job than if `tools/**` had been excluded. Worth revisiting
   the target once M63.4 lands and the true denominator is known.

### 10a. Runtime budget — a hard constraint, not a preference

The owner's bar: **~3 minutes is acceptable, 16 minutes is "waaay too long."** `SingleHit` already
took the fast bucket from 15m55s to 5m07s at zero cost to the number (§6 S3c). The remaining ~2
minutes over baseline is under active investigation (four-agent sweep, 2026-09-18): whether the
28.5bn `.pg` hits are genuine execution volume, a codegen/`#line` amplification artifact, or a
handful of tests with needlessly large iteration constants. **CI wiring (M63.7) is blocked until
the fast bucket is back near 3 minutes** — putting a 5–16 minute job on every PR contradicts the
repo's standing "CI cost is the bottleneck" rule.

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

## 12. M63.5 — the `tools/Lab` plan *(two-agent seam analysis, 2026-09-18)*

The owner chose to **test** Lab rather than exclude it (§10.1), so this is the bulk of M63.5. Measured
baseline from the M63.3 run: **Lab is 10/1,715 coverable lines = 0.6% covered.** (The oft-quoted
3,352 is *raw* lines; coverlet counts sequence points. `VizServer.cs` is 479 raw lines of which
**262 are one `private const string Page` HTML literal** — a const field has no sequence points, so
it is 105 coverable, not 479. Budget against 1,715.)

### 12.1 The structural finding

Every `*Lab.cs` has the same shape: `new CliArgs(args)` → 20–60 lines of flag reads → build an
options **record** → hand it to `LabHost.Run(...)` **inside a lambda**. The parsed values are never
returned and never observable, so 60–90% of each file — pure logic, and where every default lives —
is unreachable without booting DI and starting a training run. There is no `Parse` seam anywhere in
the project.

**One extraction per file, `internal static XOptions Parse(CliArgs a)`, converts ~600 lines from
untestable to trivially testable.** It is the same change eight times, it returns option records
that already live in the Campaigns library (so Lab gains no new types), and it changes no control
flow. `InternalsVisibleTo` for the test assembly **already exists** in the Lab `.csproj`, so nothing
needs widening to `public`; test files need `extern alias Lab;` because the project is referenced
with `Aliases="Lab"` (the Lab exe's generated `Program` would otherwise collide with RLDemo.Web's
under `WebApplicationFactory`).

### 12.2 Ranked work

| # | Target | Unlocks | Refactor cost |
|---|---|---|---|
| 1 | `CubeDaviLab.Resolve(args, cfg, …)` + `CubeDaviConfig.Load(dirs, out source)` | ~225 | ~10 lines — two halves of one config contract, test together |
| 2 | The `Parse` seam ×8 (Chess, Draughts, Snake, FruitCake, BlockDude, Cube, CubePolicy, RushHour, Connect4) | ~600 | ~6 lines each |
| 3 | `GateLines(...)` in Tetris + CrazyFruits `RunBaselines` | ~60 | ~12 lines |
| 4 | `private`→`internal` only: `Summarize`, `Emit`/`ExistingLength`, `Envelope`/`CurrentTopology`, `ParsePort`, `RotateLog` | ~50 | **zero** |
| 5 | `VizServer.SampleOnce(ref string?)` + `internal` ctor | ~30 | ~10 lines |
| 6 | `CampaignCli` CSV contract, `EvalStats` (dedupe `Std`/`Report`), `BlockDudeValueCalibration.Score` | ~65 | ~20 lines |

### 12.3 Constraints that shape the tests

- **Never capture `Console.Out`.** It is process-global; a `Console.SetOut` strategy would force test
  serialisation, undoing the parallelism M63 just bought (S3f). This is why item 3 returns
  `IEnumerable<string>` and the caller prints — the verdict strings become assertable without
  touching the console at all.
- **Never bind a port.** `HttpListener.Start()` on a fixed port would collide with the owner's own
  `--viz` session. The `VizServer` ctor does no IO (`Prefixes.Add` does not bind; `_listener.Start()`
  is in `Start()`), so an `internal` ctor gives tests a full object that never opens a socket.
- **Never call `LabHost.Run`.** It takes `TrainingDirectoryLock` on the data dir and may hit
  `Console.ReadLine`.
- `Category=Slow` is required for anything loading a checkpoint (`FileModelStore`), touching
  ILGPU/GPU, or running real episodes. `Backend.Current` is process-global static state — a test
  mutating it can corrupt others in the same assembly.
- RNG is explicitly seeded everywhere in Lab; no unseeded RNG was found. Determinism is not a risk.

### 12.4 Deliberately not tested (~550 of 1,715 lines)

Socket lifecycle (`AcceptLoop`, `ServeWebSocket`, `SendPump`, `Drop`, `Dispose`), `HeldOut` and the
search tiers in `BlockDudeLevelBench`, both Snake eval methods, `BlockDudeDemoProbe`,
`ConvForwardBench`, `StrengthCli`, `ChessDemo`'s body. These need real checkpoints, a GPU, or
minutes of episodes, and the assertions would be tautologies. **Recorded here rather than papered
over with `Assert.NotNull` smoke tests** — the owner wants the number to mean something.

### 12.5 Bugs the analysis surfaced *(fix in this PR, per the one-PR rule)*

1. **`Std` uses the population divisor `N`**, then the caller computes `SE = Std/√N`, understating
   the standard error (should be the sample SD, `N−1`). It feeds the "SIGNIFICANTLY BETTER → ship
   it" verdict in `FruitCakeAb`. Duplicated **verbatim** in `FruitCakeAb.cs` and
   `FruitCakeSearchEval.cs` — dedupe into `EvalStats` and fix once.
2. **`CliArgs.Value` has no `StartsWith("--")` guard** — `--data --seed 7` makes `Str("--data")`
   return `"--seed"`, and `Has("--grow")` is true when `--grow` appears as a *value*.
3. **`CampaignCli`: `headerWritten = File.Exists(csvPath)`** treats a zero-byte file as
   already-headered, so the CSV silently loses its header row.
4. `median = sorted[Length/2]` is the **upper** median for even N, not the mean of the middle two.
5. `CubeDaviLab` is the only file bypassing `CliArgs`, with a hand-rolled 35-branch loop whose
   `int.Parse`/`ulong.Parse` calls omit `InvariantCulture` while its `double.Parse`/`float.Parse`
   calls include it. **Not the live bug it first appeared to be** — `int.Parse` defaults to
   `NumberStyles.Integer`, which disallows thousands separators on every culture, so `"1.024"`
   throws regardless of locale; the only real exposure is a culture whose negative sign is not `-`.
   Worth fixing as consistency, and migrating the file to `CliArgs` is the actual fix.

### 12.6 Feasibility — this is the §10.4 re-baseline, with numbers

Cover **everything worth covering** in Lab (1,715 − ~550) and the repo reaches:

**10,458 → ~11,623 = 75.7%.** Still **2,196 short of 90%.**

Closing that gap needs ~69% of Campaigns + Environments + Core + Web on top (pool 3,188), where
Campaigns is long training loops deliberately excluded from the measured bucket. So 90% with
`tools/**` *in* the denominator is not reachable inside M63. The honest options:

- **Exclude `tools/**`** — denominator 13,640, today's figure becomes 76.6%, and 90% needs +1,828
  from a 3,182-line pool. Reverses §10.1.
- **Move the target** to ~80%, which the Lab work alone very nearly reaches.
- **Keep 90% as a multi-milestone arc**, with M63 landing the infrastructure and the Lab seams.

**Still open.** Recorded here so the decision is made against arithmetic rather than optimism.

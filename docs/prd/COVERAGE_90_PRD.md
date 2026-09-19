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

### S6 — Server acceptance of a `.pg`-keyed report ✅ **PASSED 2026-09-18 — measurement AND presentation**

**Result.** PR #54's CI uploaded the report (`Upload accepted`, `Finish requested (202)`) and the service
published both checks. `coverage/project` reported **70.7% (+10.9% vs base 59.8%)** and
`coverage/patch` **68.8% of added lines (141 of 205)**. Both carry `conclusion: neutral` — the GitHub UI
renders that as *skipping*, which reads like a failure but means **informational only**, because
Blocking is off in the repository gate and no patch target is configured.

**The number is the proof.** The server total matches the local figure to the decimal, and that is only
possible if it resolved the `.pg` paths:

| | covered / valid | rate |
|---|---|---|
| with `.pg` | 10,875 / 15,377 | **70.72%** ← server said 70.7% |
| without `.pg` | 7,382 / 11,658 | 63.32% |
| the `.pg` block alone | 3,493 / 3,719 | 93.9% |

Had the nine `.pg` files been dropped as unresolvable, the service would have reported **63.3%**. So
`git ls-files` suffix matching resolves a `.pg` extension, and §4's union architecture is confirmed
end to end on the C# side.

**The UI half, surveyed 2026-09-18 in an authenticated session — CONFIRMED.** Opened
`crazyfruits_solver.pg` on commit `b50fe5e`:

- **1,103 line anchors for a 1,103-line file**, rendered inside an `mp-code-snippet`. (The viewer is
  shadow-DOM, so `document.querySelector` returns nothing — a recursive walk over `shadowRoot`s is
  required to inspect it.)
- **The gutter lines up with the real source.** Line 1 (a comment) and line 1103 (a closing brace)
  are unannotated; 19 `fn cfFruitOf`, 285 `for c in 1..(Size + 1)`, 591 `clearStep`'s cell sweep and
  666 `resolveCascades` all carry hit marks. Those four line numbers were identified *independently
  from the coverage XML* during the M63 hot-line analysis, and the rendered Polyglot source shows
  exactly those constructs at exactly those lines.
- **The file total matches the local cobertura exactly**: 617/622, with the annotation histogram
  summing to 622 coverable (5 at zero) and 481 unannotated.
- **Branch coverage survived the remap and is rendered** — "Branches: 2 of 2 taken". Commit-level:
  **6,169/8,932 branches (69.1%)**, which was never visible locally.
- Commit total **74.2% (11,421/15,398 lines, 219 files)**, build Finalized, against a local
  11,418/15,392. The 3-line/6-line difference is the platform variance recorded in §8.1 of
  `CAMPAIGN_TESTABILITY_PRD.md` plus the known `Inject.g.cs` leak.

**A UI caveat caused by our own `SingleHit` decision.** The per-line `N×` badges are **not execution
counts**. With `SingleHit=true` every contributing sequence point reports `hits=1`, so `4×` means
*four generated C# statements collapsed onto that one `.pg` line*. `cfFruitOf` (line 19) displays
`1×` and actually executes ~368 million times. Harmless for coverage — covered-vs-not is exact, and
that is all anything here consumes — but actively misleading read as a profiler. Worth a label change
in `MintPlayer.Spark` (*contributing statements*, not a multiplier) for reports whose hits are 0/1.

**Also worth knowing:** a `401` renders as *"No coverage data for this file"*, which reads like
missing data rather than an auth wall, and the SPA logs `NG04002: 'login'` because it routes to a
`login` path that does not exist.


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

> **Status 2026-09-18 — outcomes are recorded per sub-milestone below and in detail in §6 (spikes),
> §12 (the Lab plan) and `PLAN.md` M63. Where a gate below was not met, it says so.**

- **M63.1 — Spikes S1 + S4** ✅. S1 passed *and falsified the design*: emitting `#line` only on
  source-line change lets braces drift onto unrelated `.pg` lines and report them covered, so the
  rule became a directive on **every** line.
- **M63.2 — Polyglot `#line` (flag-gated)** ✅, shipped as Polyglot#70 / `v0.10.0`; the option was
  renamed `--origin-info` during review.
- **M63.3 — Adopt in MintPlayer.AI** ✅ **59.86% → 68.11%**. Gate met (the total rose rather than
  dropped). S6's *browsable on the service* half is unverified — the web UI requires authentication;
  the measurement is proven, the presentation is not.
- **M63.4 — The denominator decisions** 🟡. Scope recorded in `coverlet.runsettings` — gate met.
  The `Category=Medium` half was implemented, measured and **rejected** (§12.7).
- **M63.5 — C# coverage push** 🟡 **gate NOT met, and not meetable.** The stated gate was
  "fast-bucket line rate ≥ 90%"; §12.6 shows that is unreachable with `tools/**` in the denominator.
  Delivered: `tools/Lab` 0.6% → 23.3% and four real bugs. Outstanding: ChessLab seams, and Campaigns
  — now planned as its own milestone (`CAMPAIGN_TESTABILITY_PRD.md`, M64).
- **M63.6 — Frontend tests + TS `.pg` mapping** ✅. Gate met. S5's clean route **structurally cannot
  work** (the Angular builder pre-builds with esbuild before Vitest starts), so the composition is a
  post-process; recorded either way as the gate required.
- **M63.7 — CI wiring + README/badge note** ✅.

Per the repo's batching rule the **full test suite runs once** per milestone, not per increment;
intermediate work verifies by targeted slice + type-check. **Every timing claim must be measured
with `--collect`** — see §12.7 for what happened when it was not.

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
4. **Re-baseline 90% after M63.4.** **ANSWERED in §14** (2026-09-18): 90% is not reachable in this arc — covering all of Campaigns reaches 81.5%. Originally left open as: With §10.1 resolving to "test it"
   rather than "exclude", the denominator stays large (15,355 lines today), so 90% means +3,361
   covered lines — a materially bigger job than if `tools/**` had been excluded. Worth revisiting
   the target once M63.4 lands and the true denominator is known.

### 10a. Runtime budget — **softened by the owner 2026-09-18 (M65)**

> **Correction.** This section's heading and its "CI wiring is blocked until…" clause are no longer
> accurate, and are left below only because decisions elsewhere in this PRD were made under them.
> The owner's position as of M65: **"3 minutes is just a suggestion, not a hard requirement"** — the
> Nx cache can serve several test results, so a suite that grows past the bar is not automatically a
> problem. The 180s figure is a target to steer by, not a gate. What survives unchanged is the
> *reasoning*: a test that is slow **and** shallow is still not worth its seconds (§12.7), and the
> Amdahl floor still means the longest single test bounds the suite (§6 S3f).

#### The original framing, as decided on 2026-09-18 (superseded above)

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

### 12.7 M63.4 — `Category=Medium` was tried and REJECTED *(2026-09-18)*

§10.2 settled on adding a `Medium` bucket so the `Category=Slow` tests that run in seconds would
start counting. It was implemented, measured, and **reverted**. Recorded here because the reasons are
not obvious and the idea will otherwise be re-proposed.

**The candidates were real.** Six of the 21 `Slow` tests are determinism checks, bitwise-identical
checkpoint checks and a bounded scramble solve — no training loop — and timed at 1.83s / 3.83s /
4.18s / 7.37s / 8.34s / 10.33s, 17s of wall clock in parallel. Retagging alone would have sufficed,
since the CI filter is `Category!=Slow`.

**Two findings killed it.**

**1. A timing-assertion test cannot live in the instrumented bucket — at any speed.**
`DaviTrainerTests.BatchedGreedySolve_IsFasterThanPerSuccessor` asserts that batched solving beats
per-successor solving. Under coverage instrumentation the relative timings **invert** and it fails:

```
batched (6488 ms) should beat per-successor (5906 ms)
```

This is a permanent property, not a threshold to tune: instrumentation does not slow both paths
equally. Any test whose assertion is a *performance comparison* belongs in the uninstrumented bucket
by construction. Worth remembering before the next attempt to move something into the fast bucket.

**2. The timings that justified the move were measured WITHOUT coverage.** The probe ran
`dotnet test` with no `--collect`, and those numbers were then used to argue for adding the tests to
an *instrumented* run. The same test measured **10.33s uninstrumented and 45s instrumented**. The
whole bucket went:

| | wall clock | line rate |
|---|---|---|
| before | 213s | 70.72% |
| with `Medium` | **380s (6m 20s)** | 71.90% (+1.18 pp) |

**+167 seconds, not the +17s predicted** — decisively over the §10a budget for +1.18 pp. (This is the
same probe-versus-suite extrapolation error that got the MS `Code Coverage` collector rejected in
S3e; it was then repeated one spike later. Any future timing claim about this suite must be measured
*with* `--collect`.)

**Status: §10.2's decision is superseded by measurement.** The `Medium` bucket is not worth its cost
at the current budget. If it is revisited, the route is (a) measure instrumented, always, and
(b) exclude any test whose assertion is a timing comparison. The +1.18 pp it would have bought
came mostly from Campaigns (34.3% → 40.0%), which remains the largest uncovered pool and is better
attacked with fast tests written for the purpose than by reclassifying slow ones.

---

## 13. ~~ The TypeScript upload was removed *(2026-09-18, owner's call)*~~

> **SUPERSEDED by §16 (M66).** The measurement below is correct; the conclusion drawn from it is
> not. It was taken when the repo held one frontend spec that constructed no net, so the
> TypeScript side contributed almost nothing — it measured the absence of browser-side tests, not
> the value of the union. Measured properly in §16, the union covers **155 `.pg` lines the C#
> side can never reach**. Read §16 first.

M63.7 uploaded a second report projecting the TypeScript twins' coverage back onto the `.pg` sources,
so each `.pg` line carried the union of its C# and TS hits. It worked — spike S6 proved the service
resolves `.pg` paths *and* unions two reports naming the same one — but the measurement did not
justify keeping it:

| | |
|---|---|
| CI cost | **+57s** (3m53s → 4m50s) |
| Coverage gained | **+0.1pp** (70.7% → 70.8%) |

The reason it is so small is structural, not a defect: **the C# side already covers those same `.pg`
lines at 93.9%** through the `#line` pragmas. The two reports were double-counting one source file.
Not quite zero — istanbul's statement view reached about 12 `.pg` lines the C# sequence points did
not, because the two tools see different granularity — but not 57 seconds' worth.

**What was kept, and why.** The frontend suite still runs in CI without feeding coverage. Those specs
are the only thing that would catch a **Polyglot TypeScript codegen regression**, which the C# side
structurally cannot see: both targets come from the same `.pg`, but only the TS one is exercised by
the browser. `tools/pg_coverage_remap.mjs` is retained as the working implementation.

**When to bring frontend coverage back, and in what shape.** Not by re-uploading the generated twins —
that is the double-count above. The gap worth closing is the **hand-written ClientApp** (~64 `.ts`
files: components, services, the game hosts), which is currently *not in the denominator at all*
because `coverageInclude` is scoped to `*_solver.ts`. Adding it is the honest move **once specs
exist**: with one spec in the repo today it would add a large uncovered denominator and drop the
number sharply, which measures nothing new.

## 14. §10.4 answered: 90% is not the right target for this arc

The question was left open deliberately until the denominator was known. It now is:

| | |
|---|---|
| Today | **74.18%** (11,418 / 15,392) |
| If **all** remaining Campaigns lines were covered | **81.5%** |
| 90% requires | **+2,434** lines, from 3,974 uncovered in total |

So 90% needs, on top of finishing Campaigns, most of `tools/Lab` (1,344 uncovered) **and**
`Environments` (1,048). That is not a milestone; it is a programme.

The three honest options, unchanged from §10.4 but now priced:

1. **Exclude `tools/**`** — reverses §10.1. Denominator drops to ~13,640 and today's figure becomes
   ~81%, with 90% needing ~+1,200 from Campaigns and Environments. Reachable.
2. **Move the target to ~80%** — roughly where the current trajectory lands once Campaigns is
   finished, with `tools/**` still in.
3. **Keep 90% as a multi-milestone arc** — M63 and M64 landed the infrastructure and the first half;
   two or three more milestones of test-writing would be needed.

**Recommendation: (2) or (3), not (1).** Excluding `tools/**` now would reverse a decision made
deliberately and would move the number without covering a line — the exact kind of metric change this
PRD has argued against throughout. The number is only worth having if it means something.

---

## 15. M65 — the test-writing pass, and the frontend joins the number *(2026-09-18)*

M63 built the measurement and M64 made the campaigns testable. M65 is the milestone that actually
writes tests, on the targets the service's own per-file ranking named.

| | before | after |
|---|---|---|
| C# line coverage | 74.18% (11,423 / 15,398) | **78.70% (12,180 / 15,476)** |
| C# branch coverage | 69.11% | **72.04%** |
| Tests (fast bucket) | 837 | **1,044** |
| Fast-bucket wall clock | 2m26s | **2m04s** |
| Frontend | not measured | **144 tests, 644/644 lines over 13 modules** |
| Combined, as the service will merge it | — | **12,824 / 16,120 = 79.55%** |

The denominator grew by 78 lines (the Lab `Parse` extractions), so the percentage is not flattered by a
shrinking base. The suite got **faster**, not slower — the new tests are pure-logic and sub-second, and
they spread across new xUnit collections, which the S3f class-split finding predicts.

### 15.1 What moved, and why these targets

Targets came from the service's per-file uncovered ranking at `b50fe5e`, not from guesswork. The
ranking also **corrected the plan before any code was written**: §2 and the M65 seam analysis both
expected the Kociemba block to be the headline (~435 lines), but the real report showed `K_CubieCube`
already at 347/444 — it is exercised end-to-end by `CubeApiTests`' scramble-and-solve. The block was
worth ~190, not ~435, and the effort went elsewhere.

| gain | file | how |
|---|---|---|
| +86 | `BlockDudeExpertIterationCampaign` | lifecycle without `TrainChunk` — resume, `--fresh`, the frontier sidecar, `Evaluate` |
| +72 | `ReinforceTrainer` (now 79/79) | a 3-step toy env; the loop's *bookkeeping*, never "does it learn" |
| +69 | `K_CubieCube` | coordinate round-trips, `verify()`'s error branches, move order-4 |
| +68 | `NetworkTelemetry` (was 0/72) | `NetworkInspector`'s layer pairing, topology arithmetic, heatmap block-mean |
| +61/+57/+52/+33/+33/+20 | `BlockDudeLab`, `DraughtsLab`, `ChessLab`, `SnakeLab`, `CrazyFruitsLab`, `TetrisLab` | the M63.5 `Parse` seam pattern, applied to six more labs |
| +38 | `CubeImitationCampaign` | lifecycle + progress sidecar |
| +35 | `BlockDudeGreedy` (was 0/35) | ending classification, the move log, the no-revisit invariant |
| +26 | `K_Tools` (was 0/30) | `verify` per error code |
| +11 | `CubeModelService` | the `static Rollout` only — never the `AdaptiveBackend` constructor |

Also newly covered and absent from the old top-60: `ModelServiceInfrastructure`, `BlockDudeController`,
`VersionController`.

**Six Lab production files gained `Parse`/`Options` seams** (`ChessLab`, `DraughtsLab`, `BlockDudeLab`,
`SnakeLab`, `TetrisLab`, `CrazyFruitsLab`), each a mechanical extraction of the pure flag-reading head
of `Run` — same defaults, same order, no control flow changed. Chess and Draughts needed *two* seams
each rather than one, because their flag head is split by the demo/bench/strength dispatches; folding
the later reads upward would have made a malformed *training* flag throw inside a read-only mode, which
is a behaviour change, not a refactor.

**Two production changes beyond the seams.** `K_CubieCube.getURFtoDLB`/`getURtoBR` widened from private
to `internal` — their setters were already public, so the set/get pair could not be asserted at all.
That is preferred to the reflection the first draft used, which would have pinned the method *names*
rather than the behaviour. And the `StartupCheckpoint` fix in §15.4.

### 15.2 The TypeScript half — §13 is partly reversed, and one of its claims was wrong

> **Partly superseded by §16 (M66).** What this section says about the *hand-written* ClientApp
> modules still holds. What it says about excluding the generated twins does not: that exclusion
> made `pg_coverage_remap.mjs` a silent no-op, and the twins are back in `coverageInclude` as of
> §16 (stripped from *this* report, and uploaded as a separate `.pg`-keyed one).

§13 removed the frontend upload and gave two reasons. The first still stands; the second did not
survive measurement.

**Still true: never upload the generated twins.** The `*_solver.ts` files are transpiled from the same
`.pg` the C# side already covers at 93.9% through `#line` pragmas, so uploading both double-counts one
source file for +0.1pp. They are now excluded explicitly (`coverageExclude` in `angular.json`) rather
than by omission, and `tools/pg_coverage_remap.mjs` stays unused — it exists to rewrite `.ts`
coordinates onto `.pg` line numbers, which is only needed because the twins are gitignored.

**Wrong: "it would add a large uncovered denominator and drop the number sharply."** That was reasoned,
not measured, and it is not how this builder behaves. `@angular/build`'s unit-test builder pre-bundles
with esbuild before vitest starts, so a source file **no spec imports never becomes a module the v8
provider can synthesise empty coverage for**, and `excludeAfterRemap: true` then drops it. Evidence:
`coverageInclude` was `src/app/**/*_solver.ts`, which matches 9 files; the report contained exactly 1 —
the only one with a spec.

That is a double-edged finding and the PRD should say so. It de-risks widening the globs, but it also
means **`coverageInclude` cannot be used to hold the frontend honest**: a module with no spec is not
counted as uncovered, it is simply absent, so the percentage cannot fall when someone adds untested
code. A frontend coverage number here measures *the files that have tests*, not the app. Treat it as a
regression guard on tested modules, never as an answer to "how much of the frontend is tested".
It is also a behaviour of this `@angular/build` 22 + vitest 4 combination, not a contract.

**So the globs are scoped to what actually has specs** rather than to `src/**/*.ts`. Deliberately out:
the 14 Canvas/Three.js/WebSocket renderers and 15 Angular components (~8,200 lines). Covering those
means faking a 2D context to assert "it called `fillRect`", which measures nothing; jsdom's
`getContext('2d')` returns null, and there is no TestBed precedent anywhere in the repo to build on.

**The one thing that nearly broke it.** The cobertura reporter writes filenames **relative to the
Angular project root, with the platform separator** — `src\app\chess\chess-net.ts` on a Windows agent,
verified in the real report. The service resolves a path by suffix-matching against `git ls-files`,
which stores forward-slashed repo-root-relative paths, so every file would have been silently dropped
as unmatched: no error, no warning, just a frontend report covering nothing.
`tools/reroot_frontend_coverage.mjs` normalises the separators and re-roots the paths (and is
idempotent, since the upload step runs under `always()`). It warns loudly when it finds no filenames at
all, because an empty report that uploads cleanly is indistinguishable from a healthy one.

`finish: true` moves to the frontend upload in **both** workflows — the last upload of a build closes
it. `build-master.yml` gains the frontend suite for the first time, so master and a PR measure the
same thing; a badge that disagreed with the PR number for wiring reasons would be worse than the
minute it costs.

### 15.3 What was deliberately left

- **`CubeDaviCampaign` (286 uncovered, still the largest single file)** and `CubeEfficientCampaign`
  (102): both take a concrete `AdaptiveBackend`. Constructing ILGPU under coverage instrumentation is
  the documented 37-CI-failure hazard. Reaching these needs the §12/§6 recommendation — extract the
  pure decision logic into a static, the way `BlockDudeCurriculum.Advance` was — which is a production
  change for a later milestone, not a test.
- **`CubeImitationCampaign.Evaluate`** (most of its remaining 79): an untrained net fails every greedy
  rollout, so each of 160 eval episodes pays a full 2,000-expansion A* at 11 net forwards per
  expansion. Needs eval budgets on the options record before it is affordable.
- **`BlockDudeLevelBench` (136), `FruitCakeSearchEval` (75), `FruitCakeAb`, `BlockDudeValueCalibration`**:
  §12.4's deliberate exclusions — they play real episodes or load checkpoints.
- **`VizServer`'s socket lifecycle**: never bind a port; it would collide with the owner's own `--viz`.
- **The 14 Canvas/Three.js/WebSocket renderers and 15 Angular components** (~8,200 TS lines): see §15.2.

### 15.4 Bugs the test-writing surfaced

Writing tests against code that had none is the cheapest bug-finding this repo has done. Nine real
defects fell out; one was fixed here, the rest are recorded with the reasoning rather than quietly
encoded as expected behaviour.

**Fixed in this PR** *(one-PR rule — a defect found while testing is in scope)*:

- **`StartupCheckpoint<T>.TryLoad` did not guard the loader**, while its sibling
  `RefreshingCheckpoint<T>` did. A checkpoint that *exists but cannot be read* — a truncated Git-LFS
  pointer in `models/` is enough — propagated straight out of `Initialize`, which faults
  `ModelStartupHostedService`; under .NET's default `BackgroundServiceExceptionBehavior.StopHost`
  **that takes the whole web host down at boot**. If the host survived, `Status` stayed `Loading`, so
  the lazy `Value` getter re-read and re-threw on *every request* — a 500 per click, against a class
  whose own doc comment promises it "turns a missing checkpoint into 'unavailable' rather than an
  exception". Now caught, with `Status = Failed` and an `Error` naming the real fault; `Initialize`
  no longer overwrites that message with "no checkpoint in the store", which would send a reader to
  entirely the wrong problem. `Failed` is terminal, so a known-bad checkpoint is not re-read per
  request.

**Recorded, not fixed** — each is either latent, arguably intended, or needs a decision:

| | |
|---|---|
| `SelfPlayCampaign` | **FIXED in M68.** A genuine **0.0 win rate was indistinguishable from "never evaluated"**: `Checkpoint` stores `IsNaN ? 0 : rate` and `Resume` maps a stored `0` back to `NaN`. A net that truly scores 0% vs random resumes as "unknown", which *disables the winRate signal in `MaybePromoteDifficulty`* — so the worst possible net is treated as an unmeasured one. |
| `TrainWindow.MeanAndReset` | **FIXED in M68.** Returned **0, not NaN, for an empty window**, so `SelfPlayCampaign.Evaluate` reports `policyLoss = valueLoss = 0.0000` before any batch has run — indistinguishable from a collapsed loss. The BlockDude campaigns get this right with NaN; §15 argues at length that NaN and 0 are different facts. Also affects `CubeImitationCampaign`'s `ce`/`acc`/`huber`. |
| `CubeImitationCampaign.Resume` | Unconditionally calls `CubeSolver.WarmUp()`, building the Kociemba tables (multi-second on the first call in a process) even for a run that will never reach the oracle. Shared static tables mean only the first test in the assembly pays it, but it is what stops this campaign being properly unit-testable in isolation. |
| `Kociemba K_CubieCube.multiply` | **Only multiplies corners** — `// edgeMultiply(b);` is commented out. Private and unused today, so nothing is broken; the name lies, which is how it will eventually be used wrongly. |
| `Kociemba setPruning` | An **AND, not an assignment**: an entry can be written exactly once, only from the pre-filled `-1`. The table builder happens to respect this (`== 0x0f` guard), so it is pinned as the designed contract — but it is a fragile interface. Its two nibble halves are also inconsistently guarded (`unchecked` on the even branch only), which would throw under `<CheckForOverflowUnderflow>`. |
| `Tools.randomCube` | Unseeded `new Random()`. Tests assert only that the result verifies as valid, never a specific cube. |
| `game-2048-classic.ts` | **FIXED in M68 — and this entry originally blamed the wrong file.** The cap in `game-2048-logic.ts` (`Math.min(pending + 1, 15)`) is a faithful mirror of the server (`Game2048.cs:76`, commented *cap exponent at 4 bits*) and is **load-bearing**: `NTuple2048Agent` packs four bits per cell into a 16^4 table index **unmasked**, so an exponent of 16 is an index-out-of-range or a silently corrupted trained table; the expectimax transposition key packs the same way. `ClassicEngine` was the one that disagreed, merging without the cap — and it is the **only** engine on the browser path. Reachable in three clicks: draw two 32768s in edit mode, press Solve, and the server returns a capped board the client replays as 65536, failing the playback checksum. Fixed by mirroring the cap; pinned on both sides (nothing had pinned it in C# either). |
| `snake-logic.ts` | `SnakeGame.reset()` with `size < 3` walks the body off the board (negative cells, body longer than the board). Board size is a visitor setting; if the UI can offer < 3 this is a live crash path. |
| TS checkpoint readers | The four dueling-Q readers **accept any version byte** — only `>= 2` gates the noisy flag, so version 0 or 99 parses. Looks like an oversight rather than intent; left untested pending a decision. |

### 15.5 §14 revisited: ~80% is where this lands, and it is now nearly there

§14 priced three options and recommended (2) *move the target to ~80%* or (3) *a multi-milestone arc*,
against (1) *exclude `tools/**`*. M65 is evidence for (2): **79.55% combined**, reached in one milestone
without excluding a single line from the denominator.

What is left between here and 90% is ~3,300 C# lines, and it is no longer a matter of writing more of
the same tests. It is concentrated in `CubeDaviCampaign`, the Lab's episode-playing `Run` bodies, and
the campaign `TrainChunk`s — each of which needs either a production seam or a slow test, and §12.7 has
already measured what slow-and-shallow buys. **The recommendation is now firmly (2): declare ~80% the
target, treat it as met, and make any further rise a by-product of seams that are worth having anyway.**
Excluding `tools/**` would still move the number ~7 points without covering a line, and is still wrong
for the same reason.

---

## 16. M66 — the `.pg` union, properly *(2026-09-18)*

§1's first goal was that each `.pg` line should carry the **union** of hits from its generated C# and
its generated TypeScript. M63 delivered the C# half and built the TypeScript half; §13 then retired the
TypeScript half on a measurement that was real but meant something other than what it was read to mean.
M66 puts it back, and it works.

| | C# only | **union** |
|---|---|---|
| `blockdude_solver.pg` | 358/360 | 358/360 |
| `chess_solver.pg` | 499/509 | 499/509 |
| `crazyfruits_solver.pg` | 617/622 | 617/622 |
| `draughts_solver.pg` | 385/413 | 385/413 |
| `fruitcake_solver.pg` | 315/322 | **317/322** |
| `lunarlockout_solver.pg` | 177/179 | 177/179 |
| `mountaincar_solver.pg` | 40/69 | **69/69** |
| `snake_solver.pg` | 374/496 | **493/496** |
| `tetris_solver.pg` | 729/749 | **734/749** |
| **total** | **3,494/3,719 = 93.95%** | **3,649/3,719 = 98.12%** |

> **§17 corrects the denominator below.** These figures were measured with the interim
> `pg_coverage_remap.mjs`, whose denominator was the lines it happened to find statements for.
> The 155-line gain is unchanged and robust; the percentages are not.

**155 `.pg` lines are covered by the browser and by nothing else.** Repo-wide that is
**78.70% → 79.70%**, and — this is the part that matters — **the denominator does not move**. These are
not new lines being added to be counted; they are lines the C# report already listed and already scored
as uncovered, which a second execution path turns out to reach. Frontend suite: 144 → **202 tests**.

The clearest single result is `mountaincar_solver.pg` going from **40/69 to 69/69**. Its remaining 29
lines were the entire client-side policy net, which the training path cannot reach by construction.

### 16.1 The overlay is sound — the two targets agree exactly on `.pg` line numbers

The precondition for overlaying anything is that both targets mean the same thing by "`.pg` line 271".
Verified exhaustively rather than assumed, by extracting the set of `.pg` lines each target emits origin
info for, in all nine solvers:

| solver | `.pg` LOC | C# `#line` distinct | TS map distinct | in both | C#-only | TS-only |
|---|---|---|---|---|---|---|
| blockdude | 656 | 381 | 381 | 381 | 0 | 0 |
| chess | 926 | 548 | 548 | 548 | 0 | 0 |
| crazyfruits | 1104 | 657 | 657 | 657 | 0 | 0 |
| draughts | 754 | 445 | 445 | 445 | 0 | 0 |
| fruitcake | 624 | 341 | 341 | 341 | 0 | 0 |
| lunarlockout | 342 | 190 | 190 | 190 | 0 | 0 |
| mountaincar | 153 | 74 | 74 | 74 | 0 | 0 |
| snake | 831 | 516 | 516 | 516 | 0 | 0 |
| tetris | 1447 | 797 | 797 | 797 | 0 | 0 |
| **total** | | **3,949** | **3,949** | **3,949** | **0** | **0** |

**Symmetric difference is zero in every solver.** That is not a coincidence to be re-checked each release —
it follows from the architecture: one emitter and one origin table, rendered by two backends. Worked
example: `fn reachableFreeSpace(...)` at `snake_solver.pg:271` appears as `#line 271` at
`snake_solver.cs:583` and as a mapping to `snake_solver.ts:271`.

Two limits on what the overlay may claim:

- **Line granularity only.** The C# plugin manifest declares `"column": 0`, so there is no column fidelity
  to union even though the TS map carries generated columns.
- **The denominator is the *mapped* set, not the file's raw line count.** Type declarations (`class
  PgSnakeEnv` at `snake_solver.pg:92`, and every other `class`/`record` head) sit under `#line hidden` in
  C# and have no mapping in the TS map — neither target can ever cover them. Snake is 516 mappable lines
  of 831 physical. A tool that used raw LOC would report a permanent, meaningless shortfall.

### 16.2 How a generated file is identified — and why `pgconfig.json` is not the answer

`pgconfig.json` looks like the source of truth and is not. It is a **routing override table for the
TypeScript target only**: nine `include` entries, every one `"target": "typescript"`, and **no C# output
path at all** — the C# location comes from MSBuild (`PolyglotOutDir` =
`obj/<Config>/<TFM>/polyglot/`). A tool keyed on `pgconfig.json` would silently cover only half the
problem. It also omits `__polyglot_prelude.cs`, which has no `.pg` origin and must be skipped.

**The robust markers are intrinsic to the output**, and the plugin manifests make them contractual rather
than incidental (`plugins/csharp/polyglot-plugin.json` declares
`originMapping: {style:"directive", line:"#line $n \"$f\""}`; `plugins/typescript` declares
`{style:"sourceMapV3", sidecarExtension:".map", footer:"//# sourceMappingURL=$f"}`):

| target | identify a generated file by | catches | misses |
|---|---|---|---|
| C# | any `#line N "….pg"` | all 9 solvers | `__polyglot_prelude.cs` — correctly, it has no origin |
| TypeScript | `//# sourceMappingURL=` footer **and** a `.ts.map` beside it | all 9 twins | — |

The TypeScript marker matters for a reason a glob would get wrong: `mountaincar_solver.spec.ts` is a
**hand-written, git-tracked test** that any naive `*_solver*.ts` pattern swallows. The sidecar test
excludes it. (`docs/prd/polyglot-pilot/fruitcake_solver.pg` is a similar decoy on the source side — a
tracked documentation copy that generates nothing and is absent from `pgconfig.json`.)

### 16.3 A third target would be invisible today

Polyglot ships four plugins: `csharp`, `typescript`, `php`, `python`. **The `php` and `python` manifests
have no `originMapping` key at all** — they emit no `#line`, no source map, nothing tying output back to
`.pg`. Enabling either today would produce coverage that the overlay could not attribute, and the `.pg`
denominator would not change, so the failure would be silent rather than loud.

So the overlay tool should **read `originMapping.style` and dispatch on it** (`"directive"` →
scan pragmas; `"sourceMapV3"` → decode the sidecar) rather than hard-coding two languages, and should
**warn when a declared target has no `originMapping`** — that is the condition under which a new target's
coverage would vanish without trace.

### 16.4 Why the TypeScript report must be translated before upload

Every generated artefact is gitignored — `obj/` for the C#, and an explicit rule for
`src/RLDemo.Web/ClientApp/src/app/**/*_solver.ts(.map)`. Only the nine `.pg` sources are in
`git ls-files`. The coverage service resolves a report path by suffix-matching against `git ls-files`, so
**a report naming a `.ts` twin resolves to nothing and is dropped without an error**.

The C# side already satisfies this for free: Roslyn carries `#line` into the PDB, coverlet reads it, and
the emitted cobertura already names
`src/MintPlayer.AI.ReinforcementLearning.Environments/Snake/polyglot/snake_solver.pg` — a repo-relative
`.pg` path. (`coverlet.runsettings` documents why `ExcludeByFile`'s `**/obj/**/*.cs` no longer matches
these files, and `UseSourceLink=false` exists for the same suffix-matching reason.)

The TypeScript side has the mapping infrastructure but **no coverage producer pointed at it today**, and
its raw report would name gitignored `.ts` paths. That translation step is what
`tools/pg_coverage_remap.mjs` exists to do, and it is why the twins cannot simply be added to
`coverageInclude` and uploaded like the hand-written modules in §15.2.

### 16.5 §13 and §15.2 were wrong about the twins, and this is the correction

Two earlier sections retired the TypeScript→`.pg` projection. Both are superseded. The reasoning is
worth keeping because the mistake is an easy one to repeat.

**What §13 said:** the twins are transpiled from the same `.pg` the C# already covers at 93.9%, so
uploading both "double-counts one source file" for +0.1pp at a cost of +57s.

**Why that is wrong.** It *is* one source file, but the union is over **execution paths, not reports**.
The C# and the TypeScript are the same source compiled for two different callers:

| | runs | exercises |
|---|---|---|
| generated **C#** | the training agent | env dynamics, reward, observation encoding |
| generated **TypeScript** | a visitor playing in the browser | the serving-side net forward, the look-ahead planner, human-input micro-moves, rotation physics |

A `.pg` line only a player reaches is genuinely uncovered in the C# report, and the union is the only
thing that can see it. Measured: **137 of the 225 `.pg` lines C# never reaches are on real browser call
paths**, traced to their call sites.

**What the +0.1pp measurement actually measured.** It was taken when the repo contained exactly ONE
frontend spec — `mountaincar_solver.spec.ts` — and that spec constructs no net, so it reached none of
the 29 lines that matter in its own file. The number was real; it measured **the absence of
browser-side tests**, not the value of the union. Retiring the mechanism on the strength of it was the
wrong conclusion drawn from a correct measurement, and that is the part worth remembering: a
near-zero delta from a pipeline with nothing feeding it says nothing about the pipeline.

**What §15.2 got right and kept:** the hand-written ClientApp modules are a separate concern and are
still uploaded under their own `.ts` paths. What it got wrong was excluding the twins from
`coverageInclude` entirely, which made `pg_coverage_remap.mjs` a silent no-op — the tool was audited
as correct, and was producing nothing purely because nothing was feeding it.

### 16.6 What the two untested features actually were

The percentage is the least interesting part of this. The 137 lines are not scattered noise; they are
**two whole shipped features with zero tests on either side**:

- **The snake receding-horizon beam planner** (~101 lines) — `chooseActionSearch`, `leafScoreSearch`,
  `pruneBeam`'s manual top-k, `freeSpaceAhead`'s flood fill, and the anti-fragmentation
  space-ratio term that M34 measured as *the* biggest strength lever (~81 food, +60% over the
  plateau). Subtle, hand-rolled, and until now never asserted anywhere.
- **The mountaincar policy net** (29 lines) — `PgMlpNet.forward`/`linear`/`tanhv` plus `chooseAction`.
  It exists *only* so the browser can run the PPO policy client-side; C# training uses the real tensor
  library. A transposed weight index or a tanh applied to the output layer would have shipped silently.

Two smaller ones: tetris's human-play micro-move locking and top-out (training only uses the macro
`applyPlacement`), and fruitcake's angular-velocity integration (training constructs the world with
rotation off; all three browser call sites construct it on).

### 16.7 What is deliberately NOT covered, and one thing to delete instead

- **The draughts MLP tier (~22 lines) is a trap.** It looks like net code the browser obviously runs,
  and it does not: `draughts-net.ts:127` returns `PgDraughtsNet.withConv(...)`, and the covered line
  `draughts_solver.pg:582` short-circuits to the conv path. Testing it would cover a branch that
  **ships dead**. **DELETED in M68** — and it was deader than this said: the Lab's `--arch mlp` writes
  kind `selfplay-pv`, which `draughts-net.ts` hard-rejects, so the tier was the second half of a path
  whose first half was never built.
- **~36 lines cannot be reached by anyone** and should leave the denominator or the codebase.
  `chess_solver.pg:102-105 clone()` — orphaned by the functional `makeMove` — **deleted in M68**.
  The rest are defensive MCTS fallbacks ("all visit counts zero"), unused accessors, and ~5
  declaration-line artifacts where the body is covered but the signature line holds a sequence point
  that never executes.

  > **Correction.** This paragraph also named `tetris_solver.pg:1029-1032 reachableMask()` as having
  > "no caller in any `.pg` and none in ClientApp". **That is wrong and it was nearly deleted on the
  > strength of it.** It is live: `TetrisBoard.ReachableMask()` (`TetrisBoard.cs:82-84`) wraps it and
  > `TetrisLab.cs:286` calls that under `--reach`. The grep that "proved" it dead searched the `.pg`
  > spelling; the C# facade renames it. A `.pg` method can always be reached from generated C# through
  > a facade under another name — check the facade, not just the `.pg`, before calling anything dead.
- **~52 lines are reachable only by unit-testing the twin directly**, with no browser path — snake's
  `safeMask` shield (the director hard-codes it off), snake's unused masked-greedy `chooseAction`, and
  three tetris training setters. Covering them would be honest only under "unit-test reachability",
  which is a weaker claim than this union is making. Left alone.

### 16.8 How the three reports fit together

One `dotnet test` and one `vitest` run now produce **three** reports, merged by the service under one
`(repo, sha, runId, runAttempt)` with max semantics:

| report | keyed by | produced by | covers |
|---|---|---|---|
| C# | `.pg` **and** `.cs` paths | coverlet, via `#line` in the PDB | the whole backend, incl. the `.pg` training path |
| frontend | `.ts` paths | vitest cobertura → `reroot_frontend_coverage.mjs` | hand-written ClientApp modules |
| **`.pg` union** | `.pg` paths | vitest istanbul JSON → `pg_coverage_remap.mjs` | the twins' browser path, projected through the v3 source maps |

Three details that make it work, each of which would fail *silently* if got wrong:

1. **The twins are in `coverageInclude` but stripped from the frontend report.** They must be measured
   (so the remap has input) but must not upload under `.ts` paths — those are gitignored, so the
   service would drop them unmatched, and keeping them would double-count against the `.pg` report.
   `reroot_frontend_coverage.mjs` drops any `_solver.ts` class and says how many.
2. **Exactly one upload carries `finish: true`**, and it is the last one. The `.pg` report now holds it.
3. **The remap warns and exits 0 when it finds no twin**, rather than failing the build — no twin spec
   having run is a legitimate state, and the upload's `hashFiles` guard already makes it a no-op.

Hardening applied to `pg_coverage_remap.mjs` while it was out of service (none of which changes a
number): `sourceRoot` is now honoured (Polyglot emits `""`, so ignoring it was harmless *today* and
would have silently voided the whole report if that ever changed); `<source>` is `.` rather than the
absolute agent path; the summary percentage no longer multiplies a string by 100.

**Known gap, stated rather than assumed:** the `.pg` report carries **no branch data**
(`branches-valid="0"`). If the service ever merged branch rates by average rather than by max, a 0/0
report could dilute the branch number. Unverified — worth one check against a real run before the
branch figure is trusted.

> **Both halves of that resolved in §17, and both the other way than feared.** The `.pg` report now
> carries **272/886** count-only conditions, because M67 pins `--out-format cobertura` (§17.3). And the
> dilution worry was unfounded regardless: Polyglot PR #72's SP7 read the ingest and found a branch-less
> report **cannot** dilute an existing branch number — the merge skips the block entirely. Checking the
> source beat the "worth one CI run" this paragraph settled for.

---

## 17. M67 — adopting `polyglot coverage remap`, and the denominator correction *(2026-09-19)*

`MintPlayer.Polyglot.MSBuild` **0.10.1** ships `polyglot coverage remap`
([Polyglot#71](https://github.com/MintPlayer/MintPlayer.Polyglot/issues/71) /
[PR #72](https://github.com/MintPlayer/MintPlayer.Polyglot/pull/72)). `tools/pg_coverage_remap.mjs` is
deleted; both workflows now call the shipped CLI.

### 17.1 The gain holds; the denominator does not

Running the official tool on the same lcov the interim tool consumed:

| | interim tool | **`polyglot coverage remap` 0.10.1** |
|---|---|---|
| `.pg` files in the report | 7 | **9** |
| `.pg` lines covered by the browser | 811 | **811** |
| `.pg` lines in the denominator | 1,801 | **3,949** |
| lines the browser covers and C# never reaches | 155 | **155** |

**The 155 is unchanged**, which is the number that mattered — it is the measurement that justified the
whole union, and it survives being recomputed by an independent implementation. Everything else moves.

| | C# report's denominator (3,719) | **the mappable denominator (3,949)** |
|---|---|---|
| `.pg`, C# only | 93.95% | **88.48%** |
| `.pg`, union | 98.12% | **92.40%** |

**Measured on the service** (the figures that actually matter — a local C#-only total omits the
hand-written frontend leg, which is ~636 of ~638 lines and so flatters nothing but is simply absent):

| commit | covered / coverable | repo |
|---|---|---|
| M65 `1f88737` | 12,816 / 16,114 | 79.53% |
| M66 `17221d1` | 12,971 / 16,114 | **80.50%** |
| M67 `ee7be01` | 12,971 / 16,344 | **79.36%** |

So **M66's +1.00pp was real** — the service confirms +0.97pp — and **M67 gives back 1.14pp** by
correcting the denominator. Net across both: 79.53% → 79.36%, essentially flat, while gaining 155
genuinely-covered lines and a denominator that means something.

> An earlier draft of this section reported M67 as "78.70% → 78.54%, down 0.16pp". That compared two
> *local, C#-only* totals and never included the frontend leg, so it described no number anyone sees.
> The real M67 effect is **−1.14pp**, and the real M66 effect was **+0.97pp**, not the artifact I
> implied.

### 17.2 Why the denominator grew by 230 lines

Both targets' origin data map **3,949** `.pg` lines — verified identically from the C# `#line` pragmas and
the TypeScript `.ts.map` (§16.1: symmetric difference zero). But the C# *coverage report* only declares
**3,719** coverable lines, because Roslyn emits no sequence point for ~230 of them — declaration-only
lines that carry a pragma but no IL.

The interim tool never surfaced this: it took its denominator from whatever the istanbul report happened
to contain. The official tool takes it from the **origin data**, which is the rule §4.2 already stated and
which the tool's own documentation is explicit about — *"the denominator is the mapped set, not the
file"*, and the file set comes from the sidecars, never from the report.

**Those 230 lines belong in the denominator**, because a `.pg` line is coverable when *any* target can
execute it, and the TypeScript twin can execute these. Our previous figure was flattered by Roslyn's
narrower view of the same source. The service merges line *sets* per filename, so the merged denominator
becomes 3,949 whichever leg supplies it.

### 17.3 Two wiring decisions

**`--out-format cobertura`, though the input is lcov — for branch data, not for format matching.**
The tool defaults to handing back the input format. We override it because **the remap emits branch data
count-only, and only cobertura, clover and JaCoCo can express a count** — lcov and istanbul output are
line-only. Measured both ways on this repo's report: cobertura out carries **272/886** conditions, lcov
out carries **zero**.

> **Correction.** An earlier version of this section justified the override by the ingest's `BranchFormat`
> stamping — that reports arriving in a second format would have their edges silently discarded, so both
> legs had to match. **That is no longer true, and was already fixed when this was written.**
> [MintPlayer.Spark#420](https://github.com/MintPlayer/MintPlayer.Spark/issues/420) ("format-agnostic,
> order-independent branch merge + istanbul/clover parsers") closed on 2026-09-18, *before* Polyglot PR
> #72 merged. The claim came from SP7's reading of the ingest as it stood during that spike, and I
> carried it forward without re-checking the issue — the decision happened to be right, the reasoning
> was stale.
>
> **What replaced it is stronger than a fix.** The service now stores coverage in a **generic internal
> model** — not lcov, not cobertura — and the database has been migrated. Upload order was the explicit
> design goal: it must not matter which report arrives first, whatever format each one is in. So the
> wire format now decides only what a report can **express**, never how it merges, and this repo is free
> to change either leg's format without coordinating the other.

**The CLI is located by globbing the restored package** (`tools/<rid>/polyglot`) rather than by a
hardcoded version, so it cannot drift out of step with the `PackageReference`. It fails loudly when the
binary is absent, because the alternative — an empty report — reads as "nothing was covered".

### 17.4 Not done, deliberately

- **The C# leg is not remapped.** The docs suggest running it anyway to validate and to complete the file
  set, but our C# report already names all nine `.pg` files, and the 230 extra mappable lines arrive via
  the TypeScript leg regardless, since the service unions line sets. It would cost a CI step to change
  nothing.
- **`--branch-arms` stays off.** It is safe only for a single-target consumer; we overlay two.
- **`reroot_frontend_coverage.mjs` stays.** It serves the hand-written ClientApp report, which is a
  consumer-side path convention and was never Polyglot's concern (§16 / Polyglot PRD §5).

---

## 18. M68 — fixing what the tests found *(2026-09-19)*

§15.4 recorded eight defects that writing tests surfaced but did not fix. This milestone closes the
ones worth closing, and — more usefully — **corrects two entries that were wrong**.

### 18.1 The 2048 exponent cap: §15.4 blamed the wrong file

The entry said `game-2048-logic.ts` "silently eats a tile" by saturating at `Math.min(pending + 1, 15)`.
That file is a **faithful mirror of the server** (`Game2048.cs:76`, commented *cap exponent at 4 bits*),
and the cap is **load-bearing in a way the entry missed**:

- `NTuple2048Agent.cs:92` packs four bits per cell into a **16⁴ table index, unmasked** — exponent 16 is
  an `IndexOutOfRangeException`, or in an earlier cell a silent alias that corrupts a trained table.
- `Expectimax2048.cs:180` packs the transposition key the same way, masked — exponent 16 aliases to 0.
- `Env2048.cs:82` normalises `exponent / 15f` against a declared `BoxSpace(0f, 1f, 16)`.

**`ClassicEngine` was the one that disagreed**, merging without the cap — and it is the *only* engine on
the browser path, for both human play and AI replay. `applyMove` is not on any production path at all.

**Reachable in three clicks:** edit mode cycles a cell to 32768 (`game-2048.ts:107`), the controller
accepts it (`Game2048Controller.cs:104`), so drawing two adjacent 32768s and pressing Solve makes the
server return a capped board that the client replays as 65536 — the playback checksum then fails and
every subsequent replayed state is wrong.

Fixed by mirroring the cap. Both headers corrected: `game-2048-classic.ts` claimed "identical merge
results" while diverging, and `game-2048-logic.ts` read as an arithmetic accident rather than a storage
constraint. **Pinned on both sides — nothing had pinned it in C# either**, which is how the two drifted
apart unnoticed. The TS cross-check now includes the saturation fixture it had deliberately avoided.

### 18.2 Zero versus unknown, in two places

**`SelfPlayCampaign`**: sidecar format **v1 → v2**; the metric is stored verbatim, NaN included. The
subtlety is *where* the compatibility branch lives. `CampaignProgressState` is shared, and
`CubeImitationCampaign` uses the same `LastMetric` slot for an **exact solve counter** — so a blanket
"v1 zero means NaN" translation at the format layer would turn an old cube checkpoint with 0 solves into
`(long)double.NaN` = `long.MinValue`. Instead `CampaignProgress` exposes the version it read and only
self-play branches on it. `CheckpointFormat.ReadHeader` accepts `1..max`, so existing sidecars still load.

**`TrainWindow.MeanAndReset`** returns NaN for an empty window. All four callers were checked and are
formatting-only; `CampaignEval.Metrics` is consumed in exactly one place (`CampaignCli`), which writes it
to a CSV cell where `NaN` is the standard missing-value token.

### 18.3 A crash that fell out of the second fix

`WriteManifest` serialises the tier win rate with a default `JsonSerializerOptions`, **and NaN is not
valid JSON**. The ladder promotes a baseline tier *unconditionally* on the first checkpoint, so a run
that checkpoints before it ever evaluates threw — losing the net, the optimizer and the progress sidecar,
not just the manifest. Every existing ladder test called `Evaluate()` before `Checkpoint()`, which is
exactly why none of them caught it.

Written as `null`, not `0`: zero is a measured result, and writing it would recreate the conflation
§18.2 just removed. `null` is also already what the browser expects — `chess-net.ts:53` maps a
non-number to `undefined` and the tier label omits the suffix.

### 18.4 Dead code: two of three deleted, and the third was not dead

`chess_solver.pg`'s `clone()` (orphaned by the functional `makeMove`) and the **draughts MLP tier** are
deleted. Both are `.pg`-only edits — the C# and TypeScript twins are gitignored build outputs, so one
tracked file changes per deletion.

The draughts tier was deader than §16.7 said: the Lab's `--arch mlp` writes kind `selfplay-pv`, which
`draughts-net.ts` hard-rejects, so it was the second half of a path whose first half was never built.

> **`tetris_solver.pg`'s `reachableMask()` is NOT dead, and §16.7 was wrong to say so.** It is reached
> through a facade that renames it: `TetrisBoard.ReachableMask()` → `TetrisLab.cs:286`, under `--reach`.
> The grep that "proved" it dead searched the `.pg` spelling. **A `.pg` method can always be reached from
> generated C# through a facade under another name** — this one was a step away from being deleted on the
> strength of a bad grep.

### 18.5 The remaining six — all closed

They were recorded as "needing a decision rather than a patch". On inspection every one was decidable
from the code; none needed a product call.

| defect | fix | the reasoning that decided it |
|---|---|---|
| Kociemba warm-up in `CubeImitationCampaign.Resume` | moved to the first `TrainChunk`, flag-guarded | `CubeSolver.WarmUp` only **eagerly triggers** CLR static init that happens lazily on first oracle use anyway, so moving it is behaviour-neutral for a real run. It was charging multi-seconds to every caller that merely wanted to inspect or checkpoint the campaign — the one thing stopping it being unit-testable in isolation. |
| `K_CubieCube.multiply` | deleted | Private, uncalled, and the body was `cornerMultiply(b);` with `// edgeMultiply(b);` commented out. The name lied, so anything that *started* calling it would have silently multiplied half a cube. Callers already use `cornerMultiply`/`edgeMultiply` explicitly. |
| `setPruning`'s asymmetric `unchecked` | `unchecked` on both halves | The odd branch (`0x0f \| (value << 4)`) exceeds `sbyte` for `value >= 8`. It compiles only because the project does not enable `<CheckForOverflowUnderflow>` — so the two halves behave differently the moment anything does. |
| `Tools.randomCube` unseeded | added a `Random`-taking **overload** | Changing the existing signature would be breaking. An overload gives reproducibility to a test or benchmark that needs the same scramble twice, without touching callers. |
| `SnakeGame` with `size < 3` | throws `RangeError` | `reset()` seeds a three-cell snake by walking the head column down by two, so a narrower board yields negative cells and a body longer than the board — a corrupt game, not an error. Measured: `snake.ts` clamps to `MIN_SIZE = 6`, so this is unreachable from the UI and the guard covers direct construction only. |
| dueling-Q readers accept any version | validate `1..2` | **The least cosmetic of the six.** v2 added the `noisy` flag byte immediately after the header, so a reader that treats an unknown version as v1 does **not** fail — it shifts every subsequent float by one byte and returns a net of plausible-looking garbage. Shipped checkpoints measured as v1 (`snake-net.ckpt`) and v2 (crazyfruits, fruitcake, tetris), so the range rejects nothing that exists. |

Tests added for the two with observable behaviour (the snake guard, the version rejection). The rest are
deletions or internal.

**§15.4 is now fully discharged**: of the ten defects that writing tests surfaced, one was fixed on the
spot (`StartupCheckpoint`), one turned out to be misattributed (2048), one turned out not to be a defect
at all (`reachableMask` was live), and the remaining seven are fixed.

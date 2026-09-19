# PRD — Polyglot-owned coverage attribution (`.pg` overlay across all targets)

*2026-09-18 · handoff document · written in `MintPlayer.AI` because writes into
`C:\Repos\MintPlayer.Polyglot` are hook-blocked; the Polyglot issue links here.*

## 1. Goal, and why this belongs in Polyglot

One `.pg` source compiles to several targets. Each target's test suite covers a **different part of the
same source**, because each target has a different caller — measured in `MintPlayer.AI`, the generated
C# runs the training agent while the generated TypeScript runs a visitor playing in the browser, and
**155 `.pg` lines are reachable only from the latter**. A `.pg` line should count as covered when *any*
target reached it.

The C# half of that works today and needs no tooling. The TypeScript half currently works through
`MintPlayer.AI/tools/pg_coverage_remap.mjs` — a ~150-line interim tool in the **consuming** repo. That
is the wrong home, for three reasons that are about ownership rather than tidiness:

1. **The knowledge is the compiler's.** `originMapping.style`, the `#line $n "$f"` template, the `.map`
   sidecar convention and the footer syntax are all declared in `plugins/*/polyglot-plugin.json`. A
   consumer-side tool must duplicate that vocabulary, and will drift from it silently.
2. **It is N×M in the wrong direction.** Coverage attribution is a per-target problem. Solving it in
   each consuming repo is *consumers × targets*; solving it in the compiler is *1 × targets*.
3. **Two of four targets have no attribution at all today.** `php` and `python` declare no
   `originMapping`, so their coverage cannot be attributed to `.pg` even in principle. Whoever adds
   origin info to the Python plugin should add the Python coverage story in the same commit, in this
   repo. That cannot happen while the reader lives somewhere else.

Three deliverables, matching the three scopes requested:

1. **Source mapping between `.pg` and each target** — close the `php`/`python` gap (§7 PG-C1).
2. **Overlaying the `.pg`-mapped reports** — one `.pg`-keyed report per target, merged downstream
   (§5 is emphatic about what "merging" does and does not mean here).
3. **Automate as much as possible** — one CI line per consumer, and an MSBuild hook only if it earns
   its keep (§7 PG-C6, gated by SP3).

**Status of the consuming work:** `MintPlayer.AI` PR #54 stays open with the interim tool in place. It
is not blocked on this, and this is not blocked on it.

## 2. What is already true (and reframes the work)

Three facts, verified in the Polyglot source rather than assumed. Each one shrinks the job.

**2.1 Origin recording is already target-neutral.** `MintPlayer.Polyglot.Core/include/mintplayer/polyglot/backend_spec.hpp`
defines `OriginMapping` with `recordsOrigins()`, and its own comment says it plainly: *"Both sinks are
fed by the same hook in `EmitterBase::line()`; only where the result GOES differs."* The emitter
collects `OriginRecord{outputLine, fileId, sourceLine}` for **every** backend; `style` chooses only the
serialiser.

> **So Python and PHP are not missing origin data — they are missing a sink declaration.** Giving them
> origin mapping is a manifest entry plus reuse of the existing sourcemap serialiser. It is not an
> emitter change, and nothing in the IR or the lowering has to move.

**2.2 There are two mechanisms, not four.** The v3 source-map format is language-neutral: a
line/column → line/column relation with a `sources` array. Nothing in it is JavaScript-specific except
the conventional trailer. Polyglot's generator (`sourcemap.hpp`) is already pure computation over
`(outputLine, fileId, sourceLine)` triples. So:

| style | who consumes the origin info | tool needed? | targets |
|---|---|---|---|
| `directive` | the **downstream compiler**, which rewrites its own debug metadata | **none** | C# |
| `sourceMapV3` | nobody — an out-of-band sidecar | **a post-processor** | TypeScript, and Python/PHP once declared |

The C# case deserves emphasis because it is the reason this is small: `#line` is consumed *by Roslyn*,
which writes the `.pg` into the PDB as the document. By the time coverlet instruments IL there is no
`.cs` coordinate left to remap. **Attribution happens before the coverage tool exists.** Any target
whose compiler honours such a pragma needs no tooling at all, forever.

**2.3 Granularity is line-level by design, and that is fixed.** `sourcemap.hpp`: *"LINE GRANULARITY
ONLY (PRD N1). Every mapping segment points at column 0 of both sides."* Column fidelity would mean
threading positions through the expression-rule interpreter, which returns bare strings. Every design
decision below assumes line granularity and must degrade gracefully, not pretend otherwise.

## 3. The architectural constraint: a language must be addable without touching the engine

Polyglot's backends used to be C++. They were moved to declarative JSON published as npm packages
precisely so that **the set of supported languages is not compiled into the CLI** — `pgconfig.json`
names the targets, the CLI resolves and downloads the plugin packages, and the packages carry the
transpile instructions. Adding a language is a package, not a release of the compiler.

**Coverage must obey the same rule, or it silently reintroduces the problem.** An engine with a
hardcoded `switch` over "istanbul JSON for TypeScript, `coverage json` for Python, Clover for PHP" is a
compiled-in list of supported languages wearing a different hat: a new target would transpile fine and
then be invisible to coverage until someone released the tool.

### The fix: two orthogonal axes

The mistake in an earlier draft of this PRD was to put a `reportFormat` in the plugin manifest. **A
plugin cannot know what report format its consumer emits** — that is a per-repo CI choice. This repo
alone could reasonably run cobertura for C#, lcov for TypeScript, and `coverage json` for Python, and
change any of them tomorrow without touching Polyglot.

So there are two axes, and they do not interact:

| axis | varies with | declared by | determines |
|---|---|---|---|
| **origin** | the **language** | the **plugin** (`originMapping`) | *how to project* generated lines onto `.pg` lines |
| **format** | the **consumer's toolchain** | the **invocation** (`--format`, or sniffed) | *how to read* the report, and how to write the result |

`originMapping` stays exactly as it is. Nothing about coverage formats belongs in the manifest.

**This preserves the property that matters, and strengthens it.** Format readers are
*language-independent*: a new target declares `originMapping` and immediately reuses every reader that
already exists. Adding a language still needs **zero engine code** — and now for a better reason than
"they all happen to emit cobertura", which was never going to stay true.

### Why the reader set stays small

Formats are a small, stable, shared set — far fewer than languages, and several languages emit the same
one. Verified against real artefacts in this repo rather than from documentation:

| format | emitted by | line data | branch data |
|---|---|---|---|
| **cobertura** | coverlet, vitest, `coverage xml`, php-code-coverage | `<line number= hits=>` | `condition-coverage="100% (4/4)"` |
| **lcov** | vitest, `coverage lcov`, most JS/Python tooling | `DA:<line>,<hits>` | `BRDA:<line>,<block>,<branch>,<taken>` |
| **istanbul JSON** | vitest, nyc, jest | `statementMap` + `s` (statement-level) | `branchMap` + `b` |
| **clover** | php-code-coverage, some JS tooling | `<line num= count=>` | `<line type="cond" truecount= falsecount=>` |

Both of this repo's current reports were checked directly: vitest's lcov carries **1,176 `BRDA:`
records**, exactly matching its cobertura's `branches-valid="1176"`. **The same data, differently
spelled.**

That corrects a claim carried from the earlier coverage PRD, that *"lcov is line-only and cannot be
remapped faithfully."* It is line-keyed, which is all a line-granular map needs, and it carries branch
data. The real reason istanbul JSON was chosen there was statement-level granularity — which §4.2
settles a better way (the denominator comes from the map's mapped-line set, not from the report).

**Minimum viable reader set: cobertura and lcov.** Those two cover every target ecosystem in play.
istanbul JSON and clover are additions, not prerequisites.

### The tool's interface follows from this

```
polyglot-coverage remap <report> --target <name> [--format <fmt>] [--out-format <fmt>] --out <path>
```

- `--format` sniffed by default (`TN:`/`SF:` ⇒ lcov; `<coverage>` ⇒ cobertura or clover; a JSON object
  of paths ⇒ istanbul), overridable because sniffing should never be the only option.
- `--out-format` **defaults to the input format.** The consumer's pipeline already consumes that
  format; handing back something else makes the tool a format converter it was never asked to be.
- `--target` names the plugin whose `originMapping` to use. For a `directive` target the command exits
  0 with *"needs no remap — its compiler attributes natively"*, which is the whole C# story.

### Plugin resolution should be reused, not reimplemented

The CLI already resolves plugins through `pgconfig.json` — `file:` refs, the in-box set, a verified
versioned cache, npm-registry downloads pinned by `pgconfig.lock.json`. The coverage tool must **not**
grow a second implementation of that. Two options, to settle in **SP6**:

- read the already-resolved plugin directory (`node_modules/`, the user cache, or the nupkg's
  `tools/<rid>/plugins/`) via a `--plugin-dir` flag; or
- add a small CLI introspection surface (`polyglot plugins --json`) printing each resolved target's
  `originMapping` and `coverage` blocks, and let the tool shell out to it.

The second keeps one resolver and one closed `style` vocabulary in C++ where they already live, while
leaving all report munging outside it. It is the only part of this that has a good case for touching
the CLI.

## 4. The rules the projection must follow

**4.1 Many-to-one is the normal case; combine with `max`.** Measured on the existing implementation:
tetris is 1,244+ generated lines onto **797** distinct `.pg` lines; across nine solvers, 3,949 mapped
`.pg` lines. A line is covered if **any** contributing statement ran.

`sum` is wrong, and not merely stylistically: coverage consumes covered-vs-not, so summing changes no
verdict but produces a number that *reads* like an execution count and is not one. There is already a
live example of that confusion — with `SingleHit=true` the MintPlayer coverage UI shows `cfFruitOf` as
`1×` while it actually executes ~368 million times, and a `4×` badge means *four contributing
statements*. If a genuine Σ is ever wanted it must be a separate field, never `hits`.

**4.2 The denominator is the MAPPED set, never the file's line count.** `class`/`record` heads sit
under `#line hidden` in C# and carry no mapping on the TS side — snake is **516 mappable lines of 831
physical**. A tool that used raw LOC would report a permanent, meaningless shortfall.

**4.3 One-to-many: credit every origin, and say so.** Polyglot emits one segment per output line today,
and the existing implementation takes "first segment on a line wins". That is the wrong default for a
generic tool: it would **silently** drop origins the moment a multi-segment map appeared — plausible
for Python, where a `.pg` block may collapse into a comprehension, and certain if a bundler map is ever
composed into the chain. Credit **all** segments, then combine per `.pg` line with `max`, and emit a
diagnostic count so the condition is visible.

**4.4 Branch coverage: project per-line, merge per-component, never per-arm.** Each istanbul
`branchMap[id].locations[]` carries a start line; mapping those through gives, per `.pg` line, *"k of n
arms taken"* — exactly what cobertura's `condition-coverage` expresses. What is **not** recoverable is
*arm identity*: at column 0, `if (a && b) … else …` compiles to arms whose start lines all map to one
`.pg` line. So the contract is a per-`.pg`-line `(coveredArms, totalArms)`, merged with `max` on each
component independently — monotone, never claims more covered than total, degrades gracefully. Do not
attempt arm-level union while `column: 0` holds.

This matters today: the C# side already reports branches natively on `.pg` (6,169/8,932 = 69.1%
rendered in the UI), while the TypeScript side currently contributes none.

**4.5 Prefer `sourcesContent` over path resolution.** Polyglot embeds it deliberately — *"the consumer
then needs no path resolution at all"*. The existing implementation ignores it and resolves `sources`
on disk, which is the entire class of bug that `sourceRoot` handling worries about. Treat
`sourcesContent` as the "this map is self-describing" signal and keep resolution as the fallback.

**4.6 Identify generated files by the contractual marker, not a glob.** The marker is *footer plus a
sibling sidecar*. A name glob is wrong for a reason already observed: `mountaincar_solver.spec.ts` is a
hand-written, git-tracked test that `*_solver*.ts` swallows. (The current implementation's
`endsWith('_solver.ts')` excludes it, but by accident.)

## 5. What must NOT be built here

**Merging is not Polyglot's job, and is already solved.** coverage.mintplayer.com merges reports
sharing `(repo, sha, runId, runAttempt)` with max semantics; that is what produced the measured
93.95% → 98.12% union. Polyglot's responsibility ends at *"emit one `.pg`-keyed report per target"*.
Whether a standalone N-report merger is worth offering for consumers with no such service is deferred
(§9) — it is a generic cobertura-merge problem with nothing Polyglot-specific in it.

**Report-path conventions are not Polyglot's job.** Re-rooting paths so a particular coverage service
can resolve them against `git ls-files` is consumer-side (`reroot_frontend_coverage.mjs` stays where it
is). The tool should emit repo-relative-from-a-stated-root paths and stop there.
## 6. Where the code should live — **`@mintplayer/polyglot-coverage`, an npm package**

Three homes were considered and argued from the repo, not from taste.

| | reaches MintPlayer.AI? | new-code cost | version skew |
|---|---|---|---|
| **(a)** a `polyglot coverage` C++ subcommand | **yes** — ships in the nupkg beside the exe | **highest** | none |
| **(b)** a script in `scripts/` | **no** — `csproj` packs only `build\**` | lowest | n/a |
| **(c)** a published npm package | **yes** | low | solvable — see below |

**(a) is rejected on implementation cost, and the repo's own precedent says so.** The C++ side has a
*lenient* JSON reader built for JSON-RPC (returns `Null` on malformed input — the wrong posture for
someone else's report file), **no VLQ decoder** (`sourcemap.hpp` exposes `vlqEncode` only), and **no XML
writer or escaping anywhere**. A remapper in C++ means writing all three from scratch, in the language
where they cost most, inside a binary whose selling point is *"self-contained native CLI; no extra SDK
or runtime"*. It would also have to be kept in CMake/`.vcxproj` parity (there is a gate leg for that
drift) and rebuilt across five RIDs.

Decisively, **the repo already drew this line**: `scripts/arm-trace-to-lcov.ps1` is a coverage
remapper — the compiler emits a raw trace, a *script* converts it to lcov. `coverage.ps1` already
rewrites cobertura paths; `verify-coverage-paths.ps1` already reads lcov and cobertura. Format
conversion is not the compiler's job here, by established practice.

**(b) is rejected on reach.** The consumer is in another repo, and `MintPlayer.Polyglot.MSBuild.csproj`
packs only `build\**` — a script in `scripts/` never arrives.

**(c) resolves the one real argument for (a).** Version skew between map producer and map consumer was
the strongest case for putting this in the CLI. It dissolves because **the plugin manifests are already
published as npm packages** — `plugins/typescript/package.json` → `@mintplayer/polyglot-target-typescript`,
`files: ["polyglot-plugin.json"]`. So the tool reads `originMapping` **from the installed plugin
package**: same source of truth, no duplicated `style` vocabulary, no CLI round-trip, and a
`--plugin-dir` escape hatch for the nupkg layout (`tools/<rid>/plugins/`).

The audience also fits: every consumer of this is a CI runner that already has Node (the TypeScript
case by construction; Python and PHP report-munging is language-agnostic). And publishing muscle exists
— four plugin packages and a VS Code extension already ship from this repo.

**The one thing to be honest about:** `package.json` currently says *"NX orchestration for the test gate
ONLY — the compiler itself is C++ with zero runtime deps."* A published package **with logic in it** is
a new category for this repo. That statement stays true of the *compiler*; this is a sibling tool, and
the README should say so explicitly rather than letting the claim quietly rot.

## 7. Milestones

**PG-C1 — `originMapping` for Python and PHP.** *(scope 1)* Manifest entries plus serialiser selection;
**no emitter change** — `EmitterBase::line()` already records origins for every backend. Python:
`{style: "sourceMapV3", sidecarExtension: ".map", footer: "# sourceMappingURL=$f"}` (the `#` comment
form; the footer is already optional in the CLI, guarded by `if (!om.footer.empty())`). PHP: the
TypeScript footer is literally reusable (`//` is valid PHP) **but only inside the `<?php` region** — a
generated file ending in `?>` needs the footer before it, which is the one case to test. Gate: the
existing `run-cli-smoke.ps1` P38 check, extended to all four targets.

**PG-C2 — the package skeleton.** `@mintplayer/polyglot-coverage` with a `remap` command: read
`originMapping` from an installed plugin package (or `--plugin-dir`), dispatch on `style`, and for
`directive` **exit 0 with a clear "this target needs no remap — its compiler attributes natively"**
rather than pretending to work. That message is the whole C# story and should be impossible to miss.

**PG-C3 — the projector, and two format readers.** The projector is language- and format-neutral: it
consumes `(file, line, hits, branches)` and the plugin's `originMapping`, and emits the same shape
keyed on `.pg`. Around it, **cobertura and lcov** readers/writers — those two cover every target
ecosystem in play (§3). `--out-format` defaults to the input format.

Must fix the three known defects of the interim implementation: credit **all** map segments rather than
first-wins (§4.3), prefer `sourcesContent` over on-disk path resolution (§4.5), and identify generated
files by *footer + sidecar* rather than a name glob (§4.6).

**PG-C4 — branch projection.** Per-`.pg`-line `(coveredArms, totalArms)`, merged per-component with
`max`; never arm-level while `column: 0` holds (§4.4). Both readers supply it — cobertura as
`condition-coverage="k% (c/n)"`, lcov as `BRDA:` records — so this is a property of the neutral
intermediate, not of either format.

**PG-C5 — prove it on Python and PHP end to end.** *Not* new adapters: a manifest entry from PG-C1, then
a real `coverage.py` / php-code-coverage run — **in whichever format that consumer happens to emit** —
projected onto a `.pg`. **The gate is that neither needs a line of engine code.** If a new *format*
turns up that is a reader; if a new *language* needs engine work, §3's claim is wrong and the design
needs revisiting before more targets are added.

**PG-C6 — automation.** *(scope 3)* Two seams, in order of value:
1. **A `polyglot-coverage` npm bin** a consumer calls in one CI line — already achieved by PG-C2.
2. **An MSBuild hook**, *only if SP3 says it is worth it.* The `.targets` is transpile-only today:
   every target hangs off `CoreCompile` and nothing touches `VSTest`/`Test`. Note the hook would be
   least useful exactly where MSBuild reaches — C#, which needs no remap at all. **Do not build this
   for its own sake.**

Any new option that changes emitted bytes must join the incrementality stamp the way
`PolyglotOriginInfo` does (`_PolyglotOriginInfoTag`) — omitting it once already caused a silent
"the feature doesn't work" failure.

## 8. Spikes — run before committing to the milestones

**SP1 — does a v3 sidecar actually serve Python?** *(gates PG-C5, and the whole "two mechanisms" claim)*
Emit a `.py` + `.py.map` for one `.pg`, run `coverage.py` over a script that exercises it, and project.
The claim under test is that nothing about Python (indentation, comprehensions, decorators) defeats a
line-granular map. **The specific risk: a `.pg` block collapsing into a comprehension makes one
generated line carry several `.pg` origins** — the one-to-many case §4.3 says to handle but which does
not arise in C#/TS today. If it appears here, §4.3's rule stops being theoretical.

**SP2 — PHP footer placement.** Emit a `.php` + sidecar; confirm the `//#` footer is valid where it
lands, including a file ending in `?>`. Cheap, and the only PHP-specific unknown.

**SP3 — is the MSBuild hook worth it?** Prototype an `AfterTargets="VSTest"` hook and answer: can it see
the coverage report's path, does it survive `dotnet test --collect`, and does it fire for the *non*-C#
targets where a remap is actually needed? **Expected answer: no on the last point**, which would kill
PG-C6.2 — worth ten minutes to find out rather than building it.

**SP4 — adopt or build?** `istanbul-lib-source-maps` (maintained, used by nyc) projects an istanbul
coverage map through source maps — ~80% of the projector, including the many-to-one merge. **But it is
istanbul-shaped**, and the design centres on a format-neutral intermediate fed by cobertura and lcov
readers; adopting it would mean converting into istanbul and back out on every path except the JS one.
Weigh that against ~150 lines of VLQ and merge code that already exist and are proven in the interim
tool. **Lean to hand-rolling the projector** and keeping the library in mind only if the JS route ever
needs bundler-map composition. (`remap-istanbul` is the obvious name-match and is **dead** — ~7 years
unpublished, istanbul 0.x. Do not.)

**SP5 — the `sourcesContent` shortcut.** Polyglot embeds `sourcesContent` deliberately, so *"the
consumer then needs no path resolution at all"*. Confirm a report can be produced using only the
sidecar's own contents, with on-disk resolution as fallback. If it holds, it removes the entire class of
path-rooting bug — which has already bitten once (the cobertura reporter writes project-relative
backslash paths that a `git ls-files` suffix match silently drops).

**SP6 — reuse plugin resolution, do not reimplement it.** The CLI already resolves plugins through
`pgconfig.json` (`file:` refs, in-box set, verified cache, npm downloads pinned by `pgconfig.lock.json`).
Decide between a `--plugin-dir` flag over the already-resolved directory and a `polyglot plugins --json`
introspection surface. The latter keeps one resolver and one closed `style` vocabulary in C++ where they
live, and is the only part of this work with a good case for touching the CLI.

## 9. Out of scope / genuinely not being done

- **Merging.** Already solved by coverage.mintplayer.com's max-merge (§5). A standalone N-report merger
  is a generic cobertura problem with nothing Polyglot-specific in it; revisit only if a consumer
  without such a service appears.
- **Column fidelity.** Fixed at line granularity by design; it would mean threading positions through
  the expression-rule interpreter, which returns bare strings.
- **A `coverage.py` file-tracer plugin.** It *is* the architectural prior art (`django_coverage_plugin`
  maps executed Python back to template lines, the Python analogue of the `#line` trick) and would make
  reports name `.pg` natively. Rejected: it needs the same sidecar as its data source, adds a Python
  package to maintain and a runtime dependency in the consumer's test run, and has no PHP counterpart.
  A third `style` is not worth one ecosystem.
- **Consumer path conventions.** Re-rooting for a particular coverage service stays consumer-side.
- **Deleting the interim tool.** `MintPlayer.AI/tools/pg_coverage_remap.mjs` keeps working until
  PG-C3 ships; it is explicitly interim, not the destination.

## 10. Already settled — do not re-litigate

- **The overlay is sound.** Extracting the `.pg` line set each target emits origin info for, across nine
  solvers: **symmetric difference zero** (3,949 lines; C#-only 0, TS-only 0). One emitter, one origin
  table, two renderers.
- **The service ingests `.pg`-keyed cobertura from a TypeScript run.** Not an inference — M63.7 ran
  exactly that in CI and it registered. The earlier note that this was "an inference, not a verified
  fact" is superseded.
- **The union is worth having.** Measured in MintPlayer.AI: `.pg` coverage **93.95% → 98.12%**, with
  **155 lines covered by the browser path and by nothing else** — two entire shipped features (a
  receding-horizon beam planner and a client-side policy net) that had no test on either side.

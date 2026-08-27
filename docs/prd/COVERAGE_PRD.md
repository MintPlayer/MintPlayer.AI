# PRD — Code coverage collection + upload to coverage.mintplayer.com (M56)

*2026-08-27 · branch `m56-coverage` · see `PLAN.md` M56 for milestone status*

## 1. Goal

Every CI test run (PRs into `master` and pushes to `master`) collects line/branch coverage and
uploads it to the MintPlayer coverage service at https://coverage.mintplayer.com, the same way
`MintPlayer.Dotnet.Tools` does (commit `f69b852` + its follow-up `827a945`, which refined the
original into the shape mirrored here). The service then owns history, PR baselines, patch
coverage, and the `coverage/project` / `coverage/patch` check runs — no gating logic lives in
the workflows.

Non-goals: coverage in the Docker/deploy workflow (`playground-docker.yml` runs no tests),
Angular/TypeScript coverage (the SPA has no test step in CI today), and any coverage *targets*
(that is server-side policy, configurable later via a repo-root `coverage.yml` read from the
base ref).

## 2. What the reference repos established (investigation summary)

Three sources were investigated: the `MintPlayer.Dotnet.Tools` workflows (local clone, commits
`f69b852` and `827a945`), the coverage server itself (`C:\Repos\Coverage`, repo
`MintPlayer/CodeCoverage` — `docs/upload-api.md` is the authoritative contract), and this
repo's CI.

Findings that shape the design:

- **Collection** is coverlet's VSTest data collector: `dotnet test --collect:"XPlat Code Coverage"`.
  This repo's single test project (`tests/MintPlayer.AI.ReinforcementLearning.Tests`) already
  references `coverlet.collector` 10.0.1, so no package changes are needed. Each run writes
  `coverage.cobertura.xml` into a GUID-named subdirectory of `--results-directory`.
- **Upload** is the `MintPlayer/CodeCoverage/action@master` composite action. It gzips each
  report, POSTs `multipart/form-data` to `/api/uploads` (repository, commitSha [PR *head* sha,
  not the merge commit], branch, runId/runAttempt, `git ls-files` output for path matching, one
  `files` part per report), and optionally POSTs `/api/uploads/finish`. Multiple reports/jobs
  under one `(repo, sha, runId, runAttempt)` merge server-side with max semantics — no
  reportgenerator step is needed or wanted.
- **Auth**: the server accepts a static `covt_…` token or **GitHub Actions OIDC** (id-token with
  audience = server base URL). `MintPlayer/MintPlayer.AI` is **public**, and public repos are
  auto-provisioned on their first OIDC upload — so this repo uses **OIDC** (`use-oidc: true` +
  `permissions: id-token: write`) and needs **no secret at all**. (Dotnet.Tools predates this
  choice and uses `secrets.COVERAGE_TOKEN`; token auth remains the documented fallback if the
  repo ever goes private.)
- **`--no-build` on the Test step**: in Dotnet.Tools this was the 90-second win (`--no-restore`
  alone does NOT imply `--no-build`; the suite was silently rebuilt in Debug and coverage
  measured on that build — measured 88s → 32s). **This repo already has `--no-build` on both
  Test steps and on Pack**, so that lever is already pulled; the workflows here only gain the
  collection/upload steps.
- Refinements Dotnet.Tools added after `f69b852`, all adopted here from the start:
  `--settings coverlet.runsettings`, the `hashFiles(...) != ''` no-report guard,
  `disable-search: true` (a non-matching glob must be a no-op, not an upload of stray
  unparsable files), `finish: true` (close the build immediately instead of waiting out the
  ~2-minute server debounce), `fail-ci-if-error: false` (a coverage-service outage must never
  block a merge or a NuGet release), and the fork-PR skip (forks get neither secrets nor
  OIDC, and a contributor's PR must not go red over that).

## 3. Design

### 3.1 `coverlet.runsettings` (repo root)

Mirrors Dotnet.Tools' file, with one repo-specific addition. Settings:

- `Format: cobertura` — one of the three formats the server parses (lcov / Cobertura / JaCoCo).
- `ExcludeByFile: **/obj/**/*.cs,**/*.g.cs,**/*.Designer.cs` — the repo-specific part is
  `**/obj/**/*.cs`: this repo compiles **generated code out of `obj/`** — the Polyglot
  transpiler's C# (`obj/**/polyglot/*.cs` in the Environments project) and
  `MintPlayer.SourceGenerators` output. Those files are not in `git ls-files`, so the server
  could never resolve their paths anyway; excluding them keeps the total honest (source you
  can actually view) instead of diluting it with unmatchable generated lines. The `.pg`
  *sources* stay effectively measured through nothing — accepted: coverage speaks C#/TS file
  language, and the generated C# is exercised or not together with its callers.
- `Exclude: [MintPlayer.AI.ReinforcementLearning.Ilgpu]*` — **required, found the hard way** (37
  test failures on the PR's first CI run): ILGPU compiles kernel methods from their IL at
  runtime, and coverlet's injected `RecordHit` tracker calls make that compile throw
  `ILGPU.InternalCompilerException` (in `GemmTiled_Kernel` et al.). The whole assembly stays
  uninstrumented — per-method exclusion would be fragile since any helper a kernel calls breaks
  the same way. `Ilgpu.Hosting` (DI glue, no kernels) stays covered. This is the one deliberate
  departure from Dotnet.Tools' "no assembly Exclude list" stance; that repo has no runtime IL
  compiler.
- `UseSourceLink: false` — SourceLink would rewrite report paths to raw.githubusercontent URLs,
  which breaks the server's suffix-matching against `git ls-files`.
- `DeterministicReport: false`, `SkipAutoProps: false` — same choices as Dotnet.Tools; no
  `ExcludeByAttribute`, no assembly `Exclude` list (measure what runs; don't editorialize).

### 3.2 Workflow changes

`pull-request.yml` and `build-master.yml`, Test step (the `--filter "Category!=Slow"` stays —
the number is "fast-bucket coverage", stated in the README next to the badge):

```
dotnet test --configuration Release --no-build --filter "Category!=Slow" --verbosity normal
  --settings coverlet.runsettings --collect:"XPlat Code Coverage" --results-directory coverage
```

Upload step, after Test:

- **PR workflow**: guarded by same-repo (`github.event.pull_request.head.repo.full_name ==
  github.repository`) + `hashFiles('coverage/**/coverage.cobertura.xml') != ''`,
  `continue-on-error: true`, with `base-sha: ${{ github.event.pull_request.base.sha }}`.
  Placed after Test and **before** the Angular build (upload as soon as the report exists;
  an Angular failure shouldn't cost the coverage datapoint). Job gains
  `permissions: contents: read, id-token: write`.
- **master workflow**: guarded by `always() && hashFiles(...) != ''` (upload even if a later
  step would fail; `always()` because Test failing must still not lose the datapoint),
  before Pack. Job gains the same permissions block (plus `packages: write`, which the job
  already used implicitly via `github.token` — making permissions explicit narrows nothing).

Both use `url: https://coverage.mintplayer.com`, `use-oidc: true`, `files:
coverage/**/coverage.cobertura.xml`, `disable-search: true`, `finish: true`,
`fail-ci-if-error: false`. No `flags:` (Dotnet.Tools removed theirs).

### 3.3 Housekeeping

- `.gitignore`: add `coverage/` (the `--results-directory`; `coverage*.xml` alone doesn't cover
  the directory).
- `README.md`: badge at the top —
  `[![Coverage](https://coverage.mintplayer.com/badge/MintPlayer/MintPlayer.AI.svg)](https://coverage.mintplayer.com/r/MintPlayer/MintPlayer.AI)`
  with a one-line note that the figure is the fast test bucket (`Category!=Slow`).

## 4. Spike (S1) — validate collection locally before touching CI

Run a *targeted* test slice (never the full suite locally) with the new runsettings:

```
dotnet build --configuration Release -p:EnableSpaBuilder=false
dotnet test --configuration Release --no-build --settings coverlet.runsettings \
  --collect:"XPlat Code Coverage" --results-directory coverage \
  --filter "FullyQualifiedName~Tensor&Category!=Slow"
```

Pass criteria: a `coverage/<guid>/coverage.cobertura.xml` appears, root element is `coverage`
(Cobertura), and it contains **no** `obj/`-generated file paths (grep for `polyglot` and
`.g.cs`). This validates the collector + runsettings interplay (including that the collector
works under `--no-build`) without a CI round-trip.

**Spike results (2026-08-27, clean build):** report produced and parsed as Cobertura; the
exclusion works — without the runsettings the report holds 365 files (57 under `obj/`, 41
Polyglot, 16 `.g.cs`); with it, 309 files and all Polyglot/generator files gone **except one**:
the `[Inject]`-generated constructors document
(`…Campaigns/obj/…/MintPlayer.SourceGenerators…/Inject.g.cs`) escapes both `**/obj/**/*.cs`
and `**/*.g.cs` while 15 identically-shaped sibling documents are excluded — cause not
identified (not staleness; reproduced after `dev clean` + full rebuild). Accepted as a known
one-file leak: it is uncovered DI ctor glue, 1 of 309 files, immaterial to the total.
`ExcludeByAttribute=GeneratedCodeAttribute,CompilerGeneratedAttribute` was tried and
**rejected**: it still doesn't catch that document and drops 77 further *real* source files
(309 → 232) — the cure deletes more truth than the disease.

## 5. Milestones

- **M56.1 — Spike S1**: runsettings + local targeted collection proof.
- **M56.2 — Workflows**: collection + upload in `pull-request.yml` and `build-master.yml`,
  `.gitignore` entry.
- **M56.3 — README badge** + PLAN.md entry.

Acceptance (post-merge, observable on the service): the PR run uploads and the build reaches
`Complete` on coverage.mintplayer.com; after merge, `master` shows a baseline and the badge
renders a percentage.

## 6. Out of scope / genuinely not being done

- Coverage gating / branch protection on `coverage/project`–`coverage/patch` — server-side
  setup the owner does in the service UI (and optionally `coverage.yml`), not workflow YAML.
- Slow-bucket (`Category=Slow`) coverage — those are multi-minute training/perft gates; running
  them per-PR for coverage would swamp CI for a number nobody steers by.
- Angular test coverage — there is no `ng test` in CI to instrument.

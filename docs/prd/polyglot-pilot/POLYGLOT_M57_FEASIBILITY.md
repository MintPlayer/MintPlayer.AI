# Polyglot × M57 (Tetris techniques) — feasibility handoff

**Status:** ✅ INVESTIGATED **and MIGRATED** 2026-08-30. Two separate outcomes, don't conflate them:
1. **M57 needs NO Polyglot compiler change.** Every construct the plan requires already transpiled and
   type-checked on the then-pinned **0.8.1** — the "can't express a BFS queue" constraint is measurably false.
2. **The repo nonetheless moved to 0.9.9** (§8), on the owner's decision to standardise on `constructor`.
   That was a convention call, not a technical dependency of M57.
**Owner:** Pieterjan
**Relates to:** [TETRIS_TECHNIQUES_PRD.md](../TETRIS_TECHNIQUES_PRD.md) (M57) · [PLAN.md](../PLAN.md) M57 ·
consumer pin `MintPlayer.Polyglot.MSBuild` **0.8.1 → 0.9.9** (§8) in
`src/MintPlayer.AI.ReinforcementLearning.Environments/MintPlayer.AI.ReinforcementLearning.Environments.csproj:15`
**Upstream:** `C:\Repos\MintPlayer.Polyglot` (HEAD `027a878`, P37; released as **0.9.9**, on the
`github-mintplayer` feed and nuget.org) — written here per the cross-repo handoff convention; the Polyglot repo
is not written to from this session, and **no upstream issue was filed** (there was no feature to request).

> **Sections 1–7 are the investigation as it stood against the then-current 0.8.1 pin, and are left as written
> — their conclusions are what justified the migration. Section 8 records what was actually done.**

---

## 1. Why this document exists, and how the premise changed

`TETRIS_TECHNIQUES_PRD.md` §2.1/§3.2 records a constraint inherited from `snake_solver.pg:266–269`:

> *"a queue whose pushed values derive from reads of the same list makes the transpiled TS an evolving-any
> array with circular element inference (TS7022)"*

...and concludes that a real BFS queue is not expressible, that the sanctioned idiom is bounded relaxation
over a fixed `List<bool>` (O(cells²)), and that M57 should prefer a frame-sweep enumerator *specifically to
avoid the queue*. The M57 PRD further lists `while`, `Dictionary`/`Set`, method overloading, interfaces,
nested generics and records as unavailable.

**Measured, not assumed: most of that is stale.** The constraint list describes a much older compiler. It has
been shaping `.pg` designs in this repo — the Snake reachability workaround is the visible cost — and it
should stop.

## 2. What was actually tested

Both CLIs were already on this machine, so no build was needed:

| Version | Path |
|---|---|
| **0.8.1** (the consumer pin) | `%USERPROFILE%\.nuget\packages\mintplayer.polyglot.msbuild\0.8.1\tools\win-x64\polyglot.exe` |
| **0.0.0-dev** (a local build of HEAD) | `…\mintplayer.polyglot.msbuild\0.0.0-dev\tools\win-x64\polyglot.exe` |

Scratch sources and all emitted output:
`…\scratchpad\pgtest\` (`queue_test.pg`, `m57.pg`, `m57_081.pg`, `migr/*.pg`).

### 2.1 The TS7022 queue claim does **not** reproduce — on either version

A textbook worklist BFS was written in `.pg`: a `List<i32>` frontier, a `List<bool>` seen-set with
index assignment `seen[start] = true`, a `while head < frontier.count` loop, and — the exact shape the
comment describes — **`frontier.add(nxt)` where `nxt` derives from a read of `frontier` itself**.

`polyglot check` → **no problems on 0.8.1 and on HEAD.** The emitted TypeScript:

```ts
bfs(start: number): number {
    let frontier: number[] = [];        // <-- explicitly annotated. No evolving-any.
    let seen: boolean[] = [];
    ...
    while (head < frontier.length) {
        const cur = frontier[head];
        ...
        frontier.push(nxt);
    }
    return visited;
}
```

The annotation is emitted in **both** declaration forms — the explicit `var frontier: List<i32> = List<i32>()`
*and* the inferred `var frontier = List<i32>()`. `tsc --noEmit --strict --target es2020` (the repo's own
TypeScript, `ClientApp/node_modules/.bin/tsc`) **exits 0**.

> **Consequence beyond M57:** `snake_solver.pg`'s O(cells²) bounded-relaxation workaround for
> `reachableFreeSpace` (`:266–299`) is **unnecessary**. A real BFS is expressible and would be
> asymptotically faster. That is a Snake cleanup, tracked as a follow-up below — not a Polyglot bug.

### 2.2 `while` exists, and so does most of the "unavailable" list

`docs/lang/SPEC.md` at HEAD documents `while` and `do…while` (§5, line 333), interfaces and generics (§4.4),
`record` with structural equality (§4.2), `is`/`as` (§4.6), attributes (§4.7), operator overloading (§6.1),
properties and indexers (§6.2), extension functions (§6.3), exceptions (§5.1) and `use` disposal (§5.2).
`while` was verified working on **0.8.1** as well, by the §2.1 test.

### 2.3 Every M57 construct compiles on the pinned 0.8.1

One file (`m57_081.pg`) exercising all five things the PRD needs:

| | Construct | 0.8.1 | HEAD |
|---|---|---|---|
| **A** | Flat SRS kick tables built by `add`, **including negative literals** (`-1`, `-2`) | ✅ | ✅ |
| **B** | `NesInput` port: a class with mutable `var` scalar fields, a `setTapRate` dial that **mutates** the DAS constants at runtime, a 6-`bool`-parameter `tick(...): bool`, and **another class instance held as a field** and called into (`this.input.tick(...)`) | ✅ | ✅ |
| **C** | Frame-simulation enumerator as a **bounded `for` sweep with `break`**, nested 3 deep (direction × goal-rotation × frame), accumulating into a returned `List<i32>` | ✅ | ✅ |
| **D** | Tuck spots as a **`record PgTuckSpot(rot, x, y)`** in a `List<PgTuckSpot>`, plus **nested `List<List<i32>>`** | ✅ | ✅ |
| **E** | A genuine worklist/BFS queue (§2.1) | ✅ | ✅ |

`polyglot check` clean; TS and C# both emitted; the TypeScript type-checks under `--strict`.

**The PRD's claim that `.pg` has "no interfaces, closures or function values" so the `DasHost` callback must be
inlined:** inlining works and is what was tested (B), and it is the right call for a hot path regardless — but
interfaces *are* available at HEAD (SPEC §4.4) should a future design want them.

### 2.4 The only thing separating 0.8.1 from HEAD, for this repo, is `init(` → `constructor(`

HEAD rejects the current `tetris_solver.pg` outright:

```
tetris_solver.pg:33:3: error: expected a member
tetris_solver.pg:36:5: error: expected a declaration
…
```

Line 33 is `init(seed: i32) {`. P37 (`027a878`) renamed the constructor keyword. There are **22 `init(` sites
across the 7 solvers** (chess 4, crazyfruits 3, draughts 4, fruitcake 3, mountaincar 2, snake 2, tetris 4).

A one-line `sed -E 's/^([[:space:]]*)init\(/\1constructor(/'` migrates all seven — **and all seven then
`check` clean at HEAD**, with no other change:

```
chess_solver.pg          no problems      fruitcake_solver.pg    no problems
crazyfruits_solver.pg    no problems      mountaincar_solver.pg  no problems
draughts_solver.pg       no problems      snake_solver.pg        no problems
tetris_solver.pg         no problems
```

### 2.5 A bump would move the parity checksums — one change is semantic

Emitted TypeScript for `tetris_solver.pg`, 0.8.1 vs HEAD: **20 differing lines.** Most are extra `| 0`
normalization around shifts (semantically identical for in-range values). **One is a real behaviour change:**

```ts
// 0.8.1                    // HEAD
this.bag = [];              this.bag.length = 0;
```

`List.clear()` now **mutates in place** instead of rebinding to a fresh array. HEAD's lowering is the more
faithful one (it matches C# `List<T>.Clear()`, so the two targets agree where they previously could diverge
under aliasing). It is a **fix**, but it is observable, so a bump requires re-running every game's parity
harness and re-pinning where the checksum moves.

---

## 3. Conclusions

1. **M57 needs no compiler work and was never blocked.** `TETRIS_TECHNIQUES_PRD.md` §2.1/§3.2's Polyglot risk
   (and risk 5 in §10) is **retired** — see §5. *(The repo did subsequently move to 0.9.9 by owner decision,
   §8 — a convention change, not a fix for anything M57 needed.)*
2. **The frame-sweep-over-BFS argument in the PRD stands, but for the right reason.** It is preferable because
   a bounded forward sweep is simpler and cheaper per call inside `buildObservation` (7× per beam node), **not**
   because a queue is inexpressible. The queue is available as the fallback if the sweep proves insufficient.
3. **There is no Polyglot feature to request for M57.** Filing a feature issue would be inventing a need.
4. **There is one genuine upstream ergonomics issue** — §4.
5. **A version bump is cheap and worth doing on its own schedule**, but it is *not* M57's dependency and should
   not be bundled into an already-large milestone. Cost: one `sed`, plus re-validating 7 parity pins.

---

## 4. The one real upstream issue

Not a missing feature — a **migration diagnostic**. The `init(` → `constructor(` rename is a silent breaking
change whose failure mode is a cascade of misleading parse errors (`expected a member`, `expected a
declaration`) pointing at the line *after* the real problem, with no mention of the renamed keyword. Every
downstream `.pg` consumer hits this exactly once, and the error text gives no path to the fix.

**Requested:** recognise `init(` in constructor position and emit a targeted diagnostic —
`error: 'init(' was renamed to 'constructor(' in <version>` — or accept it as a deprecated alias for one
minor version. Either turns a 20-minute archaeology session into a one-line fix.

Draft issue text: §6.

---

## 5. Edits this lands in the M57 PRD

- §2.1: strike the TS7022 / "no real BFS queue" constraint; replace with the measured result and a pointer here.
- §2.1: strike `while`, records, nested generics, interfaces from the "absent" list (verified present).
- §3.2: keep the frame-sweep recommendation; restate the rationale as cost, not expressibility. Remove the
  "owner has pre-approved compiler changes / decide at M57.3" fork — there is nothing to decide.
- §10 risk 5 ("Polyglot 0.8.1 cannot express the enumerator"): **delete**; it is measurably false.
- §11: move the version bump from an implied dependency to explicit out-of-scope, with §2.4/§2.5 as its cost.

## 6. Follow-ups for this repo (not M57 blockers)

- **Snake:** `snake_solver.pg:266–299` — replace the O(cells²) bounded relaxation with a real BFS, and delete
  the stale comment that has been propagating this constraint. Parity pin must be re-validated.
- **Version bump, as its own small arc:** `sed` the 22 `init(` sites, bump the `PackageReference`, re-run all 7
  parity harnesses, re-pin the checksums that move (expect at least the `clear()` aliasing change, §2.5).
- **Repo hygiene:** the constraint list in `TETRIS_PRD.md` §2.4 is dated to the compiler of its day. Anything
  quoting compiler limits should carry the version it was measured against.

## 7. Reproduce

```bash
PG081=~/.nuget/packages/mintplayer.polyglot.msbuild/0.8.1/tools/win-x64/polyglot.exe
PGDEV=~/.nuget/packages/mintplayer.polyglot.msbuild/0.0.0-dev/tools/win-x64/polyglot.exe

# every .pg needs the collections import — this is what a bare scratch file is missing
#   import { List } from "std.collections"

"$PG081" check m57_081.pg          # M57 constructs on the PINNED version -> no problems
"$PGDEV" check m57.pg              # same file, constructor() spelling    -> no problems
"$PG081" build queue_test.pg --target typescript --out ./out
src/RLDemo.Web/ClientApp/node_modules/.bin/tsc --noEmit --strict --target es2020 ./out/queue_test.ts
```

---

## 8. The 0.9.9 migration (executed 2026-08-30)

**Owner decision:** *"we just need to use `constructor` from now on."* Since `constructor(` is a hard break —
0.8.1 **and** 0.9.2 both reject it, 0.9.9 rejects `init(` — standardising on it required the bump.

**Publishing blocker, resolved by the owner mid-task.** v0.9.8 was the only tag containing P37's rename, and it
was unpublished: the feed topped out at **0.9.2**, which was verified to still require `init(`. The owner then
published **0.9.9** (also on nuget.org).

**Equivalence check before trusting it:** 0.9.9's emitted TypeScript for `tetris_solver.pg` is **byte-identical**
(0 diff lines) to the local `0.0.0-dev` build of HEAD that §2 was measured against — so every finding above
transfers to 0.9.9 unchanged.

### What was changed

| | |
|---|---|
| `…Environments.csproj:15` | `MintPlayer.Polyglot.MSBuild` **0.8.1 → 0.9.9** |
| 7 `.pg` solvers | **21 `init(` → `constructor(`** declaration sites, plus one stale comment in `tetris_solver.pg:65`. Anchored `sed -E 's/^([[:space:]]*)init\(/\1constructor(/'`; audited first — there are **no `init(` call sites**, only declarations |

### Verification

| Gate | Result |
|---|---|
| `polyglot check`, all 7 solvers under 0.9.9 | **no problems** ×7 |
| `dotnet build` Environments (Release) | **0 errors** |
| C# parity tests (`~Parity`) | **19/19 pass** — no checksum moved |
| `node tools/tetris_parity.mjs` | `checksum=472451993` — **matches the M54.1 pin** |
| `node tools/cf_parity.mjs` | `checksum=481681208 score=95950` — **matches the M50.6 pin** |
| `node tools/tetris_das_check.mjs` | **ALL PASS** (11 frame-exact checks) |
| Full CI bucket `Category!=Slow` | **529/529 pass** |

**No parity pin moved.** §2.5 flagged HEAD's `List.clear()` lowering change (TS `this.bag = []` →
`this.bag.length = 0`) as observable; it turns out not to be *behaviourally* observable in any pinned protocol —
the pins hash game state, and in-place clear vs rebind is indistinguishable there because nothing aliases those
lists. The change is still real and is now live in the twins.

### Two things worth knowing for next time

- **The CLI writes only when content changes.** After the migration, `chess_solver.ts` and `draughts_solver.ts`
  kept July mtimes while the other five updated — which looks exactly like the stale-codegen bug this repo has hit
  before. It wasn't: a fresh transpile to a scratch directory is **byte-identical** to both on-disk twins, because
  those two solvers emit the same TS under 0.9.9 as under 0.8.1. **Verify twin freshness by content, never by
  mtime.**
- **Confirm which CLI actually ran.** There is no `polyglot` on PATH and no `PolyglotTool`/env override in this
  repo (both checked), so MSBuild used the package's bundled `tools/win-x64/polyglot.exe`. The cheap proof is the
  codegen signature: **0.9.9 emits `this.bag.length = 0`, 0.8.1 emitted `this.bag = []`** — grep the twin.

### Convention going forward

**`constructor(...)`, never `init(...)`.** Any `.pg` written from now on — including all of M57 — uses
`constructor`. 0.9.9 rejects `init(` outright with a misleading `expected a member` at the following line, so a
stray `init(` fails loudly rather than silently.

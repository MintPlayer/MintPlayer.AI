# PRD — Tamper-resistant leaderboards for the playground games

*Status: investigation complete (2026-08-27), implementation not started.*
*Produced from an 18-agent investigation: repo-wide game/engine audit, per-game determinism
spikes with file:line evidence, MintPlayer.Spark backend recon, web prior-art survey, and a
three-way design panel with an adversarial judge. Companion milestone plan in §12.*

---

## 1. Problem statement

Add a per-game leaderboard to the playground (ai.mintplayer.com) that players can trust. The
obstacle is that every game is pure HTML/TypeScript: the client is fully inspectable and
modifiable. Storage/backend is decided: **MintPlayer.Spark** (`C:\Repos\MintPlayer.Spark`).

### Threat model

Assume the attacker can:

- read all client code and watch the network panel, then **forge the submit request** with any score;
- **tamper with any client-held value** (score variable, memory, localStorage) before submission;
- **replay someone else's submission** or their own;
- **script inputs** (a bot playing the game, including this repo's own shipped AIs);
- run the game under devtools with breakpoints, slow-motion, savestate-style snapshots.

No client-held secret survives this model. Anything the client computes (HMAC signatures,
obfuscated checksums, "encrypted" payloads) can be recomputed by the attacker from the same
code — the Unipop postmortem is the canonical demonstration: the attacker found the client's
sanity-check hash in the debugger, reimplemented it in Python, and submitted arbitrary scores.

## 2. The core insight

**The score must never be an input to the server. It must always be an output of the server's
own re-simulation.**

This is the TrackMania model (option 3 in the original framing), but fully automated: the
client submits `(challengeId, inputTrace)` — never a score — and the server **re-runs the game**
from the server-issued seed and the trace, deriving the score itself. TrackMania needs this
because its physics are deterministic at a fixed 10 ms tick; AntGame.io does the same for a
browser game (server-issued seeds bound to account + rate limits + server re-simulation).

This repo is unusually well positioned for it: **several engines are already single-source
Polyglot `.pg` files that transpile to both C# and TypeScript with byte-parity gates**
(crazy-fruits checksum 481681208 over 1000 moves; tetris `TetrisParityTests`; fruit-cake f64
state checksum over a 28-drop cascade). The C# twin the server needs already exists and is
already proven equal to the exact code the browser runs. The per-game spikes (§7) confirm
every launch game is either deterministic today or deterministic after small, enumerated
changes (mostly: seed the one `Math.random` call site from a server nonce, and record inputs).

What replay verification proves — and what it doesn't:

- ✅ Proves the score is **achievable under the rules** from a server-issued seed, submitted
  within a wall-clock window consistent with real-time play.
- ✅ Defeats: forged submits, score tampering, memory editing, replaying another player's run,
  seed-shopping, parameter-shopping (easy mode claimed as hard mode), trace editing.
- ❌ Does **not** prove a *human* produced the inputs. A bot playing at 1× with human-like
  timing jitter passes every mechanical check — this repo ships superhuman bots for these
  exact engines. That residue is handled by cost-raising, detection, and moderation (§9), and
  the board's copy must say "verified runs", never "verified humans". No system closes this
  gap cryptographically (TrackMania's answer is a native input-attestation patch plus human
  forensic review; speedrun.com's answer is pure moderation).

## 3. Rejected alternatives (and why)

| Approach | Verdict |
|---|---|
| **Client-side signing** (option 2 in the original framing — the "applet private key") | Rejected as an integrity mechanism. The key or the signing code ships in the JS; the attacker calls our own signing function from the console with a doctored score. Java applets only ever raised extraction cost. Tokens still have a role — session *binding*, not integrity (§8). |
| **Obfuscation / WASM / anti-debugging** | Never load-bearing. Unanimous in the literature: falls to AST rewriting and patience; WASM adds one disassembly step. Optional cheap top layer at most; not planned. |
| **Published "plausibility-validated" tier** (show unverified scores with a badge while verifiers are built) | Rejected by the design panel. Client-computed heartbeat hashes are forgeable, so the tier publishes forgeable entries on a public board — a liability the badge only partially covers — and the verifier rollout obsoletes it within weeks. Each game's board launches **only with its verifier**. |
| **Server-authoritative live play** (inputs streamed over WebSocket, server simulates in real time) | Rejected as the backbone: a live simulation per concurrent player vs. one batch re-sim per submission; in-memory sessions die on host restarts (and the dev workflow restarts hosts routinely); latency risk for 60 fps games; the most novel/least reusable code of the three panel designs. Two of its ideas are adopted as targeted future options (§11): server-dealt randomness against seed-foreknowledge, and server-side AI opponents for chess/draughts. |

## 4. Decided architecture (one paragraph)

A separate **Arcade** Spark host (Fleet-demo pattern) owns challenges, submissions, verification
and the leaderboard. To enter a leaderboard the client first requests a **challenge** —
a single-use, user-bound, TTL'd nonce carrying the server-chosen **seed** and pinned game
parameters/engine version. It plays the game seeded from it, recording a compact **input trace**.
It submits `(challengeId, trace)` — no score field exists. Intake does cheap structural
pre-checks (owner, single-use, TTL, size caps, wall-clock lower bound, per-game plausibility
table), then queues the submission on Spark's durable message bus. A verification recipient
replays the trace through the **C# engine twin**, derives the score, and writes the
`LeaderboardEntry` (server-side only — `security.json` grants no client group any write right
on it). Traces are retained permanently so future verifier/rule changes can re-sweep
retroactively (the Metanet lesson). Boards are served as Spark **streaming queries** (native
WebSocket diff push — live leaderboard for free). RLDemo.Web stays completely trust-free.

## 5. Backend design (MintPlayer.Spark)

Spark recon confirmed every needed capability exists; the idiomatic shape mirrors
`Demo\Fleet`:

**New host** `src/Arcade` (Spark app) + `src/Arcade.Library` (entities/messages), with a plain
`<ProjectReference>` to `MintPlayer.AI.ReinforcementLearning.Environments` for the C# engine
twins. Separate host, not inside RLDemo.Web: the demo host is restarted routinely during
development and must hold no trust or state; Spark needs RavenDB, which RLDemo.Web shouldn't
grow a dependency on. In dev, RLDemo.Web proxies `/arcade/*` to it.

**Entities** (`[GenerateIndex]` POCOs in Arcade.Library):

- `RunChallenge` — `{ Id (nonce), UserId, GameId, Seed, GameParams, EngineVersion, IssuedAtUtc, ExpiresAtUtc, Consumed }`. Consumed atomically via RavenDB optimistic concurrency (double-submit of one challenge is a conflict, not a race).
- `ReplaySubmission` — `{ ChallengeId, UserId, GameId, Trace (retained permanently), SubmittedAtUtc, Status: Pending|Verified|Rejected|Flagged, DerivedScore, RejectReason, Forensics }`.
- `LeaderboardEntry` — natural id `LeaderboardEntries/{gameId}/{userId}` (point-write per-user best; a better verified score overwrites). Fields: score, submissionId, verifiedAtUtc, quarantine state.
- `GamePolicy` — per-game plausibility table (§9): score ceiling, max trace length, min wall-clock per trace unit, params whitelist.
- `FlagQueueItem` — moderation queue item (PersistentObject ⇒ admin UI for free).

**Auth**: `MintPlayer.Spark.Authorization` (Identity on RavenDB, `/spark/auth/*`). Playing is
anonymous; **submitting requires an account** (rate-limit handle + accountability).
`security.json`: `anonymous` gets `Query`/`Read` on `LeaderboardEntry` only; **no client group
gets `New`/`Edit` on `LeaderboardEntry`** — it is written exclusively by the verifier. Challenge
issue + submission go through an `ArcadeController` (`spark.AddControllers()`,
`[SparkAuthorize]`), metered by Spark's built-in rate limiter
(`spark.AddRateLimiter(o => o.PathPrefixes = ["/spark", "/api/arcade"])`; ~20 challenge
issues/min, ~2 submissions/min per user — the AntGame numbers).

**Verification pipeline**: `[MessageQueue("ReplayVerification")]` record + `VerifyReplayRecipient
: IRecipient<ReplaySubmittedMessage>` dispatching to an `IReplayVerifier` registry (one verifier
per game, each wrapping its C# engine twin). Durable, FIFO, auto-retried, idempotent (re-verify
of a Verified submission is a no-op). Fruit-cake gets its own
`[MessageQueue("PhysicsReplays")]` so second-scale physics replays never starve the
microsecond-scale board-game queue. A `RetroVerifySweepJob` (`ISparkCronJob`) re-verifies stored
traces after any verifier/engine change.

**Engine versioning**: every challenge pins `EngineVersion` — the game's parity checksum (e.g.
crazy-fruits 481681208) or a rules-version constant bumped on any `.pg` change. Verifier
rejects version mismatch; old traces are re-swept or archived, never mis-verified against new
rules.

## 6. Protocol

```
Client                                   Arcade host
  |-- POST /api/arcade/challenge ----------->|  auth'd, rate-limited
  |     { gameId, params }                   |  validate params against GamePolicy
  |<-- { challengeId, seed, engineVersion } -|  store RunChallenge (TTL)
  |                                          |
  |  play: engine.reset(seed); record trace  |
  |  (real-time games: heartbeats, §9)       |
  |                                          |
  |-- POST /api/arcade/submit -------------->|  intake: owner? unconsumed? within TTL?
  |     { challengeId, trace }               |  size cap? wall-clock >= floor(trace)?
  |                                          |  GamePolicy structural pre-checks?
  |                                          |  atomically consume challenge
  |                                          |  broadcast ReplaySubmittedMessage
  |<-- { submissionId, status: Pending } ----|
  |                                          |  [recipient] C# twin replays trace,
  |                                          |  derives score, writes LeaderboardEntry
  |-- GET /api/arcade/submission/{id} ------>|  (or the streaming query just updates)
  |<-- { status: Verified, score } ----------|
Leaderboard read: Spark streaming query /spark/queries/{leaderboard}/stream (anonymous, live diffs)
```

Intake wall-clock check both ways: a trace claiming N ticks of play submitted before
N·tickMs of wall-clock has elapsed since issue is auto-rejected (time compression); the TTL
bounds the other side (and limits offline optimization time against a known seed — see §11
for the stronger answer).

**Client shared module** `ClientApp/src/app/shared/run-session.ts`: `startRun(gameId, params)`
→ challenge fetch; a `TraceRecorder` seam each game feeds; `submit()`; verdict via poll or the
streaming query. Plus **taint tracking**: any resume-from-snapshot, edit-mode board, solver
call, or state-restore during a run voids the challenge client-side (and the server-side
checks that can, do — e.g. cube voids a challenge if `/api/cube/solve*` is called while it's
active).

## 7. Per-game findings and required changes

All spikes verified against the actual code. Verdicts: **deterministic-now** (replayable as-is),
**deterministic-with-changes** (enumerated small changes), **hard**.

| Game | Verdict | Effort | The blockers, precisely |
|---|---|---|---|
| rush-hour | deterministic-now | S | Zero RNG, zero timing, integer-only; C# `RushHourBoard` + `RushHourSolver` (par) already exist. Only plumbing: record `(vehicle,dir)` in `tryMove`, pin scores to a **board-content hash** (the dev deck editor mutates levels under stable ids, `RushHourDeckStore.cs:59-62`). Metric: moves-over-optimal across the deck (server re-solves par at verify time). |
| crazy-fruits | deterministic-with-changes | S | Engine fully seeded (i32-only, parity checksum 481681208); human loop is synchronous turns. Changes: server seed replaces `Date.now()` at `crazy-fruits-game.ts:60`; record `(action, centerIsA:1bit)` per move — **comboCenter is a load-bearing hidden input** (combo blasts centre on the last-selected cell; engine clamps it to the swap's own two cells, so 1 bit suffices); verifier drives internal `stageSwap(action, targetCell)` via InternalsVisibleTo or a new two-arg `ApplySwap` facade overload; require `movesMade==30`; reject freePlay. Trace ≈ 34 bytes. |
| game-2048 | deterministic-with-changes | S | Only nondeterminism: unseeded spawns (`game-2048-classic.ts:126-127`). Changes: TS `Xoshiro256StarStar` port (BigInt; `NextInt` path, exact 2/4 draw via power-of-two-scaled `NextDouble` vs literal 0.9 — bit-exact); normalize empty-cell order to the server's row-major (`Game2048.cs:136-141`); **fix the merge-cap divergence** (client merges uncapped at `game-2048-classic.ts:101`, server caps exponent 15 at `Game2048.cs:76`); pin the standard start board; log actions only where `moved===true`. Verifier = existing `Board2048.ApplyMove` + `Spawn`. |
| snake | deterministic-with-changes | S | Only nondeterminism: unseeded food spawn (`snake-logic.ts:82`); state is a pure function of (tick, heading) — fixed 150 ms tick, integer-only. Changes: seeded integer food picks (`NextUInt64 % freeCount`, mirroring `SnakeEnv.cs:122,139` — **never** the float `NextDouble*count` path); record `(tickIndex, dir)` for accepted presses; adopt the `.pg` twin's 288-tick starvation cap in human play (bounds trace length by score — cheap pre-check); prefer retiring hand-mirror `snake-logic.ts` for the already-shipped generated `snake_solver.ts`. |
| tetris | deterministic-with-changes | S–M | Best-positioned: fixed-timestep 60.0988 Hz NES-frame accumulator, engine-internal seeded LCG, byte-parity C# twin with the full micro API. Changes: server seed replaces `Date.now()` at `tetris-game.ts:59` (raw i32); add a frame counter + ordered event log — 11 opcodes incl. press/release edges, pointer ops, and **ClearInputs** (pause/blur/pointer-takeover mutate DAS state and must be traced); same-gap event order preserved (rotate-then-shift ≠ shift-then-rotate). One real engine task: **port `tetris-das.ts` (~100 integer lines) into `tetris_solver.pg`** so the DAS/gravity input machine is single-sourced (server must run it to enforce gravity); `tools/tetris_das_check.mjs` becomes the parity vector. Hypertap >30 Hz = hard reject (physical ceiling the DAS machine itself encodes). |
| fruit-cake | deterministic-with-changes | M | Human play **already runs the transpiled `.pg` physics twin** (`new PgFruitCakeWorld(true)`, f64 byte-parity proven) on a fixed 1/60 accumulator. Changes: integer tier-PRNG **into the `.pg`** replacing `Math.random()` at `fruit-cake-game.ts:194`; fixed-step counter + `(step, aimXPx:f64)` drop log; `Math.hypot`→`sqrt(vx²+vy²)` at `fruit-cake-physics.ts:120` (hypot is not correctly rounded — boundary divergence risk on the rest-speed game-over check); port the ~30-line TS rules layer (cooldown/grace/game-over) as whole-step integers; **taint snapshot-resumed games ineligible** (resume zeroes velocities — unreconstructible); verifier drives the **double-precision** generated core with `dt = 1.0/60.0` exactly (the float32 `FruitCakeWorld` facade breaks parity — bypass it). Runs on the physics queue. |
| cube | deterministic-with-changes | M | Integer permutations only; C# `FaceletCube` twin's cycle tables were ported from this exact TS. Changes: challenge carries an explicit `scrambleMoves[]` from `FaceletCube.ScrambleMoves` (no TS PRNG needed); instrument an attempt lifecycle (`executeMove` log + `isSolved()` per move); void on reset/setState/gallery playback and on `/api/cube/solve*` during an active challenge. **Metric = move count** (fully recomputable); time only as a server-window-bounded tiebreaker — client wall-clock is inherently unverifiable in the async model. |
| mountaincar | deterministic-with-changes (deferred) | S | One risky op: `Math.cos(3p)` per step (~1 ulp V8 vs .NET — can shift goal-crossing by ±1 tick). Server replay is authoritative for the published step count (absorbs the ulp), or replace cos with a shared polynomial in the `.pg` for exactness. ≤200-action trace. Marginal leaderboard value; do anytime. |
| draughts | deterministic-with-changes (post-launch, needs net-math hardening) | M–L | Engine replay is integer-exact (single-source `.pg` both sides). **Decided approach: the trace records only the HUMAN's moves** — the server re-derives the AI's replies itself during replay (apply human move → server computes AI reply → next human move must be legal in that line), so the trace cannot lie about the AI's side and the forged-AI-moves hole doesn't exist. Hard prerequisite: the AI's choice must be **bit-reproducible** TS↔C# — a divergence doesn't flag a cheater, it forks the game and rejects an honest run, and with unrecorded AI moves there is no tolerance-band escape. Today it isn't: net forward uses `Math.exp`/`Math.tanh` (not correctly rounded; parity tests are f32-tolerance only; a ulp flip on a near-tied argmax picks a different move). Required: shared polynomial exp/tanh (pure `+ - * /`) or an integer-quantized forward pass in the `.pg`; T=0 tiers or challenge-seeded sampling single-sourced in the `.pg` (`draughts-director.ts:123,141` today; special-case shipped temperatures to avoid `Math.pow`); ckpt SHA-256 + tier params pinned in the challenge and hash-checked client-side before play; `netMissing` fallback games ineligible. Cheap diagnostic: client records AI moves anyway, server ignores them for state but compares after re-derivation — a parity bug reports "diverged at ply N" instead of an opaque rejection. Plus: score metric design (win/streak/points per tier). |
| chess | deterministic-with-changes (post-launch, same scheme as draughts) | L | Same human-moves-only trace + server-derived AI replies, same net-math hardening prerequisite; additionally unseeded temperature sampling (`chess-director.ts:137,156`) and mutable checkpoint manifests to pin. |

## 8. Cross-language determinism contract

The verifier replays TS gameplay in C#. The research (V8 vs .NET, IEEE-754) fixes the rules:

- **Bit-exact and allowed**: `+ - * /` on doubles, `sqrt`, comparisons, floor/ceil/trunc/abs,
  min/max, int↔double under 2^53, `fround`↔`(float)`. Both runtimes are strict-IEEE on
  scalar doubles; neither contracts to FMA.
- **Never allowed in engine state paths**: transcendentals (`sin cos exp log pow hypot …` —
  none are correctly rounded; V8, .NET-on-Windows and .NET-on-Linux all differ), `Math.random`,
  `Date.now`/`performance.now`/rAF-dt, `Math.round` (JS half-up vs C# banker's), unstable sorts,
  hash-map iteration order.
- **Integer state wherever possible** (all board games already are). PRNGs: 32-bit-ops
  generators port exactly (`Math.imul` ⇔ unchecked `uint *`); the repo's `Xoshiro256StarStar`
  needs BigInt in TS — fine for cold paths (one draw per spawn), never in hot loops.
- **Single-source via `.pg` beats hand-mirrors**: every hand-mirrored TS engine
  (snake-logic, rush-hour-logic, 2048 ClassicEngine) is a drift risk; migrate into the twin or
  the `.pg` opportunistically, and until then gate with **differential-fuzz CI**: random seed +
  random legal input stream, both engines in lockstep (TS under Node), report the first
  divergent tick. This test is required per game before its board launches.
- **Traces are engine-versioned**; periodic state checksums (FNV-1a over canonical integer
  serialization) let the verifier bisect a divergence instead of just failing.

## 9. Detection and moderation (the anti-bot layer)

Mechanical verification can't distinguish fingers from search. Layered response:

- **GamePolicy plausibility pre-checks** (synchronous, before queuing): score/trace-length
  ceilings (snake ≤ 141 food = 144 cells − start length; starvation cap bounds snake trace
  length by claimed score), params whitelist, wall-clock floor per trace unit, tetris 30 Hz
  hypertap ceiling. Bounds verify CPU too.
- **Heartbeats** (real-time games: snake, tetris, fruit-cake): every ~5 s the client posts
  `{challengeId, seq, tick, score, stateHash}`. Server checks monotonicity and **tick-vs-wall
  pacing in both directions** (kills time compression AND gross slow-motion/savestate-editing
  of an honest run); verifier later asserts every anchor against the re-sim (catches post-hoc
  trace editing, bisects divergence). **Honestly scoped**: hashes are client-computed, so a
  coherent fabricated stream passes — heartbeats are anti-editing and pacing, never
  anti-fabrication. Documented as such.
- **Trace forensics** stored per submission: inter-input timing entropy (zero jitter =
  machine), APM p99, frame-perfect streaks (tetris frame indices and cube per-move timing are
  the richest signals), reaction time to unpredictable events (fresh food/piece spawn).
- **AI-benchmark tripwires** (the repo-specific idea): this repo publishes its own bots'
  benchmarks — snake search ~81 food, crazy-fruits net-tier averages, near-optimal ~20-move
  cube solves. A human score at or beyond the resident AI's level auto-flags. Hourly
  `AnomalySweepJob` (median/MAD z-scores, run:submission ratios) feeds the same queue.
- **Quarantine-before-display for top-N**: a would-be top-10 entry goes to the `FlagQueue`
  (PersistentObject ⇒ free admin UI with Clear/Remove actions) and **auto-promotes after 72 h
  unless flagged** — neglect fails open with forensics on record, not into a black hole.

## 10. Residual risks, stated plainly

1. **Bots/TAS**: a patient bot at 1× with jittered timing passes everything. Mitigated by §9,
   never eliminated. Board copy: "verified runs".
2. **Seed foreknowledge**: the seed is known at challenge issue; within the TTL a client can
   simulate ahead (crazy-fruits' 30-move rounds benefit most). Mitigations now: short TTL +
   heartbeat pacing; the real fix is §11's server-dealt randomness, built if abuse is observed.
3. **Client-computed heartbeat hashes are forgeable** (see §9 scoping).
4. **Verifier bugs** reject honest runs or accept subtly-wrong ones — retained traces +
   `RetroVerifySweepJob` make every such bug retroactively fixable.
5. **Account farming** around rate limits — standard abuse handling, out of scope here.

## 11. Deferred, with designated architectures

- **Server-dealt randomness** (anti-precompute, turn-based games): challenge issues no seed;
  spawns/refills are dealt just-in-time per move via the engines' existing injection APIs
  (`spawnFood`, `addSpecificTile`, scripted refills). Build when precompute abuse materializes.
- **Chess/draughts "beat tier X" boards**: async replay with **human-moves-only traces** —
  the server re-derives the AI's replies during verification (see §7), so the AI side is
  unforgeable by construction. Gated on the net-math hardening (bit-exact exp/tanh or
  integer-quantized forward pass) plus metric design and a server MCTS cost budget.
- **Cube time as a headline metric**: would need a live (server-authoritative) session for
  that one game; noted, not planned.
- **mountaincar board**: S effort, anytime; low value.
- Explicitly not doing: obfuscation, published unverified tiers, client-clock time rankings.

## 12. Milestone plan

Each milestone ends in a git commit on a passing gate (repo convention). LB-numbered here;
they take global M-numbers in PLAN.md when work starts.

### LB0 — Arcade host + pipeline skeleton (backbone, everything depends on it)
`src/Arcade` + `src/Arcade.Library` (Fleet pattern), entities of §5, `security.json` posture
(anonymous read-only boards, no client writes on `LeaderboardEntry`), Spark Authorization
login, `ArcadeController` (challenge issue + submission intake with atomic consume + intake
checks), `ReplayVerification` queue + `VerifyReplayRecipient` + `IReplayVerifier` registry,
streaming top-100 query per game, `run-session.ts` client module with taint tracking,
RLDemo.Web dev proxy for `/arcade/*`.
**Gate:** end-to-end with a stub verifier — challenge → play → submit → Pending → Verified →
entry appears on the streaming query; double-submit of one challenge rejected by concurrency;
`--spark-verify-security` baseline committed.

### LB1 — First verified boards: crazy-fruits + rush-hour
The two cheapest verifiers exercise the whole pipeline. Crazy-fruits: server seed, `(action,
centerIsA)` recorder, 30-move enforcement, freePlay rejection, engine pinned to the parity
checksum, verifier via InternalsVisibleTo or a two-arg `ApplySwap` overload. Rush-hour:
`tryMove` recorder, board-content hash pinning, verify-time par re-solve, moves-over-optimal
metric.
**Gate:** golden browser-recorded traces verify identically in C#; one deliberately tampered
trace per rejection class (wrong seed, edited move, wrong length, stale engine version,
foreign board) is rejected; live board updates over the stream.

### LB2 — game-2048 + snake
Shared TS `Xoshiro256StarStar` (BigInt, integer draw paths) with a C#↔TS parity test. 2048:
row-major spawn order, merge cap 32768 fix, pinned start, moved-only action log. Snake: seeded
food picks (or migrate human play onto the generated twin), `(tick, dir)` log, starvation cap
adopted.
**Gate:** differential-fuzz CI (random seed + legal input stream, lockstep TS-under-Node vs
C#, first-divergent-tick reporting) green for both games; golden + tampered traces as LB1.

### LB3 — tetris + the heartbeat layer
Server seed; NES-frame counter + 11-opcode ordered event log incl. `ClearInputs`; **port
`tetris-das.ts` into `tetris_solver.pg`** (the one real engine task; the existing DAS check
script becomes the parity vector); verifier over `TetrisBoard`'s micro API; >30 Hz hypertap
hard-reject. Heartbeats for real-time games: endpoint, pacing checks both directions,
PauseRatio accounting, verifier anchor assertions.
**Gate:** DAS parity vector green in both languages; a full NES-rules game replays to the
identical score; slow-motion and time-compressed submissions rejected by pacing.

### LB4 — fruit-cake + cube
Fruit-cake: tier-PRNG into the `.pg`, step-indexed drop log, hypot→sqrt, integer-step rules
port, snapshot taint, double-precision verifier on the `PhysicsReplays` queue. Cube:
challenge-served scramble, attempt instrumentation, solver-API voiding, move-count metric with
server-window time tiebreaker.
**Gate:** fruit-cake golden trace (100+ drops incl. cascades) replays to the identical f64
state checksum and score in C#; cube tampered/assisted traces void correctly.

### LB5 — detection + moderation (can trail LB3)
`AnomalySweepJob` (robust z-scores, run:submission ratio, AI-benchmark tripwires per game),
per-submission trace forensics, top-N quarantine + `FlagQueue` moderation UI + 72 h
auto-promote, `RetroVerifySweepJob`.
**Gate:** injected superhuman trace (from the repo's own bots) auto-flags; quarantine and
auto-promotion behave per policy on a seeded test board.

## 13. References

Full research reports were produced during the investigation (prior-art survey with ~40
sources, V8↔.NET float determinism analysis, Spark capability recon, per-game spikes). Key
external references: TrackMania validation replays + revalidate (BigBang1112); AntGame.io
(open-source browser game with server-issued seeds + server re-simulation — the closest
existing implementation of this design); Metanet/N leaderboard postmortem (retain replays for
retroactive sweeps); Unipop devlog (why client-side signing fails); Skillz anti-cheat docs
(plausibility + telemetry); Gaffer On Games — Floating Point Determinism / Fix Your Timestep;
Photon Quantum fixed-point manual; bryc's JS PRNG survey; speedrun.com moderation rules (the
human-verification residue).

# PG0 pilot — repro & findings (2026-07-04)

Artifacts from the PG0 validation pilot for `POLYGLOT_FRUITCAKE_PRD.md`. `fruitcake_solver.pg` is a faithful
single-source port of `FruitCakeWorld.cs` + `FruitCatalog.cs` (real 1-based 11-tier catalog, f64).

## Toolchain (all on this machine, no install needed)
- CLI is bundled in the NuGet package: `~/.nuget/packages/mintplayer.polyglot.msbuild/0.1.0/tools/win-x64/polyglot.exe`
- `polyglot check <f>.pg` → type-check; `polyglot build <f>.pg --out <dir>` → emits `<f>.cs` + `<f>.ts` (create the
  out dir first — the CLI does **not** `mkdir`).
- Run C#: drop the `.cs` into a `net10.0` `Exe` csproj (`Nullable=disable`), `dotnet run`. Run TS: `node <f>.ts`
  (Node 22+ strips types).

## Result summary
| | outcome |
|---|---|
| `.pg` type-checks | ✅ "no problems" |
| generates C# + TS | ✅ both |
| generated **C#** runs correct physics | ✅ (`nan=0` every drop) |
| generated **TS** | ❌ **every body NaN from drop 1** (`nan == bodies`) |
| cross-target byte-identity | ❌ integer outputs match for 6 drops, diverge at drop 7 |

## The blocker — generated TS goes NaN, C# doesn't → ROOT-CAUSED
Same type-checked `.pg`. The C# solver is clean; the TS solver produces NaN in **every** body's `x/y/vx/vy` from the
**first** drop (a single tier-1 fruit settling). **Root cause found (2026-07-04):** a Polyglot **codegen precedence
bug** — the expression printer drops the parentheses around a `??` subexpression under `+`. `correctPosition`'s
`invSum = a.invMass + (b?.invMass ?? 0.0)` is emitted as `a.invMass + b?.invMass ?? 0.0`, which reparses (since `??`
binds looser than `+`) as `(a.invMass + b?.invMass) ?? 0.0` → for a wall (`b` null): **C# → 0.0** (early-returns, skips
wall correction — finite but wrong) / **TS → NaN** (`invMass + undefined` = NaN, `NaN ?? 0` = NaN → all bodies NaN).

Full write-up + minimal repro + fix location: **`POLYGLOT_BUG_HANDOFF.md`** (this folder) — to be referenced from the
MintPlayer.Polyglot repo. The fix (parenthesize a child whose precedence is lower than its parent operator's) lives in
the Polyglot emitter; **PG2 (TS cutover) is blocked until it lands**. PG1 (C#) is viable once the same fix corrects the
skipped wall-correction.

## Minimal repro (drop one fruit → all TS bodies NaN)
```
CLI=~/.nuget/packages/mintplayer.polyglot.msbuild/0.1.0/tools/win-x64/polyglot.exe
"$CLI" build fruitcake_solver.pg --out gen   # after: mkdir gen
# add a nanCount() over bodies (x!=x || ...) and a main that drops one tier-1 fruit and prints count+nan
# → C#: nan=0   TS: nan=1   (all-NaN in TS from the first settle)
```
The instrumented `fc_nan.pg` variant (nanCount per drop) produced:
- C#: `d1..d7  nan=0`  (clean)
- TS: `d1 nan=1 … d6 nan=3`  (every body NaN)

## Other v0.1.0 rough edges seen
- `f32` needs an explicit `(f32)` cast on every float literal (`16.0` is `f64`) → impractical for a physics file;
  use `f64` (the sketch does). Consequence: generated C# is `double`, so adopting moves the C# solver float32→double.
- `i64` return lowers to TS `BigInt(Math.trunc(x))` → throws `RangeError` on NaN/`Infinity` input.
- Generated C# top-level types are `internal` (no access modifier emitted) — cross-assembly consumers need a facade
  or `InternalsVisibleTo` (known P11 limit).

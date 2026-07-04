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

## The blocker — generated TS goes NaN, C# doesn't
Same type-checked `.pg`. The C# solver is clean; the TS solver produces NaN in **every** body's `x/y/vx/vy` from the
**first** drop (a single tier-1 fruit settling). The upstream 6-tier `fruitcake_sketch.pg` does **not** trigger it —
so something about the real catalog/params/paths this port exercises breaks the TS codegen (or is a §3.D-class
target numerical divergence that reaches NaN). This must be root-caused/fixed **in the Polyglot repo** before the TS
cutover (PG2). The generated C# path (PG1) looks viable.

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

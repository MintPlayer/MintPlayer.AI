# Polyglot codegen bugs found porting the FruitCake inference path (v0.1.4)

Context: MintPlayer.AI M32 (`FRUITCAKE_CLIENT_SIDE_AI_PRD.md`) is single-sourcing the FruitCake **inference**
path (observation + net + search) into `fruitcake_solver.pg`, on top of the already-single-sourced physics.
While porting `buildObservation` (CS1) on **MintPlayer.Polyglot.MSBuild 0.1.4**, two codegen bugs surfaced.
Both have clean `.pg`-side workarounds (used in the port), so they are **not blocking** — but both emit code
that does not compile / would crash, so they're worth fixing (and adding to the conformance gate).

Repro CLI (no install):
```
CLI=~/.nuget/packages/mintplayer.polyglot.msbuild/0.1.4/tools/win-x64/polyglot.exe
mkdir -p out && "$CLI" build repro.pg --out out
```

---

## Bug 1 — numeric cast `T(expr)` emits `new T(expr)` (invalid in both targets)

The primitive-cast call syntax `i32(x)` / `f64(n)` type-checks (`polyglot check` is clean) but lowers to a
**constructor call** on a nonexistent type in both emitters.

`repro.pg`:
```
fn toInt(x: f64): i32 { return i32(x) }
fn toF64(n: i32): f64 { return f64(n) }
```
Generated **C#** (does not compile — there is no type `i32`/`f64`; also CS0815 in `var` contexts):
```csharp
public static int toInt(double x) { return new i32(x); }
public static double toF64(int n) { return new f64(n); }
```
Generated **TS** (runtime crash — `i32`/`f64` are not classes):
```ts
export function toInt(x: number): number { return new i32(x); }
export function toF64(n: number): number { return new f64(n); }
```
**Expected:**
- `i32(x)`: C# `(int)x` (truncate toward zero); TS `Math.trunc(x)` (NOT `x | 0`, which is 32-bit-wraps and
  differs from C# `(int)` for large/negative magnitudes — match `(int)` semantics).
- `f64(n)`: C# `(double)n`; TS just `n` (all TS numbers are f64, so the cast is identity).

**Note:** plain int/float *division* is fine — `n / 11.0` (i32/f64) and `w / n` (f64/i32) both lower correctly
to float division in C# and TS. Only the explicit `T(expr)` cast is broken.

**Workaround used in the port:** avoid `i32(...)` entirely. The observation needed a float→column-index
truncation (`(int)((b.x - b.r) / binW)`); it was rewritten as a cast-free **column-overlap test**
(`br > binW*c && bl < binW*(c+1)`), which is equivalent for all realistic positions. And int→f64 was done via
division (`tier / 11.0`) rather than `f64(tier)`.

---

## Bug 2 — typed nullable local initialized to `null` emits `var x = null` in C# (CS0815)

A mutable local with an explicit nullable annotation initialized to `null` drops its type in the C# emitter.

`repro.pg`:
```
class Box { var v: i32; init(v: i32) { this.v = v } }
fn pick(): Box? {
  var best: Box? = null
  return best
}
```
Generated **C#** (CS0815 "Cannot assign <null> to an implicitly-typed variable"):
```csharp
public static Box? pick() {
    var best = null;   // <-- should be: Box? best = null;
    return best;
}
```
Generated **TS** is fine (`let best = null;`). So this is C#-emitter-only: a `var x: T? = null` local must emit
the declared type (`T? x = null`) rather than `var x = null` when the initializer is the `null` literal.

**Workaround used in the port:** track the "biggest two fruit" by **index** (`var big1: i32 = -1`) instead of
nullable references (`var big1: PgFruitBody? = null`), sidestepping the null-initialized typed local.

---

## Conformance-gate note
Bug 1's TS output (`new i32(x)`) throws at runtime and Bug 1/2's C# output does not compile — both would be
caught by a gate that (a) compiles the generated C# and (b) runs the generated TS, rather than only diffing
stdout of hand-picked samples. (Same lesson as the §6 note in `POLYGLOT_BUG_HANDOFF.md`.)

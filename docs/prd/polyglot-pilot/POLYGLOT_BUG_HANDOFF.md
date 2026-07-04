# Handoff → MintPlayer.Polyglot: null-coalescing precedence bug (dropped parentheses)

**For:** a session working in `C:\Repos\MintPlayer.Polyglot`.
**From:** the MintPlayer.AI FruitCake single-source pilot (PG0, see `POLYGLOT_FRUITCAKE_PRD.md` / `REPRO.md`).
**Severity:** high — silently wrong output on **both** targets, and **cross-target divergent** (C# vs TS differ), which
defeats the whole byte-identical promise. It makes the real FruitCake solver produce **NaN in every body in TS**.
**Version:** MintPlayer.Polyglot **0.1.0** (CLI bundled at `~/.nuget/packages/mintplayer.polyglot.msbuild/0.1.0/tools/win-x64/polyglot.exe`), Node v24, .NET 10.

---

## 1. The bug in one line
The expression printer **drops precedence-required parentheses** around a `??` (null-coalescing) subexpression when it
is an operand of a higher-precedence operator (e.g. `+`). Source `base + (opt?.v ?? 0.0)` is emitted as
`base + opt?.v ?? 0.0`. Because `??` binds **looser** than `+` in both C# and TS, that reparses as
`(base + opt?.v) ?? 0.0` — wrong, and it diverges by target.

## 2. Minimal repro
Drop this into `tests/conformance/programs/precedence_null_coalesce.pg` (it currently **fails** the C#==TS gate — that
failure is the regression test):

```
import { print } from "std.io"

class Box { var v: f64;  init(v: f64) { this.v = v } }

fn addOpt(base: f64, opt: Box?): f64 {
  return base + (opt?.v ?? 0.0)
}

fn classify(v: f64): i32 => if v != v { -1 } else { (i32)(v + 0.5) }

fn main() {
  let n = addOpt(5.0, null)     // expect 5.0
  print("null=${classify(n)}")  // EXPECT (both targets): null=5
}
```

## 3. What v0.1.0 emits (both targets drop the parens)
```
// C#  (paren_bug2.cs)
return base + opt?.v ?? 0.0;
// TS  (paren_bug2.ts)
return base + opt?.v ?? 0.0;
```

## 4. Observed vs expected
| target | prints | why | expected |
|---|---|---|---|
| C# | `null=0` | `(5.0 + (double?)null) ?? 0.0` → `null ?? 0.0` → **0.0** | `null=5` |
| TS | `null=-1` | `(5.0 + undefined) ?? 0.0` → `NaN ?? 0.0` → **NaN** (`??` ignores NaN) → `classify`=-1 | `null=5` |

Both are wrong (should be `5.0`); worse, they **disagree** (`0.0` vs `NaN`).

## 5. Root cause & fix
- **Precedence:** in C# and TS/JS, `??` has **lower** precedence than `+` (and than `*`, `-`, `/`, comparisons…). The
  source parenthesized the `??` deliberately; the printer must preserve/insert those parens.
- **Fix (expression printer):** when printing a child expression whose operator precedence is **lower** than the
  parent operator's, wrap it in parentheses. This is the standard pretty-printer precedence rule; v0.1.0 handles most
  cases but misses `??` (and probably other low-precedence operators) nested under arithmetic. Apply it uniformly in
  **both** the C# and TS emitters (same AST → same paren decision keeps targets aligned).
- **Suggested regression coverage:** `a + (x ?? y)`, `(a ?? b) * c`, `a - (b ?? c)`, `a && (b ?? c)`, ternary/`??`
  mixes, and `?.` chains under arithmetic (the `?.`→`undefined`/`null` split is what turns this into NaN-vs-0).

## 6. Why the existing FruitCake conformance sample didn't catch it (gate soundness)
`FruitCakeWorld.correctPosition` in the north-star sample has the **same** line
(`invSum = a.invMass + (b?.invMass ?? 0.0)`), so the sample's generated TS is very likely **also latently NaN**, yet
`fruitcake_sketch.pg` passes the conformance gate. The gate compares **integer/string stdout only**, and the sample's
short scripted cascade happens to yield matching integer output despite the underlying float state diverging (in the
MintPlayer.AI port, C# and TS integer outputs match for 6 scripted drops, then diverge on the 7th). **Recommendation:**
strengthen the gate for numeric programs — e.g. assert "no NaN/Inf in final state" and/or a scaled-integer checksum of
the float state, not just the printed integers. Otherwise silently-divergent float codegen passes.

## 7. Repro commands (from anywhere; no install)
```
CLI=~/.nuget/packages/mintplayer.polyglot.msbuild/0.1.0/tools/win-x64/polyglot.exe
mkdir -p out && "$CLI" build precedence_null_coalesce.pg --out out    # CLI does NOT mkdir the out dir
# C#: drop out/precedence_null_coalesce.cs into a net10.0 Exe csproj (Nullable=disable), dotnet run  -> null=0
# TS: node out/precedence_null_coalesce.ts                                                            -> null=-1
```

## 8. Minor v0.1.0 rough edges seen during the pilot (not blocking, FYI)
- `f32` requires an explicit `(f32)` cast on **every** float literal (`16.0` is `f64`) — impractical for a numeric
  file; the FruitCake port uses `f64`.
- `i64` return lowers to TS `BigInt(Math.trunc(x))`, which **throws** on `NaN`/`Infinity` input.
- The CLI `build --out <dir>` does not create `<dir>` (errors "cannot write") — must `mkdir` first.
- Generated top-level C# types are `internal` (known P11 limit) — cross-assembly consumers need a facade / public
  emission.

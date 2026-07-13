# CLAUDE.md — operational rules for this repo

Project-level instructions loaded every session. Read `docs/ARCHITECTURE.md` for the code map and
`docs/prd/PLAN.md` for milestone history.

## Running / verifying the web app (RLDemo.Web) — READ THIS FIRST

**The .NET host is ALREADY RUNNING. The user runs and owns it. Do NOT start, stop, kill, restart, or
`taskkill` it — ever. Assume it is up at http://localhost:5210 and serving the Angular frontend.**

The ASP.NET Core host **builds and serves the Angular frontend itself** — in Development it spawns and
proxies the Angular dev server (`UseAngularCliServer` via `UseSpaImproved` in `Program.cs`). The Angular app
is NOT a separate process you manage; it lives inside the running .NET host.

- **Do NOT run `dotnet run --project src/RLDemo.Web` yourself** — it is already running. A second instance
  fights for the port. (This command is how the *user* starts it; it is not your job.)
- **Never** run `ng serve` / `npm start` / `ng build` / `ng test`, and never `taskkill`/kill the `dotnet` /
  `RLDemo.Web.exe` / `node` (ng serve) processes.
- **To see a code change:** just save the file under `ClientApp/src` — the running host live-reloads the
  browser. No manual build, no restart, usually no manual reload.
- **To verify what's actually served** (suspected staleness): `curl -sk http://localhost:5210/main.js | grep <identifier>`
  — do not reach for `ng build`, and do not restart the host, to "check".
- **If output looks stale** (wedged dev-server watcher) **or new npm dependencies were added** (the host must
  re-read them): **ASK THE USER to restart their host.** Do not restart it yourself.

## Build failures are usually NOT the frontend

If `dotnet run --project src/RLDemo.Web` fails, read the error before touching anything Angular-related.

**Fixed (2026-07-13, Polyglot 0.6.0): the multi-`.pg` incremental-rebuild codegen bug** (`CS0260` "missing
partial modifier on PolyglotProgram" / `CS0101` duplicate `Option`/`Some`/`None` prelude). It was an MSBuild
`.targets` bug — a single-`.pg` edit made MSBuild's partial-incremental build hand the transpiler a subset,
which then emitted a standalone/duplicate prelude. `MintPlayer.Polyglot.MSBuild` **0.6.0** (PR #26, stamp
`Outputs` + `RemoveDir`) re-transpiles the full `.pg` set on any edit, so this no longer occurs. If you somehow
hit it on an **older** package, the manual recovery was
`rm -f src/MintPlayer.AI.ReinforcementLearning.Environments/obj/*/net10.0/polyglot/*.cs` then rebuild. History +
root cause: `docs/prd/polyglot-pilot/POLYGLOT_TOPLEVEL_RECORD_BUG.md`. Never loop build retries against stale output.

## Stopping a `dotnet run` host

`dotnet run` is a runner that spawns the app (`<App>.exe`) as a child. To stop it, kill the runner's whole
process tree (the background shell/task that launched it), not just the child exe — otherwise the next build
fails with a DLL lock (MSB3027/MSB3021). Verify nothing is left with
`Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'dotnet.exe','RLDemo.Web.exe' }`.

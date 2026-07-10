# CLAUDE.md — operational rules for this repo

Project-level instructions loaded every session. Read `docs/ARCHITECTURE.md` for the code map and
`docs/prd/PLAN.md` for milestone history.

## Running / verifying the web app (RLDemo.Web) — READ THIS FIRST

**To build, serve, run, or visually verify the frontend, do exactly one thing:**

```bash
dotnet run --project src/RLDemo.Web        # Development profile; http://localhost:5210
```

The ASP.NET Core host **builds and serves the Angular frontend itself** — in Development it spawns and
proxies the Angular dev server (`UseAngularCliServer` via `UseSpaImproved` in `Program.cs`). There is
**nothing else to do for the frontend.**

- **Never** run `ng serve` / `npm start` / `ng build` / `ng test` yourself. The host already runs one; a
  second instance just fights for ports and can wedge the dev-server file watcher.
- **To see a code change:** just save the file under `ClientApp/src` — the running host live-reloads the
  browser. No manual build, usually no manual reload.
- **To verify what's actually served** (suspected staleness): `curl -sk http://localhost:5210/main.js | grep <identifier>`
  — do not reach for `ng build` to "check".
- If output looks stale, suspect a wedged dev-server watcher: **restart the ASP.NET host**, never `ng build`.

## Build failures are usually NOT the frontend

If `dotnet run --project src/RLDemo.Web` fails, read the error before touching anything Angular-related.

**Known: stale Polyglot codegen (`CS0260` "missing partial modifier on PolyglotProgram" / `CS0101`
duplicate `Option`/`Some`/`None` prelude).** This is the documented multi-`.pg` incremental-rebuild
transpiler bug (see `docs/prd/polyglot-pilot/POLYGLOT_TOPLEVEL_RECORD_BUG.md`) — the CLI doesn't clean its
`--out` dir, so a stale duplicate lingers in `obj/`. It is unrelated to the frontend and to your changes.
Fix by clearing the stale generated files, then run again:

```bash
rm -f src/MintPlayer.AI.ReinforcementLearning.Environments/obj/*/net10.0/polyglot/*.cs
dotnet run --project src/RLDemo.Web
```

Clean/CI builds are unaffected. Never loop build retries against the stale output.

## Stopping a `dotnet run` host

`dotnet run` is a runner that spawns the app (`<App>.exe`) as a child. To stop it, kill the runner's whole
process tree (the background shell/task that launched it), not just the child exe — otherwise the next build
fails with a DLL lock (MSB3027/MSB3021). Verify nothing is left with
`Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'dotnet.exe','RLDemo.Web.exe' }`.

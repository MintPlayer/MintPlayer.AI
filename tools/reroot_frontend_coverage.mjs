#!/usr/bin/env node
// Re-roots the Angular/vitest cobertura report so coverage.mintplayer.com can resolve its files.
//
// WHY THIS EXISTS. The service resolves a report path by suffix-matching it against `git ls-files`,
// which lists forward-slashed paths relative to the REPO root. The `cobertura` reporter writes
// filenames relative to the ANGULAR PROJECT root (`src/RLDemo.Web/ClientApp`) using the platform
// separator — so on a Windows agent a class lands as `src\app\snake\snake-logic.ts`, which is a
// suffix of nothing in git and is silently counted as unmatched. Nothing errors; the file just
// vanishes from the report, which is the worst possible failure mode for a coverage number.
//
// So: normalise separators to `/`, and prefix the project root — turning `src/app/snake/snake-logic.ts`
// into `src/RLDemo.Web/ClientApp/src/app/snake/snake-logic.ts`, which IS a git path.
//
// Deliberately dependency-free (it runs before `npm ci` would matter) and deliberately NOT a general
// XML rewriter: it touches only `filename=` attributes and the `<source>` element, so a malformed or
// unexpected report is passed through rather than mangled.
//
// Usage: node tools/reroot_frontend_coverage.mjs <cobertura.xml> <repo-relative project root>

import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const [, , reportPath, projectRoot] = process.argv;

if (!reportPath || !projectRoot) {
  console.error('usage: reroot_frontend_coverage.mjs <cobertura.xml> <repo-relative-project-root>');
  process.exit(2);
}

if (!existsSync(reportPath)) {
  // Not an error: the suite may have been skipped, and the upload step has its own hashFiles guard.
  console.log(`no report at ${reportPath} — nothing to re-root`);
  process.exit(0);
}

const prefix = projectRoot.replace(/\\/g, '/').replace(/\/+$/, '');
const xml = readFileSync(reportPath, 'utf8');

let rerooted = 0;
let alreadyRooted = 0;

// Drop the Polyglot-generated twins from THIS report. They are in `coverageInclude` so that
// `pg_coverage_remap.mjs` can project them onto their `.pg` sources, but they must not be uploaded
// under their own `.ts` paths: those are gitignored build outputs, so the service — which resolves a
// path by suffix-matching `git ls-files` — would drop them as unmatched anyway. Keeping them here
// would also double-count, since the remapped `.pg` report already carries their hits.
// A twin is identified by the `_solver.ts` suffix; `*_solver.spec.ts` is hand-written and is
// excluded from coverage upstream, so it cannot reach this file.
const TWIN = /_solver\.ts$/;
let twinsDropped = 0;

const withoutTwins = xml.replace(/[ \t]*<class\b[^>]*\bfilename="([^"]*)"[\s\S]*?<\/class>\s*/g, (whole, raw) => {
  if (TWIN.test(raw.replace(/\\/g, '/'))) {
    twinsDropped++;
    return '';
  }
  return whole;
});

const out = withoutTwins
  // <class ... filename="src\app\x.ts"> → filename="src/RLDemo.Web/ClientApp/src/app/x.ts"
  .replace(/filename="([^"]*)"/g, (whole, raw) => {
    const path = raw.replace(/\\/g, '/');
    if (path.startsWith(`${prefix}/`)) {
      alreadyRooted++;
      return `filename="${path}"`;
    }
    rerooted++;
    return `filename="${prefix}/${path.replace(/^\.?\//, '')}"`;
  })
  // <sources><source>…</source></sources> points at the absolute agent path, which means nothing to
  // the service and can only mislead a reader of the raw report. Point it at the repo root instead.
  .replace(/<source>[\s\S]*?<\/source>/g, '<source>.</source>');

writeFileSync(reportPath, out, 'utf8');

console.log(
  `re-rooted ${rerooted} path(s) under ${prefix}/` +
    (alreadyRooted ? `; ${alreadyRooted} already rooted` : '') +
    (twinsDropped ? `; dropped ${twinsDropped} Polyglot twin(s) (the .pg report owns those)` : ''),
);

if (rerooted === 0 && alreadyRooted === 0) {
  // Loud, because an empty report that uploads cleanly is indistinguishable from a healthy one on
  // the service side, and would quietly drop the frontend out of the merged number.
  console.warn('WARNING: no filename attributes found — the frontend report covers nothing');
}

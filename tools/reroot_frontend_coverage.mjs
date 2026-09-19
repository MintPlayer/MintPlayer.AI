#!/usr/bin/env node
// Prepares the Angular/vitest cobertura report for upload: drops the Polyglot twins, and re-roots the
// remaining paths from project-relative to repo-relative.
//
// WHY THIS STILL EXISTS, given the upload action rebases paths itself (MintPlayer.Spark#416).
// That rebasing STRIPS a `GITHUB_WORKSPACE` prefix and unifies separators. It cannot ADD one, and has
// no input for a base path — so it fixes a collector that wrote absolute native paths, which is not
// our case. The `cobertura` reporter writes filenames relative to the ANGULAR PROJECT root
// (`src/RLDemo.Web/ClientApp`), e.g. `src\app\snake\snake-logic.ts`. Only this repo knows the prefix
// that turns that into a git path.
//
// Strictly, the server's suffix match would resolve the short form anyway: all 13 covered modules
// have exactly one candidate in `git ls-files` today (checked). Re-rooting is kept because the match
// then needs no uniqueness to hold — a second `src/app/**` tree would make several of them ambiguous,
// and an ambiguous path resolves to nothing rather than to the wrong file.
//
// Dropping the twins is the half that has no alternative. They are in `coverageInclude` so that the
// `.pg` remap can read them, but they are gitignored build outputs: uploaded under their own `.ts`
// paths they resolve to nothing, and since #416 an unmatched path is reported as an incomplete
// reason — so leaving them in would mark every build incomplete. The `.pg` report carries their hits.
//
// Deliberately dependency-free and deliberately not a general XML rewriter: it touches only `filename=`
// attributes, whole `<class>` elements and `<source>`, so an unexpected report passes through rather
// than being mangled.
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

// A twin is identified by the `_solver.ts` suffix. `*_solver.spec.ts` is hand-written, and is excluded
// from coverage upstream, so it cannot reach this file.
const TWIN = /_solver\.ts$/;
let twinsDropped = 0;

const withoutTwins = xml.replace(/[ \t]*<class\b[^>]*\bfilename="([^"]*)"[\s\S]*?<\/class>\s*/g, (whole, raw) => {
  if (TWIN.test(raw.replace(/\\/g, '/'))) {
    twinsDropped++;
    return '';
  }
  return whole;
});

let rerooted = 0;
let alreadyRooted = 0;

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
  // <source> points at the absolute agent path, which means nothing to the service and can only
  // mislead a reader of the raw report. The action does not touch it.
  .replace(/<source>[\s\S]*?<\/source>/g, '<source>.</source>');

writeFileSync(reportPath, out, 'utf8');

console.log(
  `re-rooted ${rerooted} path(s) under ${prefix}/` +
    (alreadyRooted ? `; ${alreadyRooted} already rooted` : '') +
    (twinsDropped ? `; dropped ${twinsDropped} Polyglot twin(s) (the .pg report owns those)` : ''),
);

// No "covered nothing" warning here any more. It was a proxy for "this upload will produce an empty
// report", and since #416 the server answers that question directly with its unmatched-file verdict —
// which is accurate where a rewrite count only guesses.

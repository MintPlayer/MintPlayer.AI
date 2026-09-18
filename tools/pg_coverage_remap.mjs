#!/usr/bin/env node
// pg_coverage_remap.mjs — project TypeScript coverage back onto the `.pg` sources (M63.6 / spike S5).
//
// WHY THIS EXISTS
// ---------------
// The C# half of `.pg` coverage needs no tooling: Polyglot emits `#line` pragmas, Roslyn writes them
// into the PDB, and coverlet reports straight against the `.pg`. The TypeScript half cannot work that
// way. `@angular/build:unit-test` pre-builds the app with esbuild before Vitest starts, so the
// generated `*_solver.ts` never passes through Vite's transform hooks — a Vite plugin returning the
// Polyglot `.ts.map` as an input map is simply never consulted (verified 2026-09-18: the config file
// loads, the plugin does not fire, and istanbul's output carries no `inputSourceMap`).
//
// So the composition is done here instead, after the fact: istanbul statement positions are in `.ts`
// coordinates, the sibling `*_solver.ts.map` maps those to `.pg` lines, and the result is written as
// a cobertura report keyed on the `.pg` files. coverage.mintplayer.com then merges it with the C#
// report under the same commit using max semantics, which is what makes a `.pg` line read as covered
// when EITHER target exercised it.
//
// Polyglot emits line-granular mappings (one segment per output line, column 0), so this maps line to
// line and never pretends to column fidelity it does not have.
//
// USAGE
//   node tools/pg_coverage_remap.mjs <coverage-final.json> <output.cobertura.xml> [repoRoot]

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve, relative, isAbsolute } from 'node:path';

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const B64INV = new Map([...B64].map((c, i) => [c, i]));

/** Decode one base64-VLQ run into signed integers. */
function decodeVlq(segment) {
  const out = [];
  let shift = 0;
  let value = 0;
  for (const ch of segment) {
    const digit = B64INV.get(ch);
    if (digit === undefined) throw new Error(`bad base64-VLQ character: ${ch}`);
    const cont = digit & 32;
    value += (digit & 31) << shift;
    if (cont) {
      shift += 5;
    } else {
      const negative = value & 1;
      value >>= 1;
      out.push(negative ? (value === 0 ? -0x80000000 : -value) : value);
      shift = 0;
      value = 0;
    }
  }
  return out;
}

/**
 * Build generatedLine(1-based) -> {sourceIndex, originalLine(1-based)}.
 * Fields are deltas against running state, and the generated-column field resets per line.
 */
function buildLineMap(mappings) {
  const map = new Map();
  let sourceIndex = 0;
  let originalLine = 0;
  const lines = mappings.split(';');
  for (let genLine = 0; genLine < lines.length; genLine++) {
    const raw = lines[genLine];
    if (!raw) continue;
    for (const seg of raw.split(',')) {
      if (!seg) continue;
      const f = decodeVlq(seg);
      if (f.length < 4) continue; // a 1-field segment carries no source position
      sourceIndex += f[1];
      originalLine += f[2];
      // First segment on a generated line wins: Polyglot emits one per line anyway.
      if (!map.has(genLine + 1)) {
        map.set(genLine + 1, { sourceIndex, originalLine: originalLine + 1 });
      }
    }
  }
  return map;
}

function toPosix(p) {
  return p.split('\\').join('/');
}

function main() {
  const [covPath, outPath, rootArg] = process.argv.slice(2);
  if (!covPath || !outPath) {
    console.error('usage: node tools/pg_coverage_remap.mjs <coverage-final.json> <out.xml> [repoRoot]');
    process.exit(2);
  }
  const repoRoot = resolve(rootArg ?? process.cwd());

  if (!existsSync(covPath)) {
    console.error(`no istanbul coverage at ${covPath}`);
    process.exit(1);
  }
  const cov = JSON.parse(readFileSync(covPath, 'utf-8'));

  // pgFile (absolute) -> Map<pgLine, hits>
  const perFile = new Map();
  let mappedFiles = 0;
  let skipped = [];

  for (const [tsPath, entry] of Object.entries(cov)) {
    if (!tsPath.endsWith('_solver.ts')) continue;

    const mapPath = `${tsPath}.map`;
    if (!existsSync(mapPath)) {
      skipped.push(`${tsPath} (no sibling .ts.map -- was PolyglotOriginInfo=true when it was generated?)`);
      continue;
    }

    const sm = JSON.parse(readFileSync(mapPath, 'utf-8'));
    const lineMap = buildLineMap(sm.mappings ?? '');
    const sources = (sm.sources ?? []).map((s) =>
      isAbsolute(s) ? resolve(s) : resolve(dirname(mapPath), s),
    );

    const statementMap = entry.statementMap ?? {};
    const counts = entry.s ?? {};
    let hitStatements = 0;

    for (const [id, loc] of Object.entries(statementMap)) {
      const genLine = loc?.start?.line;
      if (!genLine) continue;
      const origin = lineMap.get(genLine);
      if (!origin) continue; // generated scaffolding with no .pg origin
      const pgFile = sources[origin.sourceIndex];
      if (!pgFile) continue;

      const hits = Number(counts[id] ?? 0);
      if (hits > 0) hitStatements++;

      if (!perFile.has(pgFile)) perFile.set(pgFile, new Map());
      const lines = perFile.get(pgFile);
      // Several .ts statements collapse onto one .pg line (one-line loops, one-line fn bodies).
      // Take the MAX: the line is covered if any contributing statement ran. This mirrors what
      // coverlet does on the C# side and what the coverage service does when merging reports.
      lines.set(origin.originalLine, Math.max(lines.get(origin.originalLine) ?? 0, hits));
    }

    mappedFiles++;
    console.log(
      `  ${toPosix(relative(repoRoot, tsPath))} -> ${toPosix(relative(repoRoot, sources[0] ?? '?'))}` +
        ` (${hitStatements}/${Object.keys(statementMap).length} statements hit)`,
    );
  }

  for (const s of skipped) console.warn(`  SKIPPED ${s}`);

  if (perFile.size === 0) {
    console.error('no .pg coverage produced -- nothing was remapped');
    process.exit(1);
  }

  // ---- cobertura ------------------------------------------------------------------------
  let totalValid = 0;
  let totalCovered = 0;
  const classes = [];

  for (const [pgFile, lines] of [...perFile.entries()].sort()) {
    const rel = toPosix(relative(repoRoot, pgFile));
    const nums = [...lines.keys()].sort((a, b) => a - b);
    const covered = nums.filter((n) => lines.get(n) > 0).length;
    totalValid += nums.length;
    totalCovered += covered;

    const name = rel.split('/').pop().replace(/\.pg$/, '');
    const rate = nums.length ? (covered / nums.length).toFixed(4) : '0';
    const lineXml = nums
      .map((n) => `          <line number="${n}" hits="${lines.get(n)}" branch="false" />`)
      .join('\n');

    classes.push(
      `      <class name="${name}" filename="${rel}" line-rate="${rate}" branch-rate="0" complexity="0">\n` +
        `        <methods />\n        <lines>\n${lineXml}\n        </lines>\n      </class>`,
    );
  }

  const rate = totalValid ? (totalCovered / totalValid).toFixed(4) : '0';
  const xml =
    `<?xml version="1.0" encoding="utf-8"?>\n` +
    `<coverage line-rate="${rate}" branch-rate="0" lines-covered="${totalCovered}" ` +
    `lines-valid="${totalValid}" branches-covered="0" branches-valid="0" complexity="0" ` +
    `version="1.9" timestamp="${Math.floor(Date.now() / 1000)}">\n` +
    `  <sources>\n    <source>${toPosix(repoRoot)}</source>\n  </sources>\n` +
    `  <packages>\n    <package name="Polyglot" line-rate="${rate}" branch-rate="0" complexity="0">\n` +
    `      <classes>\n${classes.join('\n')}\n      </classes>\n    </package>\n  </packages>\n</coverage>\n`;

  // CI writes into a directory that does not exist yet.
  const outDir = dirname(resolve(outPath));
  if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });
  writeFileSync(outPath, xml, 'utf-8');
  console.log(
    `\n${mappedFiles} twin(s) remapped -> ${perFile.size} .pg file(s), ` +
      `${totalCovered}/${totalValid} lines (${(rate * 100).toFixed(2)}%)`,
  );
  console.log(`wrote ${outPath}`);
}

main();

// M57.1 verification — the WIDENED evaluator as it now lives in the engine (tetris_solver.pg), measured
// through the generated TS twin (the browser's exact code), against the recorded M54 baselines.
//
// Recorded M54 baselines (data/tetris-baselines-final.txt, 100 eps, seeds 5000+e):
//   Dellacherie   protocol A 94,636 score / 197.6 lines / 0.26 tetrises  ·  protocol B survival 363.8 ± 40.3
//   della-search  protocol A 93,678                                       ·  protocol B survival 1480 (censored)
//
// Run:  node docs/prd/tetris-spike/s4_verify_widened.mjs [episodes]

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;

const EPS = Number(process.argv[2] ?? 30);
const CAP_A = 500, CAP_B = 1500;

function play(seed, garbage, cap, tier, tapRate) {
  const b = new PgTetris();
  b.reset(seed, false, garbage);
  if (tapRate !== undefined) b.setTapRate(tapRate);
  let n = 0;
  while (!b.gameOver && n < cap) {
    const a = tier === 'della' ? b.dellacherieAction() : b.dellaSearchAction(8, 5);
    if (a < 0 || !b.placementLegal(a)) break;
    b.applyPlacement(a);
    n++;
  }
  return { pieces: n, score: b.score, lines: b.lines, tetrises: b.tetrises };
}
function agg(rows) {
  const n = rows.length, m = f => rows.reduce((a, r) => a + f(r), 0) / n;
  const s = m(r => r.score), p = m(r => r.pieces), li = m(r => r.lines), te = m(r => r.tetrises);
  const sdS = Math.sqrt(rows.reduce((a, r) => a + (r.score - s) ** 2, 0) / Math.max(1, n - 1));
  const sdP = Math.sqrt(rows.reduce((a, r) => a + (r.pieces - p) ** 2, 0) / Math.max(1, n - 1));
  return { score: s, sCi: 1.96 * sdS / Math.sqrt(n), pieces: p, pCi: 1.96 * sdP / Math.sqrt(n),
           lines: li, tetrises: te, trt: li > 0 ? 400 * te / li : 0 };
}

console.log(`M57.1 verification — widened evaluator IN THE ENGINE, ${EPS} eps, seeds 5000+\n`);
console.log('protocol A (uniform, no garbage, 500-piece cap)          M54 recorded baseline');
console.log('-'.repeat(94));
for (const tier of ['della', 'search']) {
  const r = agg([...Array(EPS)].map((_, e) => play(5000 + e, 0, CAP_A, tier)));
  const base = tier === 'della' ? '94,636 / 197.6 lines / 0.26 tet' : '93,678';
  console.log(`${(tier === 'della' ? 'dellacherie' : 'della-search').padEnd(14)} score ${r.score.toFixed(0).padStart(8)} ±${r.sCi.toFixed(0).padStart(6)}  lines ${r.lines.toFixed(1).padStart(6)}  tet ${r.tetrises.toFixed(2).padStart(6)}  TRT ${r.trt.toFixed(1).padStart(5)}%   was ${base}`);
}

console.log('\nprotocol B (garbage/10, survival)                        M54 recorded baseline');
console.log('-'.repeat(94));
for (const tier of ['della', 'search']) {
  const r = agg([...Array(EPS)].map((_, e) => play(5000 + e, 10, CAP_B, tier)));
  const base = tier === 'della' ? '363.8 ± 40.3 pieces' : '1480 (right-censored)';
  console.log(`${(tier === 'della' ? 'dellacherie' : 'della-search').padEnd(14)} pieces ${r.pieces.toFixed(1).padStart(7)} ±${r.pCi.toFixed(1).padStart(6)}  lines ${r.lines.toFixed(1).padStart(6)}  tet ${r.tetrises.toFixed(2).padStart(6)}  TRT ${r.trt.toFixed(1).padStart(5)}%   was ${base}`);
}

console.log('\ntap dial (protocol A, dellacherie) — the reachability terms should bite only at high gravity');
console.log('-'.repeat(94));
for (const [name, rate] of [['DAS 10Hz', 6], ['hyper 12Hz', 5], ['rolling 20Hz', 3]]) {
  const r = agg([...Array(EPS)].map((_, e) => play(5000 + e, 0, CAP_A, 'della', rate)));
  console.log(`${name.padEnd(14)} score ${r.score.toFixed(0).padStart(8)} ±${r.sCi.toFixed(0).padStart(6)}  lines ${r.lines.toFixed(1).padStart(6)}  tet ${r.tetrises.toFixed(2).padStart(6)}  TRT ${r.trt.toFixed(1).padStart(5)}%`);
}

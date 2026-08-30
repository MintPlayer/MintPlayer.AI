// M57 gate G7 + G6 — the HIGH-GRAVITY protocol, run against the real engine (TS twin).
//
// Why this exists: S0–S2 all started at level 0 (48 frames/row), where tap speed constrains nothing, so
// every number they produced is blind to the regime real NES play happens in. S3 showed the effect is
// decisive from level 19 up. This measures the SHIPPED engine — widened evaluator (M57.1) with the
// inaccessible-wall terms and the tap dial — under NES start levels.
//
// G7: report a start-level-18/19 protocol alongside the level-0 one.
// G6: the three technique settings, PAIRED per seed (the piece stream is seed-determined and
//     setting-independent, so pairing is exact and cancels most of the seed variance).
//
// Run:  node docs/prd/tetris-spike/s6_g7_protocol.mjs [episodes]

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;

const EPS = Number(process.argv[2] ?? 20);
const CAP = 400;
const RATES = [['DAS 10Hz', 6], ['hyper 12Hz', 5], ['rolling 20Hz', 3]];
const STARTS = [0, 18, 19, 29];

function play(seed, startLevel, tap, tier) {
  const b = new PgTetris();
  b.reset(seed, false, 0);
  b.setStartLevel(startLevel);
  b.setTapRate(tap);
  let n = 0;
  while (!b.gameOver && n < CAP) {
    const a = tier === 'search' ? b.dellaSearchAction(8, 5) : b.dellacherieAction();
    if (a < 0 || !b.placementLegal(a)) break;
    b.applyPlacement(a);
    n++;
  }
  return { pieces: n, score: b.score, lines: b.lines, tetrises: b.tetrises, level: b.level };
}
function agg(rows) {
  const n = rows.length, m = f => rows.reduce((a, r) => a + f(r), 0) / n;
  const s = m(r => r.score), li = m(r => r.lines), te = m(r => r.tetrises);
  const sd = Math.sqrt(rows.reduce((a, r) => a + (r.score - s) ** 2, 0) / Math.max(1, n - 1));
  return { score: s, ci: 1.96 * sd / Math.sqrt(n), pieces: m(r => r.pieces), lines: li,
           tetrises: te, trt: li > 0 ? 400 * te / li : 0 };
}

for (const tier of ['della', 'search']) {
  console.log(`\n=== tier: ${tier === 'della' ? 'dellacherie (widened)' : 'della-search(8,5)'} — ${EPS} eps, ${CAP}-piece cap, seeds 5000+`);
  console.log('start level  rate            score       ±CI   pieces   lines  tetris   TRT%');
  console.log('-'.repeat(82));
  const byStart = {};
  for (const st of STARTS) {
    byStart[st] = {};
    for (const [rn, p] of RATES) {
      const rows = [...Array(EPS)].map((_, e) => play(5000 + e, st, p, tier));
      byStart[st][rn] = rows;
      const r = agg(rows);
      console.log(`L${String(st).padEnd(11)} ${rn.padEnd(14)} ${r.score.toFixed(0).padStart(8)} ${r.ci.toFixed(0).padStart(9)} ${r.pieces.toFixed(0).padStart(8)} ${r.lines.toFixed(1).padStart(7)} ${r.tetrises.toFixed(2).padStart(7)} ${r.trt.toFixed(1).padStart(6)}`);
    }
  }

  // G6: PAIRED per-seed deltas — the piece stream is identical across settings for a given seed.
  console.log('\n  G6 paired deltas vs DAS (same seeds; bootstrap 95% CI on the mean delta):');
  for (const st of STARTS) {
    for (const [rn] of RATES.slice(1)) {
      const d = byStart[st][rn].map((r, i) => r.score - byStart[st]['DAS 10Hz'][i].score);
      const mean = d.reduce((a, x) => a + x, 0) / d.length;
      const sd = Math.sqrt(d.reduce((a, x) => a + (x - mean) ** 2, 0) / Math.max(1, d.length - 1));
      const ci = 1.96 * sd / Math.sqrt(d.length);
      const sig = Math.abs(mean) > ci ? 'SIGNIFICANT' : 'ns';
      console.log(`    L${String(st).padEnd(3)} ${rn.padEnd(14)} Δscore ${mean >= 0 ? '+' : ''}${mean.toFixed(0).padStart(8)} ± ${ci.toFixed(0).padStart(7)}   ${sig}`);
    }
  }
}

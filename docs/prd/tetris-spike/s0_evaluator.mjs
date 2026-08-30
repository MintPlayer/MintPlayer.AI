// M57 spike S0 — "is the low tetris rate the TARGET FUNCTION, not the horizon?"
//
// TETRIS_TECHNIQUES_PRD.md §0 claims the dense regression target (= the Dellacherie basis) is
// ANTI-TETRIS: its −20·Δwells term penalizes the well a tetris requires, and +8·eroded pays for
// clearing lines now. If that is right, then simply WIDENING the evaluator with StackRabbit's
// tetris-aware terms should raise the tetris rate with NO training, NO action-space change and NO
// engine change — because every scripted tier and the dense target all read the same basis.
//
// This spike changes nothing in the .pg. It re-implements the evaluator OUTSIDE the engine, over
// the same board state (engine features are reused via the twin's own methods), and plays protocol A.
//
// Gate (PRD §6 S0): GO if some weighting reaches >= 2.0 tetrises/episode while keeping score >= 85,000.
// NO-GO if every weighting that raises tetrises drops score below ~70,000.
//
// Run:  node docs/prd/tetris-spike/s0_evaluator.mjs [episodesPerConfig]

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;

const W = 10, H = 20, ACTIONS = 40;
const WELL = 9;                     // right well: the I-piece favours it and it is 4 taps from spawn
const EPS = Number(process.argv[2] ?? 30);
const PIECE_CAP = 500;              // protocol A

// ── board-shape features the Dellacherie basis cannot express ────────────────────────────────────
function colHeight(b, c) {
  for (let y = 0; y < H; y++) if ((b.rows[y] >> c & 1) === 1) return H - y;
  return 0;
}
// wellSum with the designated well column EXCLUDED — the sign-trap fix (PRD §3.1 item 1)
function wellSumExcluding(b, skipCol) {
  let sum = 0;
  for (let c = 0; c < W; c++) {
    if (c === skipCol) continue;
    for (let y = 0; y < H; y++) {
      if ((b.rows[y] >> c & 1) === 1) break;
      const l = c === 0 ? 1 : (b.rows[y] >> (c - 1) & 1);
      const r = c === W - 1 ? 1 : (b.rows[y] >> (c + 1) & 1);
      if (l === 1 && r === 1) {
        let yy = y;
        while (yy < H && (b.rows[yy] >> c & 1) === 0) { sum++; yy++; }
        break;
      }
    }
  }
  return sum;
}
// rows complete except the well column, with the well not stacked too high to feed
function tetrisReady(b) {
  const wellMask = 1 << WELL;
  const full = (1 << W) - 1;
  let n = 0;
  for (let y = 0; y < H; y++) if ((b.rows[y] | wellMask) === full && (b.rows[y] & wellMask) === 0) n++;
  return (colHeight(b, WELL) <= 16) ? n : 0;
}
// any filled cell in the well column above its own floor ⇒ the well is capped
function coveredWell(b) {
  let depth = 0;
  for (let y = 0; y < H; y++) {
    if ((b.rows[y] >> WELL & 1) === 1) { // something in the well
      // count empty cells beneath it — those are sealed off
      for (let yy = y + 1; yy < H; yy++) if ((b.rows[yy] >> WELL & 1) === 0) depth++;
      break;
    }
  }
  return depth;
}
function aggHeight(b) { let s = 0; for (let c = 0; c < W; c++) s += colHeight(b, c); return s; }

// ── the widened evaluator ────────────────────────────────────────────────────────────────────────
// baseline Dellacherie terms come from the engine itself (identical maths, so the comparison is honest)
function scoreFor(b, piece, rot, x, w) {
  const y = b.dropY(piece, rot, x);
  if (y < 0) return -1e18;
  const saved = b.saveRows();
  const cleared = b.simPlace(piece, rot, x, y);
  const landing = b.simLanding, eroded = b.simEroded;
  const rowT = b.rowTransitions(), colT = b.colTransitions(), holes = b.holes();

  const wells = wellSumExcluding(b, w.splitWell ? WELL : -1);
  const ready = tetrisReady(b);
  const covered = coveredWell(b);
  const col9h = colHeight(b, WELL - 1);
  const wellH = colHeight(b, WELL);
  const avgH = aggHeight(b) / W;
  b.restoreRows(saved);

  // burn = clearing rows that are NOT a tetris; the term that makes a single COST something
  const burn = (cleared > 0 && cleared < 4) ? cleared : 0;
  const isTetris = cleared === 4 ? 1 : 0;

  return (0.0
    - landing + eroded - rowT - colT - 4 * holes - wells      // Dellacherie baseline
    + w.tetris * isTetris
    + w.ready * ready
    + w.covered * covered
    + w.burn * burn
    + w.col9 * Math.max(0, col9h - w.maxSafeCol9)
    + w.wellDepth * Math.min(wellH === 0 ? 4 : 0, 4)          // mild pull toward keeping it open
    + w.avgH * Math.max(0, avgH - 8)
  );
}

function bestAction(b, w) {
  let best = -1e18, bestA = -1;
  for (let a = 0; a < ACTIONS; a++) {
    if (!b.placementLegal(a)) continue;
    const s = scoreFor(b, b.current, b.actionRot(a), b.actionCol(a), w);
    if (s > best) { best = s; bestA = a; }
  }
  return bestA;
}

// ── protocol A: uniform pieces, no garbage, 500-piece cap, fixed seeds ───────────────────────────
function runConfig(w, episodes, seedBase) {
  let score = 0, lines = 0, tetrises = 0, pieces = 0, topouts = 0;
  for (let e = 0; e < episodes; e++) {
    const b = new PgTetris();
    b.reset(seedBase + e, false, 0);
    let n = 0;
    while (!b.gameOver && n < PIECE_CAP) {
      const a = bestAction(b, w);
      if (a < 0) break;
      b.applyPlacement(a);
      n++;
    }
    if (b.gameOver) topouts++;
    score += b.score; lines += b.lines; tetrises += b.tetrises; pieces += n;
  }
  const trt = lines > 0 ? (4 * tetrises / lines) * 100 : 0;
  return {
    score: score / episodes, lines: lines / episodes, tetrises: tetrises / episodes,
    trt, spp: score / Math.max(1, pieces), topoutRate: topouts / episodes,
  };
}

const BASE = { tetris: 0, ready: 0, covered: 0, burn: 0, col9: 0, maxSafeCol9: 8, wellDepth: 0, avgH: 0, splitWell: false };
// NOTE (2026-08-30): the first pass copied StackRabbit's ABSOLUTE weights (ready +6, covered −10,
// burn −12) onto the Dellacherie basis. That is a scale error — StackRabbit prices holes at −50 where
// Dellacherie prices them at −4, so those terms arrive ~12× overweight, drown the safety terms, and the
// agent tops out 100% of the time. Weights below are scaled TO THE DELLACHERIE BASIS (holes = −4).
const S = 4 / 50; // StackRabbit -> Dellacherie scale factor, anchored on the hole coefficient
const SR = (v) => +(v * S).toFixed(2);
const configs = [
  ['baseline (exact Dellacherie)',        { ...BASE }],
  ['well-sign split only',                { ...BASE, splitWell: true }],
  ['split + ready .5 + covered -.8',      { ...BASE, splitWell: true, ready: 0.5, covered: -0.8 }],
  ['split + StackRabbit scaled (x0.08)',  { ...BASE, splitWell: true, ready: SR(6), covered: SR(-10), burn: SR(-12), col9: SR(-3), tetris: SR(50) }],
  ['+ scaled, burn -0.5',                 { ...BASE, splitWell: true, ready: SR(6), covered: SR(-10), burn: -0.5, col9: SR(-3), tetris: SR(50) }],
  ['+ scaled, burn -1.5',                 { ...BASE, splitWell: true, ready: SR(6), covered: SR(-10), burn: -1.5, col9: SR(-3), tetris: SR(50) }],
  ['+ scaled, burn -3',                   { ...BASE, splitWell: true, ready: SR(6), covered: SR(-10), burn: -3,   col9: SR(-3), tetris: SR(50) }],
  ['+ scaled, burn -1.5, ready 2',        { ...BASE, splitWell: true, ready: 2, covered: SR(-10), burn: -1.5, col9: SR(-3), tetris: SR(50) }],
  ['+ scaled, burn -1.5, ready 2, tet 8', { ...BASE, splitWell: true, ready: 2, covered: SR(-10), burn: -1.5, col9: SR(-3), tetris: 8 }],
  ['+ scaled, burn -3, ready 2, tet 8',   { ...BASE, splitWell: true, ready: 2, covered: SR(-10), burn: -3,   col9: SR(-3), tetris: 8 }],
];

console.log(`S0 — widened evaluator, protocol A (uniform, no garbage, ${PIECE_CAP}-piece cap), ${EPS} eps/config, seeds 5000+e\n`);
console.log('config                                  score      lines  tetris/ep    TRT%   score/piece  topout%');
console.log('-'.repeat(104));
const rows = [];
for (const [name, w] of configs) {
  const r = runConfig(w, EPS, 5000);
  rows.push([name, r]);
  console.log(
    `${name.padEnd(38)} ${r.score.toFixed(0).padStart(8)} ${r.lines.toFixed(1).padStart(10)} ` +
    `${r.tetrises.toFixed(2).padStart(10)} ${r.trt.toFixed(1).padStart(7)} ${r.spp.toFixed(1).padStart(13)} ` +
    `${(r.topoutRate * 100).toFixed(0).padStart(8)}`
  );
}

const base = rows[0][1];
const best = rows.reduce((a, b) => (b[1].tetrises > a[1].tetrises ? b : a));
console.log('\n' + '-'.repeat(104));
console.log(`baseline: ${base.tetrises.toFixed(2)} tetrises/ep, TRT ${base.trt.toFixed(1)}%, score ${base.score.toFixed(0)}`);
console.log(`best    : ${best[0]} -> ${best[1].tetrises.toFixed(2)} tetrises/ep, TRT ${best[1].trt.toFixed(1)}%, score ${best[1].score.toFixed(0)}`);
const go = best[1].tetrises >= 2.0 && best[1].score >= 85000;
const nogo = rows.every(([, r]) => r.tetrises < 2.0 || r.score < 70000);
console.log(`\nGATE: ${go ? 'GO' : nogo ? 'NO-GO' : 'PARTIAL'} — need >=2.0 tetrises/ep AND score >=85,000`);

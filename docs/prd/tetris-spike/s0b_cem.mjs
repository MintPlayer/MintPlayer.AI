// M57 spike S0b — CEM-tune the WIDENED evaluator against NES score on protocol A.
//
// This is TETRIS_PRD.md's un-run M54.7 stretch, on a basis that can actually EXPRESS tetris play.
// Important caveat from the literature survey: CMA-ES on the NARROW Dellacherie basis provably
// converges back to Dellacherie's hand weights — so widening the basis first is what makes tuning
// worth anything. S0 established the widened basis is a strict improvement; this finds its optimum.
//
// Fitness = mean NES score, protocol A (uniform, no garbage, 500-piece cap), fixed seeds.
// Gate (PRD §6 S0b): GO if a tuned vector beats the hand weights, CI-separated.
//
// Run:  node docs/prd/tetris-spike/s0b_cem.mjs [pop] [iters] [epsPerEval]

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;

const W = 10, H = 20, ACTIONS = 40, WELL = 9, PIECE_CAP = 500;
const POP = Number(process.argv[2] ?? 20);
const ITERS = Number(process.argv[3] ?? 6);
const FIT_EPS = Number(process.argv[4] ?? 4);
const ELITE = Math.max(3, Math.round(POP * 0.25));

function colHeight(b, c) { for (let y = 0; y < H; y++) if ((b.rows[y] >> c & 1) === 1) return H - y; return 0; }
function wellSumExcluding(b, skip) {
  let sum = 0;
  for (let c = 0; c < W; c++) {
    if (c === skip) continue;
    for (let y = 0; y < H; y++) {
      if ((b.rows[y] >> c & 1) === 1) break;
      const l = c === 0 ? 1 : (b.rows[y] >> (c - 1) & 1);
      const r = c === W - 1 ? 1 : (b.rows[y] >> (c + 1) & 1);
      if (l === 1 && r === 1) { let yy = y; while (yy < H && (b.rows[yy] >> c & 1) === 0) { sum++; yy++; } break; }
    }
  }
  return sum;
}
function tetrisReady(b) {
  const wellMask = 1 << WELL, full = (1 << W) - 1;
  let n = 0;
  for (let y = 0; y < H; y++) if ((b.rows[y] | wellMask) === full && (b.rows[y] & wellMask) === 0) n++;
  return colHeight(b, WELL) <= 16 ? n : 0;
}
function coveredWell(b) {
  for (let y = 0; y < H; y++) if ((b.rows[y] >> WELL & 1) === 1) {
    let d = 0; for (let yy = y + 1; yy < H; yy++) if ((b.rows[yy] >> WELL & 1) === 0) d++; return d;
  }
  return 0;
}
function aggHeight(b) { let s = 0; for (let c = 0; c < W; c++) s += colHeight(b, c); return s; }

// weight vector: [landing, eroded, rowT, colT, holes, wells, ready, covered, burn, col9, tetris, avgH]
const NAMES = ['landing', 'eroded', 'rowT', 'colT', 'holes', 'wells', 'ready', 'covered', 'burn', 'col9', 'tetris', 'avgH'];
const HAND = [-1, 1, -1, -1, -4, -1, 2, -0.8, -3, -0.24, 8, 0];   // S0's best config
const MAX_SAFE_COL9 = 8, SCARE_H = 8;

function scoreFor(b, piece, rot, x, w) {
  const y = b.dropY(piece, rot, x);
  if (y < 0) return -1e18;
  const saved = b.saveRows();
  const cleared = b.simPlace(piece, rot, x, y);
  const landing = b.simLanding, eroded = b.simEroded;
  const rowT = b.rowTransitions(), colT = b.colTransitions(), holes = b.holes();
  const wells = wellSumExcluding(b, WELL);
  const ready = tetrisReady(b), covered = coveredWell(b);
  const col9h = colHeight(b, WELL - 1), avgH = aggHeight(b) / W;
  b.restoreRows(saved);
  const burn = (cleared > 0 && cleared < 4) ? cleared : 0;
  const isTet = cleared === 4 ? 1 : 0;
  return w[0] * landing + w[1] * eroded + w[2] * rowT + w[3] * colT + w[4] * holes + w[5] * wells
    + w[6] * ready + w[7] * covered + w[8] * burn + w[9] * Math.max(0, col9h - MAX_SAFE_COL9)
    + w[10] * isTet + w[11] * Math.max(0, avgH - SCARE_H);
}

function play(w, seed) {
  const b = new PgTetris();
  b.reset(seed, false, 0);
  let n = 0;
  while (!b.gameOver && n < PIECE_CAP) {
    let best = -1e18, bestA = -1;
    for (let a = 0; a < ACTIONS; a++) {
      if (!b.placementLegal(a)) continue;
      const s = scoreFor(b, b.current, b.actionRot(a), b.actionCol(a), w);
      if (s > best) { best = s; bestA = a; }
    }
    if (bestA < 0) break;
    b.applyPlacement(bestA);
    n++;
  }
  return { score: b.score, lines: b.lines, tetrises: b.tetrises, pieces: n, topout: b.gameOver };
}
function fitness(w, eps, seedBase) {
  let s = 0;
  for (let e = 0; e < eps; e++) s += play(w, seedBase + e).score;
  return s / eps;
}
function evaluate(w, eps, seedBase) {
  const rs = [];
  for (let e = 0; e < eps; e++) rs.push(play(w, seedBase + e));
  const n = rs.length;
  const mean = (f) => rs.reduce((a, r) => a + f(r), 0) / n;
  const score = mean(r => r.score);
  const sd = Math.sqrt(rs.reduce((a, r) => a + (r.score - score) ** 2, 0) / Math.max(1, n - 1));
  const lines = mean(r => r.lines), tet = mean(r => r.tetrises);
  return { score, ci: 1.96 * sd / Math.sqrt(n), lines, tetrises: tet,
           trt: lines > 0 ? 400 * tet / lines : 0, spp: score / Math.max(1, mean(r => r.pieces)),
           topout: mean(r => (r.topout ? 1 : 0)) };
}

// deterministic RNG so the spike is reproducible
let _s = 12345;
function rnd() { _s = (Math.imul(_s, 1103515245) + 12345) & 0x7fffffff; return _s / 0x7fffffff; }
function gauss() { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()); }

let mu = HAND.slice();
let sigma = HAND.map(v => Math.max(1.0, Math.abs(v) * 0.6));

console.log(`S0b — CEM over the widened basis. pop=${POP} elite=${ELITE} iters=${ITERS} eps/eval=${FIT_EPS}`);
console.log(`fitness = mean NES score, protocol A, seeds 7000+ (tuning seeds, DISJOINT from eval)\n`);
for (let it = 0; it < ITERS; it++) {
  const cands = [];
  for (let p = 0; p < POP; p++) {
    const w = mu.map((m, i) => m + sigma[i] * gauss());
    // sign priors: these terms are adverse/beneficial by construction; keep CEM out of nonsense regions
    w[4] = -Math.abs(w[4]); w[8] = -Math.abs(w[8]); w[7] = -Math.abs(w[7]); w[10] = Math.abs(w[10]);
    cands.push({ w, f: fitness(w, FIT_EPS, 7000) });
  }
  cands.sort((a, b) => b.f - a.f);
  const elite = cands.slice(0, ELITE);
  mu = mu.map((_, i) => elite.reduce((a, c) => a + c.w[i], 0) / ELITE);
  sigma = mu.map((m, i) => Math.sqrt(elite.reduce((a, c) => a + (c.w[i] - m) ** 2, 0) / ELITE) + 0.05);
  console.log(`iter ${it + 1}/${ITERS}  best=${cands[0].f.toFixed(0)}  eliteMean=${(elite.reduce((a, c) => a + c.f, 0) / ELITE).toFixed(0)}`);
}

console.log('\ntuned weights:');
console.log(NAMES.map((n, i) => `  ${n.padEnd(8)} ${mu[i].toFixed(3)}`).join('\n'));

console.log('\nHELD-OUT evaluation, seeds 9000+ (never used in tuning), 30 episodes:');
const hand = evaluate(HAND, 30, 9000);
const tuned = evaluate(mu, 30, 9000);
const dell = evaluate([-1, 1, -1, -1, -4, -1, 0, 0, 0, 0, 0, 0], 30, 9000);
const row = (n, r) => `${n.padEnd(26)} ${r.score.toFixed(0).padStart(8)} ±${r.ci.toFixed(0).padStart(6)} ${r.lines.toFixed(1).padStart(8)} ${r.tetrises.toFixed(2).padStart(9)} ${r.trt.toFixed(1).padStart(7)} ${r.spp.toFixed(1).padStart(11)} ${(r.topout * 100).toFixed(0).padStart(8)}`;
console.log('policy                        score      ±CI    lines tetris/ep    TRT%  score/piece  topout%');
console.log('-'.repeat(96));
console.log(row('Dellacherie (baseline)', dell));
console.log(row('S0 hand-widened', hand));
console.log(row('CEM-tuned', tuned));
const sep = tuned.score - tuned.ci > hand.score + hand.ci;
console.log(`\nGATE: CEM ${sep ? 'BEATS' : 'does NOT beat'} the hand weights CI-separated on held-out seeds.`);

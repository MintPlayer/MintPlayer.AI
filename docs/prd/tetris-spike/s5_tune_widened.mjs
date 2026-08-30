// M57.1 tuning — find weights that raise TETRISES and score on protocol A WITHOUT regressing protocol-B
// survival (gate G1). The first hand-set vector scored +40% on A but -25% on B; S0b showed that CEM on
// raw score alone just buys variance (30% top-out). So the fitness is deliberately CONSTRAINED:
//
//     fit = (Ascore/100k + 0.6*Atet/4) * min(1, Bpieces / B_TARGET)
//
// The multiplicative min() term means survival below the M54 baseline scales the whole fitness down
// proportionally and cannot be bought back with score — while exceeding the baseline earns nothing, so
// CEM spends its effort on tetrises instead of on hoarding survival.
//
// Mirrors the ENGINE formula in tetris_solver.pg exactly, including the LINEOUT and DIG mode switches.
// Run:  node docs/prd/tetris-spike/s5_tune_widened.mjs [pop] [iters]

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;

const W = 10, H = 20, WELL = 9;
const POP = Number(process.argv[2] ?? 16);
const ITERS = Number(process.argv[3] ?? 8);
const ELITE = Math.max(3, Math.round(POP * 0.3));
const FIT_A = 3, FIT_B = 3, CAP_A = 500, CAP_B = 600;
const B_TARGET = 364;              // the M54 Dellacherie protocol-B baseline

// weight vector (mirrors the .pg consts)
// 0 holes 1 wells 2 ready 3 burn 4 burnDig 5 holeDig 6 covered 7 col9 8 tetris 9 inaccessible
const NAMES = ['holes', 'wells', 'ready', 'burn', 'burnDig', 'holeDig', 'covered', 'col9', 'tetris', 'inacc'];
const CURRENT = [-4, -1, 2.0, -3.0, -0.25, -0.6, -0.8, -0.24, 8.0, -8.0];
const MAX_SAFE_COL9 = 8;

function colHeight(b, c) { for (let y = 0; y < H; y++) if ((b.rows[y] >> c & 1) === 1) return H - y; return 0; }
function wellSumExceptWell(b) {
  let sum = 0;
  for (let c = 0; c < W; c++) {
    if (c === WELL) continue;
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
  if (colHeight(b, WELL) > 16) return 0;
  const wm = 1 << WELL, full = (1 << W) - 1;
  let n = 0;
  for (let y = 0; y < H; y++) if ((b.rows[y] | wm) === full && (b.rows[y] & wm) === 0) n++;
  return n;
}
function coveredWell(b) {
  for (let y = 0; y < H; y++) if ((b.rows[y] >> WELL & 1) === 1) {
    let d = 0; for (let yy = y + 1; yy < H; yy++) if ((b.rows[yy] >> WELL & 1) === 0) d++; return d;
  }
  return 0;
}
function maxTapHeight(b, taps, tap) { return H - 3 - Math.floor(taps * tap / b.gravityFrames(b.level)); }

function scoreFor(b, piece, rot, x, y, w, tap) {
  const saved = b.saveRows();
  const cleared = b.simPlace(piece, rot, x, y);
  const holes = b.holes();
  let s = 0.0 - b.simLanding + b.simEroded - b.rowTransitions() - b.colTransitions()
    + w[0] * holes + w[1] * wellSumExceptWell(b);
  const m5 = maxTapHeight(b, 5, tap), m4 = maxTapHeight(b, 4, tap);
  const lineout = m5 < 4, dig = holes > 0;
  const burnLines = (cleared > 0 && cleared < 4) ? cleared : 0;
  if (!lineout) {
    if (!dig) s += w[2] * tetrisReady(b);
    s += (dig ? w[4] : w[3]) * burnLines;
  }
  if (dig) s += w[5] * holes;
  s += w[6] * coveredWell(b);
  s += w[8] * (cleared === 4 ? 1 : 0);
  const c9 = colHeight(b, WELL - 1) - MAX_SAFE_COL9;
  if (c9 > 0) s += w[7] * c9;
  const oL = colHeight(b, 0) - m5; if (oL > 0) s += w[9] * oL;
  const oR = colHeight(b, W - 1) - m4; if (oR > 0) s += w[9] * oR;
  b.restoreRows(saved);
  return s;
}

function play(w, seed, garbage, cap, tap = 6) {
  const b = new PgTetris();
  b.reset(seed, false, garbage);
  let n = 0;
  while (!b.gameOver && n < cap) {
    const piece = b.current;
    let best = -1e18, bp = null;
    for (let r = 0; r < b.rotCount[piece]; r++) {
      for (let x = 0; x + b.rotW[piece * 4 + r] <= W; x++) {
        const y = b.dropY(piece, r, x);
        if (y < 0) continue;
        const v = scoreFor(b, piece, r, x, y, w, tap);
        if (v > best) { best = v; bp = [r, x, y]; }
      }
    }
    if (!bp) break;
    const cleared = b.simPlace(piece, bp[0], bp[1], bp[2]);
    b.afterLock(cleared);
    n++;
    if (!b.gameOver && !b.hasLegalPlacement()) break;
  }
  return { pieces: n, score: b.score, lines: b.lines, tetrises: b.tetrises };
}

function fitness(w, seedBase) {
  let aS = 0, aT = 0, bP = 0;
  for (let e = 0; e < FIT_A; e++) { const r = play(w, seedBase + e, 0, CAP_A); aS += r.score; aT += r.tetrises; }
  for (let e = 0; e < FIT_B; e++) { bP += play(w, seedBase + 500 + e, 10, CAP_B).pieces; }
  aS /= FIT_A; aT /= FIT_A; bP /= FIT_B;
  return (aS / 100000 + 0.6 * aT / 4) * Math.min(1, bP / B_TARGET);
}
function evaluate(w, eps, seedBase) {
  const A = [...Array(eps)].map((_, e) => play(w, seedBase + e, 0, CAP_A));
  const B = [...Array(eps)].map((_, e) => play(w, seedBase + 500 + e, 10, 1500));
  const m = (rs, f) => rs.reduce((a, r) => a + f(r), 0) / rs.length;
  const aS = m(A, r => r.score), aT = m(A, r => r.tetrises), aL = m(A, r => r.lines);
  const bP = m(B, r => r.pieces);
  const sdB = Math.sqrt(B.reduce((a, r) => a + (r.pieces - bP) ** 2, 0) / Math.max(1, B.length - 1));
  return { aScore: aS, aTet: aT, aLines: aL, trt: aL > 0 ? 400 * aT / aL : 0,
           bPieces: bP, bCi: 1.96 * sdB / Math.sqrt(B.length) };
}

let _s = 987654321;
const rnd = () => { _s = (Math.imul(_s, 1103515245) + 12345) & 0x7fffffff; return _s / 0x7fffffff; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

let mu = CURRENT.slice();
let sigma = CURRENT.map(v => Math.max(0.4, Math.abs(v) * 0.5));
console.log(`S5 — constrained CEM. pop=${POP} elite=${ELITE} iters=${ITERS}; fitness couples A-score+tetrises to B-survival (target ${B_TARGET}).`);
console.log(`tuning seeds 7000+, held-out evaluation on 9000+\n`);
for (let it = 0; it < ITERS; it++) {
  const cands = [];
  for (let p = 0; p < POP; p++) {
    const w = mu.map((m, i) => m + sigma[i] * gauss());
    // sign priors — these terms are adverse/beneficial by construction
    w[0] = -Math.abs(w[0]); w[1] = -Math.abs(w[1]); w[3] = -Math.abs(w[3]); w[4] = -Math.abs(w[4]);
    w[5] = -Math.abs(w[5]); w[6] = -Math.abs(w[6]); w[7] = -Math.abs(w[7]); w[9] = -Math.abs(w[9]);
    w[2] = Math.abs(w[2]); w[8] = Math.abs(w[8]);
    cands.push({ w, f: fitness(w, 7000) });
  }
  cands.sort((a, b) => b.f - a.f);
  const el = cands.slice(0, ELITE);
  mu = mu.map((_, i) => el.reduce((a, c) => a + c.w[i], 0) / ELITE);
  sigma = mu.map((m, i) => Math.sqrt(el.reduce((a, c) => a + (c.w[i] - m) ** 2, 0) / ELITE) + 0.03);
  console.log(`iter ${it + 1}/${ITERS} best=${cands[0].f.toFixed(3)} eliteMean=${(el.reduce((a, c) => a + c.f, 0) / ELITE).toFixed(3)}`);
}
console.log('\ntuned weights:');
console.log(NAMES.map((n, i) => `  ${n.padEnd(9)} ${mu[i].toFixed(3)}`).join('\n'));
console.log('\nHELD-OUT (seeds 9000+, 20 eps):');
const cur = evaluate(CURRENT, 20, 9000), tun = evaluate(mu, 20, 9000);
const row = (n, r) => `${n.padEnd(16)} A ${r.aScore.toFixed(0).padStart(8)}  lines ${r.aLines.toFixed(1).padStart(6)}  tet ${r.aTet.toFixed(2).padStart(6)}  TRT ${r.trt.toFixed(1).padStart(5)}%   B ${r.bPieces.toFixed(1).padStart(7)} ±${r.bCi.toFixed(1)}`;
console.log(row('current (hand)', cur));
console.log(row('CEM-tuned', tun));
console.log(`\nM54 baselines: A 94,636 / 197.6 lines / 0.26 tet   B 363.8 ± 40.3`);
console.log(`G1 no-regression on B: current ${cur.bPieces >= 324 ? 'PASS' : 'FAIL'}, tuned ${tun.bPieces >= 324 ? 'PASS' : 'FAIL'} (need >= ~324)`);

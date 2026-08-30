// M57 spike S3 — "does tap speed determine whether the agent can feed a well at the SIDE?"
//
// Owner hypothesis (2026-08-30): the model may prefer a flat field because it cannot get pieces over
// to the side — i.e. flatness is a symptom of slow input, not an evaluator preference.
//
// S0-S2 could not answer this: they all started at LEVEL 0 (48 frames/row), where input speed binds on
// nothing. Gate G7 pre-registered exactly this failure mode. This spike pins gravity instead.
//
// Part 1 — LATERAL REACH: the max4TapHeight / max5TapHeight analysis, measured on this engine rather
//   than taken from the literature. On a flat stack of height h, which columns can the piece still
//   reach before it locks? Column 9 is 4 taps from spawn, column 0 is 5 (spawn x=5).
// Part 2 — STRENGTH: the S0-widened evaluator playing a TAP-CONSTRAINED action set at fixed gravity.
//   DAS vs hypertapping vs rolling, against an unconstrained (all-40) reference.
//
// Run:  node docs/prd/tetris-spike/s3_lateral_reach.mjs [episodes]

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;

const W = 10, H = 20, WELL = 9;
const EPS = Number(process.argv[2] ?? 10);
const CAP = 400;
const RATES = [['DAS 10Hz', 6], ['hyper 12Hz', 5], ['hyper 15Hz', 4], ['rolling 20Hz', 3], ['30Hz cap', 2]];
const LEVELS = [['L9  (6f/row)', 6], ['L18 (3f/row)', 3], ['L19 (2f/row)', 2], ['L29 (1f/row)', 1]];

function reachableLocks(b, piece, g, p) {
  const sr = b.spawnRot[piece], x0 = 5 + b.rotOffX[piece * 4 + sr], y0 = 0;
  if (!b.fitsAt(piece, sr, x0, y0)) return null;
  const rc = b.rotCount[piece], FMAX = (H + 3) * g + 4 * p + 8;
  const key = (x, y, r, f) => (((f * 4 + r) * (H + 3) + (y + 2)) * (W + 4)) + (x + 2);
  const seen = new Set([key(x0, y0, sr, 0)]);
  const locks = new Set();
  let q = [[x0, y0, sr, 0]];
  while (q.length) {
    const nx = [];
    for (const [x, y, r, f] of q) {
      if (f >= FMAX) continue;
      const inp = (f % p) === 0;
      const cands = [];
      for (const dr of (inp ? [0, 1, rc - 1] : [0])) {
        const nr = (r + dr) % rc;
        let px = x, py = y;
        if (dr !== 0) {
          px = x - b.rotOffX[piece * 4 + r] + b.rotOffX[piece * 4 + nr];
          py = y - b.rotOffY[piece * 4 + r] + b.rotOffY[piece * 4 + nr];
          if (!b.fitsAt(piece, nr, px, py)) continue;
        }
        for (const dx of (inp ? [0, -1, 1] : [0])) {
          const fx = px + dx;
          if (!b.fitsAt(piece, nr, fx, py)) continue;
          cands.push([fx, py, nr]);
        }
      }
      for (const [cx, cy, cr] of cands) {
        if (((f + 1) % g) === 0) {
          if (b.fitsAt(piece, cr, cx, cy + 1)) {
            const k = key(cx, cy + 1, cr, f + 1);
            if (!seen.has(k)) { seen.add(k); nx.push([cx, cy + 1, cr, f + 1]); }
          } else locks.add(cr * 400 + (cy + 2) * 20 + cx);
        } else {
          const k = key(cx, cy, cr, f + 1);
          if (!seen.has(k)) { seen.add(k); nx.push([cx, cy, cr, f + 1]); }
        }
      }
    }
    q = nx;
  }
  return locks;
}

// ── Part 1: lateral reach on a FLAT stack of height h ────────────────────────────────────────────
// A "4-tap" placement puts a piece against the right wall; a "5-tap" against the left. We report the
// greatest stack height at which each is still reachable — StackRabbit's max4TapHeight / max5TapHeight.
function flatBoard(h) {
  const b = new PgTetris();
  b.reset(1, false, 0);
  for (let y = 0; y < H; y++) b.rows[y] = (y >= H - h) ? (1 << W) - 1 : 0;
  return b;
}
function reachSpan(g, p, h) {
  const b = flatBoard(h);
  let minCol = 99, maxCol = -1;
  for (let piece = 0; piece < 7; piece++) {
    const L = reachableLocks(b, piece, g, p);
    if (!L) continue;
    for (const l of L) {
      const x = l % 20, r = Math.floor(l / 400);
      minCol = Math.min(minCol, x);
      maxCol = Math.max(maxCol, x + b.rotW[piece * 4 + r] - 1);
    }
  }
  return { minCol, maxCol };
}

console.log('S3 PART 1 — lateral reach on a flat stack (spawn x=5; col 9 is 4 taps, col 0 is 5 taps)');
console.log('Greatest stack height at which the piece can still reach each wall:\n');
console.log('level          rate            max height reaching col 0   max height reaching col 9');
console.log('-'.repeat(88));
const reachTable = {};
for (const [ln, g] of LEVELS) {
  for (const [rn, p] of RATES) {
    let max0 = -1, max9 = -1;
    for (let h = 0; h <= 16; h++) {
      const { minCol, maxCol } = reachSpan(g, p, h);
      if (minCol === 0) max0 = h;
      if (maxCol === W - 1) max9 = h;
    }
    reachTable[`${ln}|${rn}`] = { max0, max9 };
    console.log(`${ln.padEnd(14)} ${rn.padEnd(14)} ${String(max0).padStart(20)} ${String(max9).padStart(28)}`);
  }
}

// ── Part 2: strength under a tap budget, at PINNED gravity ───────────────────────────────────────
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
  const wm = 1 << WELL, full = (1 << W) - 1;
  let n = 0;
  for (let y = 0; y < H; y++) if ((b.rows[y] | wm) === full && (b.rows[y] & wm) === 0) n++;
  return colHeight(b, WELL) <= 16 ? n : 0;
}
function coveredWell(b) {
  for (let y = 0; y < H; y++) if ((b.rows[y] >> WELL & 1) === 1) {
    let d = 0; for (let yy = y + 1; yy < H; yy++) if ((b.rows[yy] >> WELL & 1) === 0) d++; return d;
  }
  return 0;
}
// S0's best hand-widened vector
function widenedAt(b, piece, rot, x, y) {
  const saved = b.saveRows();
  const cleared = b.simPlace(piece, rot, x, y);
  const s = 0.0 - b.simLanding + b.simEroded - b.rowTransitions() - b.colTransitions()
    - 4 * b.holes() - wellSumExcluding(b, WELL)
    + 2 * tetrisReady(b) - 0.8 * coveredWell(b)
    - 3 * ((cleared > 0 && cleared < 4) ? cleared : 0)
    - 0.24 * Math.max(0, colHeight(b, WELL - 1) - 8)
    + 8 * (cleared === 4 ? 1 : 0);
  b.restoreRows(saved);
  return s;
}
function playConstrained(seed, g, p) {
  const b = new PgTetris();
  b.reset(seed, false, 0);
  let n = 0, wellFeeds = 0;
  while (!b.gameOver && n < CAP) {
    const piece = b.current;
    const locks = p === null ? null : reachableLocks(b, piece, g, p);
    let best = -1e18, bp = null;
    if (p === null) {                                   // unconstrained reference: all 40 hard drops
      for (let a = 0; a < 40; a++) {
        if (!b.placementLegal(a)) continue;
        const r = b.actionRot(a), x = b.actionCol(a), y = b.dropY(piece, r, x);
        const v = widenedAt(b, piece, r, x, y);
        if (v > best) { best = v; bp = [r, x, y]; }
      }
    } else {
      if (!locks || locks.size === 0) break;
      const ordered = [...locks].sort((a, c) => (Math.floor(a / 400) - Math.floor(c / 400)) || ((a % 20) - (c % 20)));
      for (const L of ordered) {
        const r = Math.floor(L / 400), y = Math.floor((L % 400) / 20) - 2, x = L % 20;
        if (y < 0) continue;
        const v = widenedAt(b, piece, r, x, y);
        if (v > best) { best = v; bp = [r, x, y]; }
      }
    }
    if (!bp) break;
    const [r, x, y] = bp;
    if (x + b.rotW[piece * 4 + r] - 1 >= WELL) wellFeeds++;
    const cleared = b.simPlace(piece, r, x, y);
    b.afterLock(cleared);
    n++;
    if (!b.gameOver && !b.hasLegalPlacement()) break;
  }
  return { pieces: n, score: b.score, lines: b.lines, tetrises: b.tetrises, wellFeeds };
}
function agg(rows) {
  const n = rows.length, m = f => rows.reduce((a, r) => a + f(r), 0) / n;
  const s = m(r => r.score);
  const sd = Math.sqrt(rows.reduce((a, r) => a + (r.score - s) ** 2, 0) / Math.max(1, n - 1));
  const li = m(r => r.lines), te = m(r => r.tetrises);
  return { score: s, ci: 1.96 * sd / Math.sqrt(n), pieces: m(r => r.pieces), lines: li, tetrises: te,
           trt: li > 0 ? 400 * te / li : 0, wellFeeds: m(r => r.wellFeeds) };
}

console.log('\n\nS3 PART 2 — widened evaluator under a TAP BUDGET, gravity PINNED (this is gate G7)');
console.log(`${EPS} episodes, ${CAP}-piece cap, seeds 5000+. "reach col9%" = placements touching the well column.\n`);
for (const [ln, g] of [['L18 (3f/row)', 3], ['L19 (2f/row)', 2], ['L29 (1f/row)', 1]]) {
  console.log(`### ${ln}`);
  console.log('rate              score       ±CI   pieces   lines  tetris   TRT%   col9-touch/ep');
  console.log('-'.repeat(86));
  const ref = agg([...Array(EPS)].map((_, e) => playConstrained(5000 + e, g, null)));
  console.log(`unconstrained  ${ref.score.toFixed(0).padStart(9)} ${ref.ci.toFixed(0).padStart(9)} ${ref.pieces.toFixed(0).padStart(8)} ${ref.lines.toFixed(1).padStart(7)} ${ref.tetrises.toFixed(2).padStart(7)} ${ref.trt.toFixed(1).padStart(6)} ${ref.wellFeeds.toFixed(1).padStart(15)}`);
  for (const [rn, p] of RATES) {
    const r = agg([...Array(EPS)].map((_, e) => playConstrained(5000 + e, g, p)));
    console.log(`${rn.padEnd(14)} ${r.score.toFixed(0).padStart(9)} ${r.ci.toFixed(0).padStart(9)} ${r.pieces.toFixed(0).padStart(8)} ${r.lines.toFixed(1).padStart(7)} ${r.tetrises.toFixed(2).padStart(7)} ${r.trt.toFixed(1).padStart(6)} ${r.wellFeeds.toFixed(1).padStart(15)}`);
  }
  console.log('');
}

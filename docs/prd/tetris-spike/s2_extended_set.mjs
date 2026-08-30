// M57 spike S2 — "are the extra (movement-aware) placements worth anything AT ALL?"
//
// The decisive economic test, and the cheapest way to cancel a 6-hour retrain. Runs the SAME
// Dellacherie evaluator twice: once choosing over the 40 hard-drop placements (today's action space),
// once over the frame-simulation reachable set (tucks included). No training, no net.
//
// Reasoning behind the gate (PRD §6 S2): a PERFECT evaluator that cannot exploit the extra placements
// is decisive evidence that a distilled gamma=0 net will not either.
//
// Gate: GO if protocol-B survival improves >= +15% CI-separated. NO-GO if flat or worse.
//
// Run:  node docs/prd/tetris-spike/s2_extended_set.mjs [episodes] [pieceCap]

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;

const W = 10, H = 20;
const EPS = Number(process.argv[2] ?? 12);
const CAP = Number(process.argv[3] ?? 800);
const TAP = 6;                       // DAS 10 Hz — the most restrictive dial setting

// exact frame simulation (same model as S1): shift -> rotate -> gravity, NRS pivot, no kicks
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

// Dellacherie value of landing `piece` in state (rot,x,y) — identical maths to the engine's
// dellaScoreFor, but with an EXPLICIT y so tucked landings can be scored.
function dellaAt(b, piece, rot, x, y) {
  const saved = b.saveRows();
  b.simPlace(piece, rot, x, y);
  const s = 0.0 - b.simLanding + b.simEroded - b.rowTransitions() - b.colTransitions()
    - 4 * b.holes() - b.wellSum();
  b.restoreRows(saved);
  return s;
}

function playHardDrop(seed, garbage) {
  const b = new PgTetris();
  b.reset(seed, false, garbage);
  let n = 0;
  while (!b.gameOver && n < CAP) {
    const a = b.dellacherieAction();
    if (a < 0 || !b.placementLegal(a)) break;
    b.applyPlacement(a);
    n++;
  }
  return { pieces: n, score: b.score, lines: b.lines, tetrises: b.tetrises };
}

function playExtended(seed, garbage, hardOnly = false) {
  const b = new PgTetris();
  b.reset(seed, false, garbage);
  let n = 0, tucksUsed = 0;
  while (!b.gameOver && n < CAP) {
    const g = b.gravityFrames(b.level);
    const piece = b.current;
    const locks = reachableLocks(b, piece, g, TAP);
    if (!locks || locks.size === 0) break;
    let best = -1e18, bp = null, bestIsTuck = false;
    // Match the ENGINE's tie-break exactly (dellacherieAction: rotation-major, then column, first
    // strict improvement wins). Iterating a Set in insertion order silently changes which of several
    // equal-valued placements is taken, and that alone moved survival ~26% in the control run.
    const ordered = [...locks].sort((p1, p2) => {
      const r1 = Math.floor(p1 / 400), x1 = p1 % 20, y1 = Math.floor((p1 % 400) / 20);
      const r2 = Math.floor(p2 / 400), x2 = p2 % 20, y2 = Math.floor((p2 % 400) / 20);
      return (r1 - r2) || (x1 - x2) || (y1 - y2);
    });
    for (const L of ordered) {
      const r = Math.floor(L / 400), y = Math.floor((L % 400) / 20) - 2, x = L % 20;
      if (y < 0) continue;
      if (hardOnly && b.dropY(piece, r, x) !== y) continue;   // CONTROL: drop tucks
      const v = dellaAt(b, piece, r, x, y);
      if (v > best) { best = v; bp = [r, x, y]; bestIsTuck = (b.dropY(piece, r, x) !== y); }
    }
    if (!bp) break;
    const [r, x, y] = bp;
    if (bestIsTuck) tucksUsed++;
    // apply the chosen placement directly (mirrors applyPlacement's lock path)
    const cleared = b.simPlace(piece, r, x, y);
    b.afterLock(cleared);
    n++;
    if (!b.gameOver && !b.hasLegalPlacement()) break;
  }
  return { pieces: n, score: b.score, lines: b.lines, tetrises: b.tetrises, tucksUsed };
}

function agg(rows) {
  const n = rows.length, m = (f) => rows.reduce((a, r) => a + f(r), 0) / n;
  const p = m(r => r.pieces);
  const sd = Math.sqrt(rows.reduce((a, r) => a + (r.pieces - p) ** 2, 0) / Math.max(1, n - 1));
  return { pieces: p, ci: 1.96 * sd / Math.sqrt(n), score: m(r => r.score), lines: m(r => r.lines),
           tetrises: m(r => r.tetrises), tucks: m(r => r.tucksUsed ?? 0) };
}

for (const [label, garbage] of [['protocol B (garbage/10, survival)', 10], ['protocol A (no garbage)', 0]]) {
  console.log(`\n=== ${label} — ${EPS} episodes, ${CAP}-piece cap, DAS 10Hz, seeds 5000+`);
  const hd = [], ex = [];
  const ct = [];
  for (let e = 0; e < EPS; e++) { hd.push(playHardDrop(5000 + e, garbage)); ex.push(playExtended(5000 + e, garbage)); ct.push(playExtended(5000 + e, garbage, true)); }
  const A = agg(hd), B = agg(ex), C = agg(ct);
  console.log('action space          pieces      ±CI     score   lines  tetris  tucksUsed');
  console.log('-'.repeat(76));
  console.log(`hard-drop (40)     ${A.pieces.toFixed(1).padStart(8)} ${A.ci.toFixed(1).padStart(8)} ${A.score.toFixed(0).padStart(9)} ${A.lines.toFixed(1).padStart(7)} ${A.tetrises.toFixed(2).padStart(7)}          -`);
  console.log(`movement-aware     ${B.pieces.toFixed(1).padStart(8)} ${B.ci.toFixed(1).padStart(8)} ${B.score.toFixed(0).padStart(9)} ${B.lines.toFixed(1).padStart(7)} ${B.tetrises.toFixed(2).padStart(7)} ${B.tucks.toFixed(2).padStart(10)}`);
  console.log(`CONTROL: reachable-set-restricted-to-hard-drop  ${C.pieces.toFixed(1).padStart(8)} ${C.ci.toFixed(1).padStart(8)} ${C.score.toFixed(0).padStart(9)} ${C.lines.toFixed(1).padStart(7)}`);
  console.log(`  (should track the hard-drop baseline; a gap here means the enumerator or tie-break differs, not that tucks hurt)`);
  const delta = 100 * (B.pieces - A.pieces) / Math.max(1, A.pieces);
  const sep = (B.pieces - B.ci) > (A.pieces + A.ci);
  console.log(`\ndelta survival: ${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%   CI-separated: ${sep ? 'YES' : 'no'}`);
  if (garbage === 10) console.log(`S2 GATE (protocol B): ${(delta >= 15 && sep) ? 'GO' : 'NO-GO'} — need >= +15% AND CI-separated`);
}

// M57 spike S1 — reachability census. "Does a movement-aware enumerator find placements a hard drop
// cannot reach, and how does that depend on tap speed and gravity?"
//
// Exact frame simulation (StackRabbit's model), NOT an approximation:
//   state = (x, y, rot, frame); y is tracked explicitly because an NRS pivot can move the box
//   vertically. Per frame, in the ROM's order: shift -> rotate -> gravity. Shift and rotate may both
//   occur on the same frame (different buttons). A piece LOCKS when a gravity step fails.
//   Rotation is the engine's NRS in-place pivot (rotOffX/rotOffY), target-cells-only validity, no kicks.
//
// Gate (PRD §6 S1): GO if >=20% of mid-game boards expose >=1 tuck at the DAS (10 Hz) setting AND the
// 99th-percentile tuck depth <= 4 (so N=160 suffices) AND mean |reachable| <= 60.
//
// Run:  node docs/prd/tetris-spike/s1_reachability.mjs [boardsPerConfig]

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris, PgTetRng } = mod;

const W = 10, H = 20, PIECES = 7;
const BOARDS = Number(process.argv[2] ?? 250);

// tap timelines: frames per shift (PRD §4.1)
const RATES = [['DAS 10Hz', 6], ['hyper 12Hz', 5], ['rolling 20Hz', 3]];
// NES gravity: frames per row
const LEVELS = [['L9', 6], ['L18', 3], ['L19-28', 2], ['L29+ (kill)', 1]];

function enumerateReachable(b, piece, g, p) {
  const sr = b.spawnRot[piece];
  const x0 = 5 + b.rotOffX[piece * 4 + sr];
  const y0 = 0;
  if (!b.fitsAt(piece, sr, x0, y0)) return null;      // top-out: piece cannot spawn

  const rc = b.rotCount[piece];
  const FMAX = (H + 3) * g + 4 * p + 8;
  const key = (x, y, r, f) => (((f * 4 + r) * (H + 3) + (y + 2)) * (W + 4)) + (x + 2);
  const seen = new Set();
  const locks = new Set();                            // encoded rot*400 + y*20 + x  (y may be -2..H)
  let queue = [[x0, y0, sr, 0]];
  seen.add(key(x0, y0, sr, 0));

  while (queue.length) {
    const next = [];
    for (const [x, y, r, f] of queue) {
      if (f >= FMAX) continue;
      const inputFrame = (f % p) === 0;
      // candidate (x,y,rot) after this frame's shift+rotate
      const cands = [];
      const drs = inputFrame ? [0, 1, rc - 1] : [0];
      const dxs = inputFrame ? [0, -1, 1] : [0];
      for (const dr of drs) {
        const nr = (r + dr) % rc;
        let nx = x, ny = y;
        if (dr !== 0) {                               // NRS in-place pivot about the origin
          nx = x - b.rotOffX[piece * 4 + r] + b.rotOffX[piece * 4 + nr];
          ny = y - b.rotOffY[piece * 4 + r] + b.rotOffY[piece * 4 + nr];
          if (!b.fitsAt(piece, nr, nx, ny)) continue; // rotation blocked -> that branch dies
        }
        for (const dx of dxs) {
          const fx = nx + dx;
          if (dx !== 0 && !b.fitsAt(piece, nr, fx, ny)) continue;
          if (!b.fitsAt(piece, nr, fx, ny)) continue;
          cands.push([fx, ny, nr]);
        }
      }
      for (const [cx, cy, cr] of cands) {
        // gravity fires on its own clock
        const grav = ((f + 1) % g) === 0;
        if (grav) {
          if (b.fitsAt(piece, cr, cx, cy + 1)) {
            const k = key(cx, cy + 1, cr, f + 1);
            if (!seen.has(k)) { seen.add(k); next.push([cx, cy + 1, cr, f + 1]); }
          } else {
            locks.add(cr * 400 + (cy + 2) * 20 + cx);  // cannot descend -> LOCKS here
          }
        } else {
          const k = key(cx, cy, cr, f + 1);
          if (!seen.has(k)) { seen.add(k); next.push([cx, cy, cr, f + 1]); }
        }
      }
    }
    queue = next;
  }
  return locks;
}

function census(boards, g, p) {
  let nBoards = 0, boardsWithTuck = 0, sumReach = 0, sumHard = 0, tuckCount = 0, totalLocks = 0;
  const depths = [];
  let unreachableHard = 0;
  for (const b of boards) {
    let boardHasTuck = false;
    for (let piece = 0; piece < PIECES; piece++) {
      const locks = enumerateReachable(b, piece, g, p);
      if (locks === null) continue;
      // hard-drop reference set
      const hard = new Set();
      for (let r = 0; r < b.rotCount[piece]; r++) {
        for (let x = 0; x < W; x++) {
          const y = b.dropY(piece, r, x);
          if (y >= 0) hard.add(r * 400 + (y + 2) * 20 + x);
        }
      }
      sumReach += locks.size; sumHard += hard.size; totalLocks += locks.size;
      for (const L of locks) {
        if (!hard.has(L)) {
          const r = Math.floor(L / 400), y = Math.floor((L % 400) / 20) - 2, x = L % 20;
          const dy = b.dropY(piece, r, x);
          tuckCount++; boardHasTuck = true;
          if (dy >= 0) depths.push(y - dy); else unreachableHard++;
        }
      }
      nBoards++;
    }
    if (boardHasTuck) boardsWithTuck++;
  }
  depths.sort((a, b2) => a - b2);
  const pct = (q) => depths.length ? depths[Math.min(depths.length - 1, Math.floor(q * depths.length))] : 0;
  return {
    meanReach: sumReach / Math.max(1, nBoards),
    meanHard: sumHard / Math.max(1, nBoards),
    boardsWithTuckPct: 100 * boardsWithTuck / Math.max(1, boards.length),
    tucksPerPiece: tuckCount / Math.max(1, nBoards),
    p50: pct(0.50), p99: pct(0.99), maxDepth: depths.length ? depths[depths.length - 1] : 0,
    colUnreachable: unreachableHard,
  };
}

// ── board populations ───────────────────────────────────────────────────────────────────────
// The population matters more than any other knob here. Dellacherie MINIMIZES holes and row
// transitions, so it builds flat clean stacks with nothing to tuck under — measuring only those
// boards answers the wrong question. Garbage mode (protocol B) is where overhangs actually occur.
function sampleBoards(n, kind) {
  const out = [];
  let seed = 3000;
  let guard = 0;
  while (out.length < n && guard++ < n * 50) {
    const b = new PgTetris();
    b.reset(seed++, false, kind === 'garbage' ? 10 : 0);
    const depth = 25 + (seed % 60);
    for (let i = 0; i < depth && !b.gameOver; i++) {
      const a = kind === 'random' ? b.randomAction(new PgTetRng(seed * 31 + i)) : b.dellacherieAction();
      if (a < 0) break;
      b.applyPlacement(a);
    }
    if (!b.gameOver) out.push(b);
  }
  return out;
}

console.log(`S1 — reachability census. ${BOARDS} mid-game Dellacherie boards x ${PIECES} pieces per config.`);
console.log(`Exact frame sim: shift->rotate->gravity, NRS pivot, no kicks. "tuck" = a reachable lock the hard drop cannot produce.\n`);
const results = {};
for (const kind of ['dellacherie (clean)', 'garbage/10 (dirty)', 'random (messy)']) {
  const key = kind.startsWith('dell') ? 'clean' : kind.startsWith('garbage') ? 'garbage' : 'random';
  const boards = sampleBoards(BOARDS, key);
  console.log(`
### population: ${kind}  (${boards.length} boards)`);
  console.log('level        rate           |reach|  |hard|   boards    tucks   depth   depth   max   col');
  console.log('                                                w/tuck%  /piece    p50     p99  depth  unreach');
  console.log('-'.repeat(100));
  for (const [ln, g] of LEVELS) {
    for (const [rn, p] of RATES) {
      const r = census(boards, g, p);
      results[`${key}|${ln}|${rn}`] = r;
      console.log(
        `${ln.padEnd(12)} ${rn.padEnd(14)} ${r.meanReach.toFixed(1).padStart(7)} ${r.meanHard.toFixed(1).padStart(7)} ` +
        `${r.boardsWithTuckPct.toFixed(0).padStart(8)} ${r.tucksPerPiece.toFixed(1).padStart(8)} ` +
        `${String(r.p50).padStart(7)} ${String(r.p99).padStart(7)} ${String(r.maxDepth).padStart(6)} ${String(r.colUnreachable).padStart(7)}`
      );
    }
  }
}

console.log('='.repeat(100));
for (const key of ['clean', 'garbage', 'random']) {
  const d = results[`${key}|L18|DAS 10Hz`];
  console.log(`${key.padEnd(8)} @L18/DAS: boards w/tuck ${d.boardsWithTuckPct.toFixed(0).padStart(3)}%  tucks/piece ${d.tucksPerPiece.toFixed(2)}  p99 depth ${d.p99}  mean|reach| ${d.meanReach.toFixed(1)}`);
}
const g18 = results['garbage|L18|DAS 10Hz'];
console.log(`
GATE (garbage boards, DAS 10Hz @ L18): boards with >=1 tuck ${g18.boardsWithTuckPct.toFixed(0)}% (need >=20)`);
console.log(`                                       99th-pct tuck depth ${g18.p99} (need <=4 for N=160)`);
console.log(`                                       mean |reachable| ${g18.meanReach.toFixed(1)} (need <=60)`);
const go = g18.boardsWithTuckPct >= 20 && g18.p99 <= 4 && g18.meanReach <= 60;
console.log(`
S1 GATE: ${go ? 'GO' : 'NO-GO / review'}`);

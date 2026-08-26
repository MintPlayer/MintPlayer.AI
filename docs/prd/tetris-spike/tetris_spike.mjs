// Spike: 10x20 Tetris, hard-drop macro-actions, uniform-random pieces.
// Measures: (a) placements/sec, (b) lines per truncated episode for
// random-placement vs Dellacherie, to calibrate PRD gates.

const W = 10, H = 20;

// Shapes per rotation as row-strings (top row first).
const SHAPES = {
  I: [["XXXX"], ["X", "X", "X", "X"]],
  O: [["XX", "XX"]],
  T: [["XXX", ".X."], [".X", "XX", ".X"], [".X.", "XXX"], ["X.", "XX", "X."]],
  S: [[".XX", "XX."], ["X.", "XX", ".X"]],
  Z: [["XX.", ".XX"], [".X", "XX", "X."]],
  L: [["X.", "X.", "XX"], ["XXX", "X.."], ["XX", ".X", ".X"], ["..X", "XXX"]],
  J: [[".X", ".X", "XX"], ["X..", "XXX"], ["XX", "X.", "X."], ["XXX", "..X"]],
};

// Precompute per rotation: cells [(x,y)], width, height, bottom[px] (max y per column).
const PIECES = Object.entries(SHAPES).map(([name, rots]) => ({
  name,
  rots: rots.map(rows => {
    const cells = [];
    rows.forEach((r, y) => [...r].forEach((ch, x) => { if (ch === "X") cells.push([x, y]); }));
    const w = Math.max(...cells.map(c => c[0])) + 1;
    const h = Math.max(...cells.map(c => c[1])) + 1;
    const bottom = Array(w).fill(-1);
    for (const [x, y] of cells) bottom[x] = Math.max(bottom[x], y);
    return { cells, w, h, bottom };
  }),
}));

// Board = array of H row bitmasks, row 0 = top.
function newBoard() { return new Int32Array(H); }
const FULL = (1 << W) - 1;

function colTops(board) {
  // top[c] = smallest y with a filled cell, or H if column empty
  const top = Array(W).fill(H);
  for (let y = 0; y < H; y++)
    for (let c = 0; c < W; c++)
      if (top[c] === H && (board[y] >> c) & 1) top[c] = y;
  return top;
}

// Enumerate all placements for piece p on board. Returns list of
// {rot, col, yoff, cleared, board: newBoard, landingHeight, eroded}
function placements(board, p) {
  const out = [];
  const top = colTops(board);
  for (let ri = 0; ri < p.rots.length; ri++) {
    const r = p.rots[ri];
    for (let col = 0; col + r.w <= W; col++) {
      let yoff = H; // maximal drop
      for (let px = 0; px < r.w; px++)
        yoff = Math.min(yoff, top[col + px] - 1 - r.bottom[px]);
      if (yoff + 0 < 0) {
        // piece would stick out above the top => top-out placement; check min cell y
        let minY = Math.min(...r.cells.map(c => c[1]));
        if (yoff + minY < 0) continue; // illegal (no room)
      }
      const nb = Int32Array.from(board);
      for (const [px, py] of r.cells) nb[yoff + py] |= 1 << (col + px);
      // clear lines
      let cleared = 0, eroded = 0;
      const kept = [];
      for (let y = 0; y < H; y++) {
        if (nb[y] === FULL) {
          cleared++;
          // count piece cells in this row
          for (const [px, py] of r.cells) if (yoff + py === y) eroded++;
        } else kept.push(nb[y]);
      }
      const fb = new Int32Array(H);
      for (let i = 0; i < kept.length; i++) fb[H - kept.length + i] = kept[i];
      const landingHeight = H - yoff - r.h / 2;
      out.push({ rot: ri, col, board: fb, cleared, landingHeight, eroded: cleared * eroded });
    }
  }
  return out;
}

function features(pl) {
  const b = pl.board;
  // row transitions (walls count as filled)
  let rowT = 0;
  for (let y = 0; y < H; y++) {
    let prev = 1;
    for (let c = 0; c < W; c++) { const cur = (b[y] >> c) & 1; if (cur !== prev) rowT++; prev = cur; }
    if (prev === 0) rowT++;
  }
  // col transitions (floor filled, above-top empty), holes, wells
  let colT = 0, holes = 0, wells = 0;
  for (let c = 0; c < W; c++) {
    let prev = 0, seen = false;
    for (let y = 0; y < H; y++) {
      const cur = (b[y] >> c) & 1;
      if (cur !== prev) colT++;
      prev = cur;
      if (cur) seen = true;
      else if (seen) holes++;
    }
    if (prev === 0) colT++; // floor transition
  }
  // wells: empty cell with both lateral neighbors filled (wall = filled), cumulative depth
  for (let c = 0; c < W; c++) {
    for (let y = 0; y < H; y++) {
      const cur = (b[y] >> c) & 1;
      if (cur) break; // only count well cells from top of column until first fill? canonical scans all
      const left = c === 0 ? 1 : (b[y] >> (c - 1)) & 1;
      const right = c === W - 1 ? 1 : (b[y] >> (c + 1)) & 1;
      if (!cur && left && right) {
        // cumulative: count down while empty
        let d = 0, yy = y;
        while (yy < H && !((b[yy] >> c) & 1)) { d++; wells += 1; yy++; }
        break;
      }
    }
  }
  return { rowT, colT, holes, wells };
}

function dellacherie(pl) {
  const f = features(pl);
  return -pl.landingHeight + pl.eroded - f.rowT - f.colT - 4 * f.holes - f.wells;
}

// minstd LCG like PgCfRng
let seed = 12345;
function rnd() { seed = (seed * 48271) % 2147483647; return seed; }
function rndInt(n) { return rnd() % n; }

function episode(policy, maxPieces, garbageEvery = 0) {
  let board = newBoard(), lines = 0, pieces = 0, plCount = 0;
  while (pieces < maxPieces) {
    const p = PIECES[rndInt(7)];
    const pls = placements(board, p);
    plCount += pls.length;
    if (pls.length === 0) return { lines, pieces, plCount, topOut: true };
    const pick = policy(pls);
    board = pick.board;
    lines += pick.cleared;
    pieces++;
    if (garbageEvery > 0 && pieces % garbageEvery === 0) {
      if (board[0] !== 0) return { lines, pieces, plCount, topOut: true }; // shift would overflow
      for (let y = 0; y < H - 1; y++) board[y] = board[y + 1];
      board[H - 1] = FULL & ~(1 << rndInt(W)); // full row, one random gap
    }
  }
  return { lines, pieces, plCount, topOut: false };
}

const randomPolicy = pls => pls[rndInt(pls.length)];
const dellaPolicy = pls => {
  let best = pls[0], bs = -Infinity;
  for (const pl of pls) { const s = dellacherie(pl); if (s > bs) { bs = s; best = pl; } }
  return best;
};

function stats(arr) {
  const n = arr.length, mean = arr.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  return { mean: mean.toFixed(2), ci95: (1.96 * sd / Math.sqrt(n)).toFixed(2), min: Math.min(...arr), max: Math.max(...arr) };
}

// --- Random policy: full episodes until top-out ---
seed = 12345;
let rEps = [];
let rPieces = [];
for (let e = 0; e < 200; e++) { const r = episode(randomPolicy, 100000); rEps.push(r.lines); rPieces.push(r.pieces); }
console.log("random  full-episode lines:", JSON.stringify(stats(rEps)), "pieces:", JSON.stringify(stats(rPieces)));

// --- Dellacherie: truncated at 500 pieces ---
seed = 999;
let dEps = [], dTop = 0;
const t0 = Date.now();
let totalPl = 0, totalPieces = 0;
for (let e = 0; e < 50; e++) {
  const r = episode(dellaPolicy, 500);
  dEps.push(r.lines); if (r.topOut) dTop++;
  totalPl += r.plCount; totalPieces += r.pieces;
}
const dt = (Date.now() - t0) / 1000;
console.log("della   500-piece lines:", JSON.stringify(stats(dEps)), "topOuts:", dTop + "/50");
console.log(`throughput: ${(totalPieces / dt).toFixed(0)} pieces/s eval'd, ${(totalPl / dt).toFixed(0)} placements/s (incl. feature eval), avg branching ${(totalPl / totalPieces).toFixed(1)}`);

// --- Dellacherie long run: does it survive 10k pieces? ---
seed = 777;
const t1 = Date.now();
const long = episode(dellaPolicy, 10000);
console.log(`della 10k-piece run: lines=${long.lines} topOut=${long.topOut} in ${((Date.now() - t1) / 1000).toFixed(1)}s`);

// --- Garbage mode (bottom row w/ one gap every 10 pieces): survival ---
seed = 4242;
let gr = [], gd = [];
for (let e = 0; e < 100; e++) gr.push(episode(randomPolicy, 100000, 10).pieces);
for (let e = 0; e < 100; e++) gd.push(episode(dellaPolicy, 100000, 10).pieces);
console.log("garbage/10 survival pieces — random:", JSON.stringify(stats(gr)));
console.log("garbage/10 survival pieces — della :", JSON.stringify(stats(gd)));
seed = 4243;
let gd5 = [];
for (let e = 0; e < 100; e++) gd5.push(episode(dellaPolicy, 100000, 5).pieces);
console.log("garbage/5  survival pieces — della :", JSON.stringify(stats(gd5)));

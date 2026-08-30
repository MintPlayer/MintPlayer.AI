// M57.5 post-mortem — "is the WIDENED evaluator simply less safe to approximate?"
//
// Every widened-basis training run tops out 20/20 episodes at ~75-140 pieces, while the old narrow-basis
// net reached 455 pieces by 40K steps. The net fits the widened target at R^2 ~= 0.54, which is not
// terrible — so the question is whether a 54%-accurate approximation of THIS evaluator is simply lethal,
// where the same accuracy on the NARROW one is survivable.
//
// Test: play both evaluators exactly, then with Gaussian noise added per action, and watch survival.
// Narrow = the pre-M57.1 Dellacherie basis. Widened = what the engine ships now.
// Run:  node docs/prd/tetris-spike/s7_noise_robustness.mjs [episodes]

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;
const W = 10, H = 20, EPS = Number(process.argv[2] ?? 12), CAP = 500;

let _s = 424242;
const rnd = () => { _s = (Math.imul(_s, 1103515245) + 12345) & 0x7fffffff; return _s / 0x7fffffff; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

// the pre-M57.1 narrow basis, recomputed here so both are measured on the same engine
function narrowAt(b, piece, rot, x) {
  const y = b.dropY(piece, rot, x);
  if (y < 0) return null;
  const saved = b.saveRows();
  b.simPlace(piece, rot, x, y);
  const s = -b.simLanding + b.simEroded - b.rowTransitions() - b.colTransitions() - 4 * b.holes() - b.wellSum();
  b.restoreRows(saved);
  return s;
}
function widenedAt(b, piece, rot, x) {
  const y = b.dropY(piece, rot, x);
  if (y < 0) return null;
  return b.dellaScoreFor(piece, rot, x);
}

function play(seed, scorer, noiseSd) {
  const b = new PgTetris();
  b.reset(seed, false, 0);
  let n = 0;
  while (!b.gameOver && n < CAP) {
    let best = -1e18, bp = null;
    for (let a = 0; a < 40; a++) {
      if (!b.placementLegal(a)) continue;
      const r = b.actionRot(a), x = b.actionCol(a);
      let v = scorer(b, b.current, r, x);
      if (v === null) continue;
      if (noiseSd > 0) v += noiseSd * gauss();
      if (v > best) { best = v; bp = a; }
    }
    if (bp === null) break;
    b.applyPlacement(bp);
    n++;
  }
  return { pieces: n, score: b.score, lines: b.lines, tetrises: b.tetrises, topout: b.gameOver };
}
function agg(rows) {
  const n = rows.length, m = f => rows.reduce((a, r) => a + f(r), 0) / n;
  return { pieces: m(r => r.pieces), score: m(r => r.score), lines: m(r => r.lines),
           tetrises: m(r => r.tetrises), topout: m(r => r.topout ? 1 : 0) };
}

// per-action spread of each basis, so the noise levels are COMPARABLE (matched R^2, not matched absolute sd)
function spread(scorer) {
  const all = [];
  for (let e = 0; e < 30; e++) {
    const b = new PgTetris();
    b.reset(7700 + e, false, 0);
    for (let i = 0; i < 40 && !b.gameOver; i++) b.applyPlacement(b.dellacherieAction());
    if (b.gameOver) continue;
    const vs = [];
    for (let a = 0; a < 40; a++) {
      if (!b.placementLegal(a)) continue;
      const v = scorer(b, b.current, b.actionRot(a), b.actionCol(a));
      if (v !== null) vs.push(v);
    }
    if (vs.length < 2) continue;
    const m = vs.reduce((x, y) => x + y, 0) / vs.length;
    all.push(Math.sqrt(vs.reduce((x, y) => x + (y - m) ** 2, 0) / vs.length));
  }
  return all.reduce((a, x) => a + x, 0) / all.length;
}

const sdN = spread(narrowAt), sdW = spread(widenedAt);
console.log(`per-state action spread:  narrow sd=${sdN.toFixed(2)}   widened sd=${sdW.toFixed(2)}\n`);
console.log('basis      R^2   noise sd   pieces   score    lines  tetris  topout%');
console.log('-'.repeat(72));
for (const [name, scorer, sd] of [['narrow ', narrowAt, sdN], ['widened', widenedAt, sdW]]) {
  for (const r2 of [1.0, 0.8, 0.54, 0.3]) {
    const noise = sd * Math.sqrt(1 - r2);
    const r = agg([...Array(EPS)].map((_, e) => play(5000 + e, scorer, noise)));
    console.log(`${name}  ${r2.toFixed(2)}  ${noise.toFixed(2).padStart(9)} ${r.pieces.toFixed(0).padStart(8)} ${r.score.toFixed(0).padStart(8)} ${r.lines.toFixed(1).padStart(8)} ${r.tetrises.toFixed(2).padStart(7)} ${(r.topout * 100).toFixed(0).padStart(8)}`);
  }
  console.log('');
}
console.log('R^2 0.54 is what the trained net actually achieved on the widened target.');

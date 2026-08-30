const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;
const vals = [];
for (let e = 0; e < 40; e++) {
  const b = new PgTetris();
  b.reset(6000 + e, false, e % 2 ? 10 : 0);
  for (let i = 0; i < 40 + e && !b.gameOver; i++) b.applyPlacement(b.dellacherieAction());
  if (b.gameOver) continue;
  for (let a = 0; a < 40; a++) {
    if (!b.placementLegal(a)) continue;
    vals.push(b.dellaScoreFor(b.current, b.actionRot(a), b.actionCol(a)));
  }
}
vals.sort((x, y) => x - y);
const q = p => vals[Math.floor(p * (vals.length - 1))];
const mean = vals.reduce((a, x) => a + x, 0) / vals.length;
const sd = Math.sqrt(vals.reduce((a, x) => a + (x - mean) ** 2, 0) / vals.length);
console.log(`n=${vals.length}  min=${q(0).toFixed(1)}  p05=${q(.05).toFixed(1)}  median=${q(.5).toFixed(1)}  p95=${q(.95).toFixed(1)}  max=${q(1).toFixed(1)}`);
console.log(`mean=${mean.toFixed(2)}  sd=${sd.toFixed(2)}`);
console.log(`=> current /10 target: mean ${(mean/10).toFixed(2)}, sd ${(sd/10).toFixed(2)}, range ${(q(.05)/10).toFixed(1)}..${(q(.95)/10).toFixed(1)}`);
console.log(`   M54 narrow basis for comparison targeted roughly sd~1, |target| <~3`);

const mod = await import('../../../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris } = mod;
// variance of the CENTRED per-state target — if training loss ~= this, the net explains ~none of it
let all = [], perState = [];
for (let e = 0; e < 60; e++) {
  const b = new PgTetris();
  b.reset(6500 + e, false, e % 2 ? 10 : 0);
  for (let i = 0; i < 30 + e && !b.gameOver; i++) b.applyPlacement(b.dellacherieAction());
  if (b.gameOver) continue;
  const vs = [];
  for (let a = 0; a < 40; a++) {
    if (!b.placementLegal(a)) continue;
    vs.push(b.dellaScoreFor(b.current, b.actionRot(a), b.actionCol(a)));
  }
  if (vs.length < 2) continue;
  const m = vs.reduce((x, y) => x + y, 0) / vs.length;
  const centred = vs.map(v => (v - m) / 10);
  perState.push(Math.sqrt(centred.reduce((x, y) => x + y * y, 0) / centred.length));
  all.push(...centred);
}
const mean = all.reduce((a, x) => a + x, 0) / all.length;
const varr = all.reduce((a, x) => a + (x - mean) ** 2, 0) / all.length;
console.log(`centred targets: n=${all.length} mean=${mean.toFixed(3)} variance=${varr.toFixed(3)} sd=${Math.sqrt(varr).toFixed(3)}`);
console.log(`mean per-state sd: ${(perState.reduce((a,x)=>a+x,0)/perState.length).toFixed(3)}`);
console.log(`training loss plateaued at ~1.10  =>  R^2 = ${(1 - 1.10/varr).toFixed(3)}`);

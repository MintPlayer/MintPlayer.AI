// Head-to-head: shipped net (tet4, 128x128) vs tet6 keep-best (256x256, dense-weight 8).
// HELD-OUT seeds 9000+e (training keep-best was selected on 5000+e — reusing those would bias
// toward tet6). Protocol A: uniform, no garbage, 500-piece cap, metric NES score. Protocol B:
// garbage/10, survival (pieces placed). Plain net 30 eps cap 5000; net-search(8) 10 eps capB 1500.
import { readFileSync } from 'node:fs';
const { PgTetris, PgTetDuelingNet } = await import('file:///C:/Repos/MintPlayer.AI/src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');

function parseCkpt(buffer) {
  const dv = new DataView(buffer);
  let off = 0;
  const i32 = () => { const v = dv.getInt32(off, true); off += 4; return v; };
  const u32 = () => { const v = dv.getUint32(off, true); off += 4; return v; };
  const u8 = () => dv.getUint8(off++);
  const read7bit = () => { let r = 0, s = 0, b; do { b = u8(); r |= (b & 0x7f) << s; s += 7; } while (b & 0x80); return r; };
  const readString = () => { const len = read7bit(); const bytes = new Uint8Array(buffer, off, len); off += len; return new TextDecoder().decode(bytes); };
  const readFloats = () => { const n = i32(); const a = new Array(n); for (let i = 0; i < n; i++) { a[i] = dv.getFloat32(off, true); off += 4; } return a; };
  if (u32() !== 0x434e4c52) throw new Error('bad magic');
  if (readString() !== 'dueling-q') throw new Error('bad kind');
  const version = i32(), inputSize = i32(), hiddenCount = i32();
  const hidden = []; for (let i = 0; i < hiddenCount; i++) hidden.push(i32());
  const actions = i32();
  const noisy = version >= 2 && u8() !== 0;
  const tw = [], tb = [];
  for (let l = 0; l < hiddenCount; l++) { for (const f of readFloats()) tw.push(f); for (const f of readFloats()) tb.push(f); }
  let vw, vb, aw, ab;
  if (!noisy) { vw = readFloats(); vb = readFloats(); aw = readFloats(); ab = readFloats(); }
  else { vw = readFloats(); readFloats(); vb = readFloats(); readFloats(); aw = readFloats(); readFloats(); ab = readFloats(); readFloats(); }
  return new PgTetDuelingNet(inputSize, actions, hidden, tw, tb, vw, vb, aw, ab);
}

function loadNet(path) {
  const buf = readFileSync(path);
  return parseCkpt(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

function runProtocol(label, act, { episodes, garbage, cap, metric }) {
  let sum = 0, sumSq = 0, lines = 0, tetrises = 0, tops = 0;
  for (let e = 0; e < episodes; e++) {
    const g = new PgTetris();
    g.reset(9000 + e, false, garbage);
    for (let s = 0; s < cap && !g.gameOver; s++) {
      const a = act(g);
      if (a < 0 || g.applyPlacement(a) < 0) break;
    }
    if (g.gameOver) tops++;
    const m = metric === 'score' ? g.score : g.piecesPlaced;
    sum += m; sumSq += m * m; lines += g.lines; tetrises += g.tetrises;
  }
  const mean = sum / episodes;
  const ci = 1.96 * Math.sqrt(Math.max(0, sumSq / episodes - mean * mean) / episodes);
  console.log(`  ${label.padEnd(26)} mean ${mean.toFixed(1).padStart(9)} ± ${ci.toFixed(1)}, lines ${(lines / episodes).toFixed(1)} · tetrises ${(tetrises / episodes).toFixed(2)} · top-outs ${tops}/${episodes}`);
  return { mean, ci };
}

const nets = {
  shipped: loadNet(process.argv[2]),
  tet6: loadNet(process.argv[3]),
};
for (const [name, net] of Object.entries(nets))
  console.log(`${name}: input ${net.inputSize}, hidden ${net.hidden.join('x')}`);

const results = {};
console.log('\nProtocol A — uniform, no garbage, 500-piece cap, NES score (held-out seeds 9000+e):');
for (const [name, net] of Object.entries(nets)) {
  results[`${name}.A`] = runProtocol(`${name} net`, g => g.netAction(net), { episodes: 30, garbage: 0, cap: 500, metric: 'score' });
}
for (const [name, net] of Object.entries(nets)) {
  results[`${name}.A.search`] = runProtocol(`${name} net-search(8)`, g => g.netSearchAction(net, 8), { episodes: 10, garbage: 0, cap: 500, metric: 'score' });
}

console.log('\nProtocol B — garbage/10, survival in pieces placed:');
for (const [name, net] of Object.entries(nets)) {
  results[`${name}.B`] = runProtocol(`${name} net`, g => g.netAction(net), { episodes: 30, garbage: 10, cap: 5000, metric: 'pieces' });
}
for (const [name, net] of Object.entries(nets)) {
  results[`${name}.B.search`] = runProtocol(`${name} net-search(8)`, g => g.netSearchAction(net, 8), { episodes: 10, garbage: 10, cap: 1500, metric: 'pieces' });
}

function verdict(tag) {
  const a = results[`shipped.${tag}`], b = results[`tet6.${tag}`];
  const sep = b.mean - b.ci > a.mean + a.ci ? 'tet6 CI-SEPARATED ahead'
    : a.mean - a.ci > b.mean + b.ci ? 'shipped CI-SEPARATED ahead' : 'OVERLAPPING';
  console.log(`${tag}: tet6 ${(100 * (b.mean - a.mean) / a.mean).toFixed(1)}% vs shipped — ${sep}`);
}
console.log('\nVerdicts:');
verdict('A'); verdict('A.search'); verdict('B'); verdict('B.search');

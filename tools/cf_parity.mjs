// C#<->TS parity harness (M49.1/M50.1 gate): runs the exact protocol of CrazyFruitsParityTests.ParityChecksum
// against the GENERATED TypeScript twin and prints the rolling checksum. Must print the same number the
// C# test pins.
const mod = await import('../src/RLDemo.Web/ClientApp/src/app/crazy-fruits/crazyfruits_solver.ts');
const { PgCrazyFruits, PgCfRng } = mod;

const board = new PgCrazyFruits();
board.reset(12345);
const policy = new PgCfRng(999);

const P = 1000000007;
let h = 0;
for (let move = 0; move < 1000; move++) {
  const action = board.randomAction(policy);
  const points = board.applySwap(action);
  h = (h * 31 + action) % P;
  h = (h * 31 + points) % P;
  for (let i = 0; i < 64; i++) h = (h * 31 + board.grid[i]) % P;
}
console.log(`checksum=${h} score=${board.score} reshuffles=${board.reshuffles}`);


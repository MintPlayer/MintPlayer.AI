// C#<->TS parity harness (M54.1 gate): runs the exact protocol of TetrisParityTests.ParityChecksum
// against the GENERATED TypeScript twin and prints the rolling checksum. Must print the same number the
// C# test pins. The protocol mixes uniform and 7-bag episodes, both garbage rates, and all three
// deterministic tiers (random / Dellacherie / Dellacherie-search) so every rules + feature path is pinned.
const mod = await import('../src/RLDemo.Web/ClientApp/src/app/tetris/tetris_solver.ts');
const { PgTetris, PgTetRng } = mod;

const board = new PgTetris();
board.reset(1000, false, 10);
const policy = new PgTetRng(999);

const P = 1000000007;
let h = 0;
let episodes = 0;
for (let step = 0; step < 1000; step++) {
  if (board.gameOver) {
    episodes++;
    board.reset(1000 + episodes, episodes % 2 === 1, episodes % 2 === 0 ? 10 : 5);
  }
  const action = step % 50 === 7 ? board.dellaSearchAction(8, 5)
    : step % 4 === 0 ? board.dellacherieAction()
    : board.randomAction(policy);
  const cleared = board.applyPlacement(action);
  h = (h * 31 + action) % P;
  h = (h * 31 + cleared) % P;
  for (let y = 0; y < 20; y++) h = (h * 31 + board.rows[y]) % P;
  h = (h * 31 + board.current) % P;
  h = (h * 31 + board.next) % P;
}
console.log(`checksum=${h} episodes=${episodes} lines=${board.lines}`);

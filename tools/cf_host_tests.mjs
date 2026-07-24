// Functional tests for the Crazy Fruits web HOST layer (crazy-fruits-game.ts): the 30-move round rule,
// the endless-mode toggle, and best-score exemption — pure game functionality, no rendering. Runs the REAL
// browser TypeScript under node's type stripping (node 22.6+):  node tools/cf_host_tests.mjs
// (The engine rules themselves are covered by the C# suite + tools/cf_parity.mjs cross-language pin;
// the stepwise-protocol ≡ applySwap equivalence is pinned by CrazyFruitsSpecialsTests in xunit.)
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = dirname(dirname(fileURLToPath(import.meta.url)));
const appDir = join(repo, 'src', 'RLDemo.Web', 'ClientApp', 'src', 'app', 'crazy-fruits');

// node can't resolve the game's extensionless './crazyfruits_solver' import (the bundler does in the app):
// rewrite it into a temp copy, pointing at the generated twin explicitly.
const solverUrl = pathToFileURL(join(appDir, 'crazyfruits_solver.ts')).href;
const gameSrc = readFileSync(join(appDir, 'crazy-fruits-game.ts'), 'utf8')
  .replace("from './crazyfruits_solver'", `from '${solverUrl}'`);
const tmp = mkdtempSync(join(tmpdir(), 'cf-host-'));
const gamePath = join(tmp, 'crazy-fruits-game.ts');
writeFileSync(gamePath, gameSrc);

globalThis.localStorage = { getItem: () => null, setItem: () => {} };
const { CrazyFruitsGame } = await import(pathToFileURL(gamePath).href);

let failures = 0;
const assert = (cond, msg) => { if (!cond) { failures++; console.error(`FAIL: ${msg}`); } };
const drain = g => { let n = 0; while (g.animating) { g.update(50); if (++n > 10000) { assert(false, 'animation never drained'); break; } } };
const play = (g, moves) => { for (let i = 0; i < moves; i++) { assert(g.tryAction(g.board.greedyAction()), `move ${i} rejected`); drain(g); } };

// 1) A normal round closes at exactly 30 moves, not before, and updates "best".
const g = new CrazyFruitsGame();
play(g, 29);
g.checkRoundOver(30);
assert(!g.roundOver, 'round must not close before 30 moves');
play(g, 1);
g.checkRoundOver(30);
assert(g.roundOver, 'round must close at 30 moves');
assert(g.best === g.board.score, 'best must track a normal round');

// 2) Round over blocks further moves; tap-to-restart is the component's job, newGame resets.
assert(!g.trySwap(0, 1), 'no moves allowed on the round-over screen');

// 3) Endless from the round-over screen: resumes the SAME board past 30; best freezes.
const bestBefore = g.best;
g.setFreePlay(true);
assert(!g.roundOver, 'enabling endless must dismiss the round-over screen');
play(g, 10);
g.checkRoundOver(30);
assert(!g.roundOver, 'endless mode must never close the round');
assert(g.board.movesMade === 40, `expected 40 moves, got ${g.board.movesMade}`);
assert(g.best === bestBefore, 'best must not move once endless touched the game');

// 4) A game STARTED under endless is exempt from best from move one.
g.newGame();
play(g, 5);
assert(g.best === bestBefore, 'a game started under endless must not update best');

// 5) Turning endless off restores rounds and best for the NEXT game.
g.setFreePlay(false);
g.newGame();
play(g, 30);
g.checkRoundOver(30);
assert(g.roundOver, 'rounds must work again with endless off');
assert(g.best >= bestBefore, 'best tracking must resume');

// 6) Illegal swaps never consume a move or points, in any mode.
g.newGame();
const before = g.board.movesMade;
let illegal = -1;
for (let a = 0; a < 112; a++) if (!g.board.swapIsLegal(a)) { illegal = a; break; }
if (illegal >= 0) {
  g.tryAction(illegal);
  drain(g);
  assert(g.board.movesMade === before, 'an illegal swap must not consume a move');
}

if (failures) { console.error(`${failures} failure(s)`); process.exit(1); }
console.log('OK: host round/endless/best functionality verified');

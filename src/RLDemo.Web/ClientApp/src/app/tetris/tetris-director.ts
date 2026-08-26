// Watch-AI director (client-side, no server — Pattern C): when the board is idle, picks the next
// PLACEMENT with the selected tier's policy and hands it to the animating game. The policies live in the
// single-source engine (tetris_solver.pg) — random/Dellacherie/Dellacherie-search are the same scripted
// tiers the C# eval gates run; 'net' is the trained masked dueling DQN via the generated PgTetDuelingNet
// forward, and 'net-search' is the beam-8 one-ply expectimax rollout over it (TETRIS_PRD.md §3.8/§3.9 —
// search runs synchronously: ~10 ms/move, far inside the frame budget, so no worker).

import { TetrisGame } from './tetris-game';
import { PgTetDuelingNet, PgTetRng } from './tetris_solver';
import { loadTetrisNet } from './tetris-net';

export type Tier = 'random' | 'dellacherie' | 'della-search' | 'net' | 'net-search';

const MOVE_PAUSE_MS = 220;
const GAME_OVER_PAUSE_MS = 2200;

export class TetrisDirector {
  private net: PgTetDuelingNet | null = null;
  netStatus: 'loading' | 'ready' | 'missing' = 'loading';

  tier: Tier = 'net';
  lastLines = 0;
  lastScore = 0;
  episodes = 0;

  private readonly policyRng = new PgTetRng((Date.now() % 2147483646) + 1);
  private pause = 0;
  private overHandled = false;

  constructor(private readonly game: TetrisGame) {
    void loadTetrisNet().then(net => {
      // Stale-ckpt guard (the M51.1 lesson): a net trained on an older observation layout cannot forward
      // against the current engine — treat it as missing (Dellacherie fallback) instead of erroring per move.
      const obsSize = this.game.board.buildObservation().length;
      if (net && net.inputSize !== obsSize) {
        console.warn(`tetris: checkpoint input ${net.inputSize} ≠ observation ${obsSize} — stale checkpoint, falling back to Dellacherie`);
        net = null;
      }
      this.net = net;
      this.netStatus = net ? 'ready' : 'missing';
    });
  }

  /** The tier actually playing (net tiers fall back to Dellacherie while loading / when missing). */
  get effectiveTier(): Tier {
    return (this.tier === 'net' || this.tier === 'net-search') && !this.net ? 'dellacherie' : this.tier;
  }

  reset(): void {
    this.game.newGame();
    this.pause = MOVE_PAUSE_MS;
  }

  update(dtMs: number): void {
    if (this.game.animating) return;

    if (this.game.gameOver) {
      if (!this.overHandled) {
        // Let the game-over screen breathe before the next episode.
        this.overHandled = true;
        this.lastLines = this.game.board.lines;
        this.lastScore = this.game.board.score;
        this.pause = GAME_OVER_PAUSE_MS;
        return;
      }
      this.pause -= dtMs;
      if (this.pause > 0) return;
      this.episodes++;
      this.overHandled = false;
      this.game.newGame();
      this.pause = MOVE_PAUSE_MS;
      return;
    }

    this.pause -= dtMs;
    if (this.pause > 0) return;
    this.pause = MOVE_PAUSE_MS;

    const board = this.game.board;
    let action: number;
    switch (this.effectiveTier) {
      case 'random': action = board.randomAction(this.policyRng); break;
      case 'dellacherie': action = board.dellacherieAction(); break;
      case 'della-search': action = board.dellaSearchAction(8, 5); break;
      case 'net': action = board.netAction(this.net!); break;
      case 'net-search': action = board.netSearchAction(this.net!, 8); break;
    }
    if (action >= 0) this.game.startDrop(action);
  }
}

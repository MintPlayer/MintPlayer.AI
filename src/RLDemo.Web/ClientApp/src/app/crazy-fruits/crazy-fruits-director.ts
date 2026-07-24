// Watch-AI director (client-side, no server — Pattern C): when the board is idle, picks the next swap with
// the selected tier's policy and feeds it to the animating game. The policies themselves live in the
// single-source engine (crazyfruits_solver.pg) — random/greedy/expectimax-1 are the same scripted baselines
// the C# eval gates run, and 'net' is the trained DQN via the generated PgCfDuelingNet forward.

import { CrazyFruitsGame } from './crazy-fruits-game';
import { PgCfDuelingNet, PgCfRng } from './crazyfruits_solver';
import { loadCrazyFruitsNet } from './crazyfruits-net';

export type Tier = 'random' | 'greedy' | 'specials' | 'expectimax' | 'expectimax2' | 'net';
export const MOVES_PER_EPISODE = 30; // the training/eval episode framing (PRD §3.5)

const MOVE_PAUSE_MS = 350;

export class CrazyFruitsDirector {
  private net: PgCfDuelingNet | null = null;
  netStatus: 'loading' | 'ready' | 'missing' = 'loading';

  tier: Tier = 'net';
  lastScore = 0;
  episodes = 0;

  private readonly policyRng = new PgCfRng((Date.now() % 2147483646) + 1);
  private pause = 0;

  constructor(private readonly game: CrazyFruitsGame) {
    void loadCrazyFruitsNet().then(net => {
      this.net = net;
      this.netStatus = net ? 'ready' : 'missing';
    });
  }

  /** The tier actually playing (the net tier falls back to expectimax while loading / when missing). */
  get effectiveTier(): Tier {
    return this.tier === 'net' && !this.net ? 'expectimax' : this.tier;
  }

  reset(): void {
    this.game.newGame();
    this.pause = MOVE_PAUSE_MS;
  }

  update(dtMs: number): void {
    if (this.game.animating) return;

    if (this.game.board.movesMade >= MOVES_PER_EPISODE) {
      this.lastScore = this.game.board.score;
      this.episodes++;
      this.game.newGame();
      this.pause = MOVE_PAUSE_MS * 2;
      return;
    }

    this.pause -= dtMs;
    if (this.pause > 0) return;
    this.pause = MOVE_PAUSE_MS;

    const board = this.game.board;
    let action: number;
    switch (this.effectiveTier) {
      case 'random': action = board.randomAction(this.policyRng); break;
      case 'greedy': action = board.greedyAction(); break;
      case 'specials': action = board.specialsGreedyAction(); break;
      case 'expectimax': action = board.expectimaxAction(); break;
      case 'expectimax2': action = board.expectimax2Action(); break;
      case 'net': action = board.netAction(this.net!); break;
    }
    if (action >= 0) this.game.tryAction(action);
  }
}

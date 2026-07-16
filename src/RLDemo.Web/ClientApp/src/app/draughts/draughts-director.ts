import { PgDraughtsState, PgDraughtsMcts, PgDraughtsNet } from './draughts_solver';
import { DraughtsDifficulty, loadDraughtsNet, loadDifficulties } from './draughts-net';

// Client-side draughts "play against the AI" director — the whole AI runs in the browser, single-sourced
// from draughts_solver.pg (the same engine + conv net + MCTS the C# training uses); the chess pattern
// (M40/M47.5). You are White; the AI replies with PgDraughtsMcts over the net loaded from the shipped
// .ckpt. Currently the english 8×8 variant (the validated M47.4 net); the 10×10 dammen net drops in via
// the manifest once its campaign has run.
//
// The engine speaks mover-relative action indices; the board UI speaks (from, to) squares. A complete
// capture sequence is ONE move, so clicking its destination plays every jump at once (captures are
// forced by rule — when a capture exists, only capture moves are offered).

export interface DecodedMove { index: number; from: number; to: number; captures: number[]; }

export class DraughtsDirector {
  private state = PgDraughtsState.english();
  private net: PgDraughtsNet | null = null;
  private readonly netCache = new Map<string, PgDraughtsNet | null>(); // by ckpt url

  netReady = false;   // the current tier's checkpoint fetch has resolved (success or not)
  netMissing = false; // resolved but no usable net → AI falls back to a random legal move
  thinking = false;   // an AI search is in flight
  lastMove: { from: number; to: number; captures: number[] } | null = null;

  difficulties: DraughtsDifficulty[] = [];
  current: DraughtsDifficulty = { label: 'Default', ckpt: '/models/checkers8.az.ckpt', sims: 8, temperature: 0, cpuct: 1.5 };

  private sims = 8;
  private cpuct = 1.5;
  private temperature = 0;

  /** Resolves when the manifest + the default tier's checkpoint have settled (the UI awaits this). */
  readonly ready: Promise<void>;

  constructor() {
    this.ready = this.init();
  }

  private async init(): Promise<void> {
    this.difficulties = await loadDifficulties();
    // Default to the strongest tier (the manifest is ordered weakest→strongest).
    await this.setDifficulty(this.difficulties[this.difficulties.length - 1]);
  }

  /** Switch tier: adopt its search knobs and load its checkpoint (cached by url). */
  async setDifficulty(d: DraughtsDifficulty): Promise<void> {
    this.current = d;
    this.sims = d.sims;
    this.cpuct = d.cpuct;
    this.temperature = d.temperature;
    this.netReady = false;
    if (this.netCache.has(d.ckpt)) {
      this.net = this.netCache.get(d.ckpt)!;
    } else {
      this.net = await loadDraughtsNet(d.ckpt);
      this.netCache.set(d.ckpt, this.net);
    }
    this.netMissing = this.net === null;
    this.netReady = true;
  }

  reset(): void {
    this.state = PgDraughtsState.english();
    this.lastMove = null;
    this.thinking = false;
  }

  /** A copy of the N²-cell board for rendering (never the engine's internal array). */
  board(): number[] { return [...this.state.squares]; }
  get size(): number { return this.state.size; }
  get whiteToMove(): boolean { return this.state.whiteToMove; }

  private legal(): DecodedMove[] {
    return this.state.legalMoves().map(m => ({
      index: this.state.moveIndex(m), from: m.from, to: m.to, captures: m.captures,
    }));
  }

  /** Destination squares for a piece on `from` (a multi-jump shows only its FINAL square — one click
   *  plays the whole sequence). */
  legalTargets(from: number): number[] {
    return this.legal().filter(m => m.from === from).map(m => m.to);
  }

  /** Squares of pieces that have a legal move right now (capture-forcing narrows this a lot). */
  movablePieces(): Set<number> {
    return new Set(this.legal().map(m => m.from));
  }

  /** True when the side to move MUST capture (all legal moves are capture sequences). */
  mustCapture(): boolean {
    const moves = this.state.legalMoves();
    return moves.length > 0 && moves[0].captures.length > 0;
  }

  /** Apply the human's (White) move if legal. applyIndex resolves rare same-(from,to) capture forks with
   *  the engine's canonical pick — identical to training. */
  humanMove(from: number, to: number): boolean {
    if (this.thinking || !this.state.whiteToMove) return false;
    const match = this.legal().find(m => m.from === from && m.to === to);
    if (!match) return false;
    this.state = this.state.applyIndex(match.index);
    this.lastMove = { from, to, captures: match.captures };
    return true;
  }

  /** Play the AI's move for whichever side is to move (the reply in play mode; either side in watch
   *  mode). Yields to the UI first so the board paints before the blocking search. */
  async aiStep(): Promise<void> {
    if (this.result() !== 0) return;
    this.thinking = true;
    await new Promise(resolve => setTimeout(resolve, 30));
    let index: number;
    if (this.net) {
      if (this.temperature > 0) {
        index = DraughtsDirector.sampleFromPi(PgDraughtsMcts.search(this.net, this.state, this.sims, this.cpuct), this.temperature);
      } else {
        index = PgDraughtsMcts.chooseMove(this.net, this.state, this.sims, this.cpuct);
      }
    } else {
      const moves = this.state.legalMoveIndices();
      index = moves[Math.floor(Math.random() * moves.length)];
    }
    const m = this.legal().find(mv => mv.index === index);
    this.state = this.state.applyIndex(index);
    this.lastMove = m ? { from: m.from, to: m.to, captures: m.captures } : null;
    this.thinking = false;
  }

  // Temperature sampling over the visit-count distribution π (nonzero = visited). T→0 approaches argmax.
  private static sampleFromPi(pi: number[], temperature: number): number {
    const invT = 1 / temperature;
    const idx: number[] = [];
    const weights: number[] = [];
    let sum = 0;
    for (let a = 0; a < pi.length; a++) {
      if (pi[a] > 0) { const w = Math.pow(pi[a], invT); idx.push(a); weights.push(w); sum += w; }
    }
    if (idx.length === 0) return 0;
    let r = Math.random() * sum;
    for (let k = 0; k < idx.length; k++) { r -= weights[k]; if (r <= 0) return idx[k]; }
    return idx[idx.length - 1];
  }

  /** Captured-material tally per side (english start = 12 men each; a promotion keeps the piece, so
   *  counting surviving pieces is exact). */
  captured(): { white: number; black: number } {
    let white = 0, black = 0;
    for (const p of this.state.squares) {
      if (p > 0) white++;
      else if (p < 0) black++;
    }
    return { white: 12 - white, black: 12 - black };
  }

  /** Terminal result from the side-to-move's view: 0 ongoing, 1 loss (no move), 2 draw (no-progress). */
  result(): number { return this.state.result(); }

  /** Human-facing outcome. In draughts a blocked side loses (there is no stalemate draw). */
  outcome(): 'ongoing' | 'you-win' | 'ai-wins' | 'draw' {
    const r = this.result();
    if (r === 1) return this.state.whiteToMove ? 'ai-wins' : 'you-win'; // the side to move has lost
    if (r === 2) return 'draw';
    return 'ongoing';
  }
}

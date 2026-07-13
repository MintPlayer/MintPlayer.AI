import { PgChessState, PgChessMcts, PgPolicyValueNet } from './chess_solver';
import { ChessDifficulty, loadChessNet, loadDifficulties } from './chess-net';

// Client-side chess "play against the AI" director — the whole AI runs in the browser, single-sourced from
// chess_solver.pg (the same engine + net + MCTS the C# training uses). You are White; the AI is Black and replies
// with PgChessMcts over the net loaded from the shipped .ckpt. No server move computation (the FruitCake pattern).
//
// The engine speaks AlphaZero action indices (0..4671). The board UI speaks (from, to) squares, so the director
// resolves a human click to a legal index (auto-queen on promotion) and reads decoded moves back for highlighting.

// Start position as the engine's mailbox: index = rank*8 + file (a1 = 0); +1..+6 White P,N,B,R,Q,K, −1..−6 Black.
// Mirrors ChessFen.StartFen / ChessState.StartPosition (the engine has no FEN parser on the browser side).
const START_SQUARES: number[] = [
   4,  2,  3,  5,  6,  3,  2,  4,   // rank 1  a1..h1  R N B Q K B N R
   1,  1,  1,  1,  1,  1,  1,  1,   // rank 2  white pawns
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
  -1, -1, -1, -1, -1, -1, -1, -1,   // rank 7  black pawns
  -4, -2, -3, -5, -6, -3, -2, -4,   // rank 8  a8..h8
];
const CASTLE_ALL = 1 | 2 | 4 | 8; // WK|WQ|BK|BQ

export interface DecodedMove { index: number; from: number; to: number; promo: number; }

export class ChessDirector {
  private state = ChessDirector.startState();
  private net: PgPolicyValueNet | null = null;
  private readonly netCache = new Map<string, PgPolicyValueNet | null>(); // by ckpt url; sims-only switches don't refetch

  netReady = false;   // the current tier's checkpoint fetch has resolved (success or not)
  netMissing = false; // resolved but no usable net → AI falls back to a random legal move
  thinking = false;   // an AI search is in flight
  lastMove: { from: number; to: number } | null = null;

  // Difficulty (M40.4): the tier list comes from the Lab-written manifest; `current` selects one.
  difficulties: ChessDifficulty[] = [];
  current: ChessDifficulty = { label: 'Default', ckpt: '/models/chess.az.ckpt', sims: 96, temperature: 0, cpuct: 1.5 };

  // Search knobs, set from the current difficulty. sims = latency/strength; temperature > 0 samples for variety.
  private sims = 96;
  private cpuct = 1.5;
  private temperature = 0;

  /** Resolves when the manifest + the default tier's checkpoint have settled (the UI awaits this). */
  readonly ready: Promise<void>;

  constructor() {
    this.ready = this.init();
  }

  private async init(): Promise<void> {
    this.difficulties = await loadDifficulties();
    // Default to the strongest tier (the manifest is ordered weakest→strongest); the player can pick easier.
    await this.setDifficulty(this.difficulties[this.difficulties.length - 1]);
  }

  /** Switch tier: adopt its search knobs and load its checkpoint (cached by url, so sims-only switches are instant). */
  async setDifficulty(d: ChessDifficulty): Promise<void> {
    this.current = d;
    this.sims = d.sims;
    this.cpuct = d.cpuct;
    this.temperature = d.temperature;
    this.netReady = false;
    if (this.netCache.has(d.ckpt)) {
      this.net = this.netCache.get(d.ckpt)!;
    } else {
      this.net = await loadChessNet(d.ckpt);
      this.netCache.set(d.ckpt, this.net);
    }
    this.netMissing = this.net === null;
    this.netReady = true;
  }

  private static startState(): PgChessState {
    return new PgChessState([...START_SQUARES], true, CASTLE_ALL, -1, 0);
  }

  reset(): void {
    this.state = ChessDirector.startState();
    this.lastMove = null;
    this.thinking = false;
  }

  /** A copy of the 64-cell board for rendering (never the engine's internal array). */
  board(): number[] { return [...this.state.squares]; }
  get whiteToMove(): boolean { return this.state.whiteToMove; }

  private legal(): DecodedMove[] {
    return this.state.legalMoveIndices().map(index => {
      const m = PgChessState.decode(index);
      return { index, from: m.from, to: m.to, promo: m.promotion };
    });
  }

  /** Destination squares for a piece on `from` (for move-target highlighting). */
  legalTargets(from: number): number[] {
    return this.legal().filter(m => m.from === from).map(m => m.to);
  }

  // Resolve a human (from,to) to a legal action index; prefer the queen-plane index on a promotion (auto-queen —
  // applyIndex infers Queen), matching the engine. Returns -1 if the move is illegal.
  private resolve(from: number, to: number): number {
    const matches = this.legal().filter(m => m.from === from && m.to === to);
    if (matches.length === 0) return -1;
    return (matches.find(m => m.promo === 0) ?? matches[0]).index;
  }

  /** Apply the human's (White) move if legal. Returns false (no-op) if it's not your turn or the move is illegal. */
  humanMove(from: number, to: number): boolean {
    if (this.thinking || !this.state.whiteToMove) return false;
    const index = this.resolve(from, to);
    if (index < 0) return false;
    this.state = this.state.applyIndex(index);
    this.lastMove = { from, to };
    return true;
  }

  /** Play the AI's move for whichever side is to move (Black's reply in play mode; either side in watch mode).
   *  Yields to the UI first (so the board paints the previous move + a "thinking" cue), then runs the search
   *  synchronously. Falls back to a random legal move if the checkpoint didn't load. No-op at a terminal position. */
  async aiStep(): Promise<void> {
    if (this.result() !== 0) return;
    this.thinking = true;
    await new Promise(resolve => setTimeout(resolve, 30));
    let index: number;
    if (this.net) {
      if (this.temperature > 0) {
        // Sample ∝ visitᵢ^(1/T) over the moves MCTS actually visited — varied but never an unexplored blunder.
        index = ChessDirector.sampleFromPi(PgChessMcts.search(this.net, this.state, this.sims, this.cpuct), this.temperature);
      } else {
        index = PgChessMcts.chooseMove(this.net, this.state, this.sims, this.cpuct);
      }
    } else {
      const moves = this.state.legalMoveIndices();
      index = moves[Math.floor(Math.random() * moves.length)];
    }
    const m = PgChessState.decode(index);
    this.state = this.state.applyIndex(index);
    this.lastMove = { from: m.from, to: m.to };
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

  // Full starting material as [signed piece code, count]: +1..+6 White P,N,B,R,Q,K, −1..−6 Black.
  private static readonly START_COUNTS: ReadonlyArray<readonly [number, number]> = [
    [-1, 8], [-2, 2], [-3, 2], [-4, 2], [-5, 1],   // Black (the AI's pieces you've captured)
    [1, 8], [2, 2], [3, 2], [4, 2], [5, 1],        // White (your pieces the AI has captured)
  ];

  /** Pieces no longer on the board (signed codes), i.e. the captured material — for a captured-pieces tray. Derived
   *  from the board vs the starting set (a promotion can make this off by the promoted pawn, which is rare here). */
  capturedPieces(): number[] {
    const current = new Map<number, number>();
    for (const p of this.state.squares) if (p !== 0) current.set(p, (current.get(p) ?? 0) + 1);
    const out: number[] = [];
    for (const [code, count] of ChessDirector.START_COUNTS) {
      const lost = count - (current.get(code) ?? 0);
      for (let i = 0; i < lost; i++) out.push(code);
    }
    return out;
  }

  /** Terminal result from the side-to-move's view: 0 ongoing, 1 loss (checkmated), 2 draw. */
  result(): number { return this.state.result(); }
  inCheck(): boolean { return this.state.inCheck(this.state.whiteToMove); }
  private hasLegalMoves(): boolean { return this.state.legalMoveIndices().length > 0; }

  /** Human-facing outcome. */
  outcome(): 'ongoing' | 'you-win' | 'ai-wins' | 'stalemate' | 'draw' {
    const r = this.result();
    if (r === 1) return this.state.whiteToMove ? 'ai-wins' : 'you-win'; // the mated side is the one to move
    if (r === 2) return !this.hasLegalMoves() && !this.inCheck() ? 'stalemate' : 'draw';
    return 'ongoing';
  }
}

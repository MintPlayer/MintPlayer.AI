// The classic 2048 engine — Gabriele Cirulli's original mechanics, ported from the
// owner's historic app (C:\Repos\WebGames\Game2048: game_manager.ts / grid.ts / tile.ts).
// This drives the in-browser EXPERIENCE only: traversal-order merging with `mergedFrom`
// double-merge prevention and per-tile previous positions so the DOM renderer can play
// the original slide/pop/appear animations. Merge RESULTS are identical to the server's
// `Board2048`/`applyMove` (both are standard 2048 rules); the API keeps speaking
// exponent boards and server action ids — conversions live at this boundary only.

/** Server action ids (game-2048-api): 0=left 1=down 2=right 3=up. Classic directions: 0=up 1=right 2=down 3=left. */
export const serverActionToClassicDirection = (action: number): number => 3 - action;

export interface Position {
  x: number; // column
  y: number; // row
}

export class ClassicTile {
  previousPosition: Position | null = null;
  mergedFrom: [ClassicTile, ClassicTile] | null = null;
  isNew = false;

  constructor(
    public readonly id: number,
    public x: number,
    public y: number,
    public value: number,
  ) {}

  savePosition(): void {
    this.previousPosition = { x: this.x, y: this.y };
  }

  updatePosition(position: Position): void {
    this.x = position.x;
    this.y = position.y;
  }
}

/** What the template renders: merged sources slide under the popping merged tile. */
export interface RenderTile {
  id: number;
  x: number;
  y: number;
  value: number;
  state: 'none' | 'new' | 'merged';
}

const SIZE = 4;

export class ClassicEngine {
  private grid: (ClassicTile | null)[][] = ClassicEngine.emptyGrid();
  private nextId = 1;
  score = 0;
  won = false;

  private static emptyGrid(): (ClassicTile | null)[][] {
    return Array.from({ length: SIZE }, () => new Array<ClassicTile | null>(SIZE).fill(null));
  }

  /** Loads an exponent board (the playground's wire format) without animation state. */
  static fromExponents(cells: number[]): ClassicEngine {
    const engine = new ClassicEngine();
    for (let i = 0; i < 16; i++) {
      if (cells[i] > 0) {
        const x = i % 4, y = Math.floor(i / 4);
        engine.grid[x][y] = new ClassicTile(engine.nextId++, x, y, 1 << cells[i]);
      }
    }
    return engine;
  }

  toExponents(): number[] {
    const cells = new Array<number>(16).fill(0);
    this.eachCell((x, y, tile) => {
      if (tile) cells[y * 4 + x] = Math.round(Math.log2(tile.value));
    });
    return cells;
  }

  /** Applies a move by SERVER action id; spawns nothing (callers decide random vs scripted). */
  move(action: number): { moved: boolean; gained: number } {
    const direction = serverActionToClassicDirection(action);
    const vector = ClassicEngine.getVector(direction);
    const traversals = ClassicEngine.buildTraversals(vector);
    let moved = false;
    let gained = 0;

    this.prepareTiles();

    for (const x of traversals.x) {
      for (const y of traversals.y) {
        const cell = { x, y };
        const tile = this.cellContent(cell);
        if (!tile) continue;

        const positions = this.findFarthestPosition(cell, vector);
        const next = this.cellContent(positions.next);

        if (next && next.value === tile.value && !next.mergedFrom) {
          // Merge — only once per tile per move (the classic double-merge prevention).
          const merged = new ClassicTile(this.nextId++, positions.next.x, positions.next.y, tile.value * 2);
          merged.mergedFrom = [tile, next];

          this.grid[positions.next.x][positions.next.y] = merged;
          this.grid[tile.x][tile.y] = null;
          tile.updatePosition(positions.next);

          this.score += merged.value;
          gained += merged.value;
          if (merged.value === 2048) this.won = true;
        } else {
          this.moveTile(tile, positions.farthest);
        }

        if (cell.x !== tile.x || cell.y !== tile.y) moved = true;
      }
    }

    return { moved, gained };
  }

  /** Spawns 2 (90%) or 4 (10%) in a random empty cell — manual play only. */
  addRandomTile(): void {
    const cells = this.availableCells();
    if (cells.length === 0) return;
    const cell = cells[Math.floor(Math.random() * cells.length)];
    this.spawn(cell, Math.random() < 0.9 ? 2 : 4);
  }

  /** Spawns the server-scripted tile of an AI playout step (deterministic replays). */
  addSpecificTile(spawnIndex: number, spawnValue: number): void {
    this.spawn({ x: spawnIndex % 4, y: Math.floor(spawnIndex / 4) }, spawnValue);
  }

  movesAvailable(): boolean {
    if (this.availableCells().length > 0) return true;
    for (let x = 0; x < SIZE; x++) {
      for (let y = 0; y < SIZE; y++) {
        const value = this.grid[x][y]!.value;
        if (x < SIZE - 1 && this.grid[x + 1][y]!.value === value) return true;
        if (y < SIZE - 1 && this.grid[x][y + 1]!.value === value) return true;
      }
    }
    return false;
  }

  maxTile(): number {
    let max = 0;
    this.eachCell((_x, _y, tile) => {
      if (tile && tile.value > max) max = tile.value;
    });
    return max;
  }

  /** The animation-aware view: grid tiles plus the merge sources sliding underneath. */
  renderTiles(): RenderTile[] {
    const tiles: RenderTile[] = [];
    this.eachCell((_x, _y, tile) => {
      if (!tile) return;
      if (tile.mergedFrom) {
        for (const source of tile.mergedFrom) {
          tiles.push({ id: source.id, x: tile.x, y: tile.y, value: source.value, state: 'none' });
        }
        tiles.push({ id: tile.id, x: tile.x, y: tile.y, value: tile.value, state: 'merged' });
      } else {
        tiles.push({ id: tile.id, x: tile.x, y: tile.y, value: tile.value, state: tile.isNew ? 'new' : 'none' });
      }
    });
    return tiles.sort((a, b) => a.id - b.id);
  }

  // ------------------------------------------------------------------ internals

  private spawn(cell: Position, value: number): void {
    const tile = new ClassicTile(this.nextId++, cell.x, cell.y, value);
    tile.isNew = true;
    this.grid[cell.x][cell.y] = tile;
  }

  private prepareTiles(): void {
    this.eachCell((_x, _y, tile) => {
      if (tile) {
        tile.mergedFrom = null;
        tile.isNew = false;
        tile.savePosition();
      }
    });
  }

  private moveTile(tile: ClassicTile, cell: Position): void {
    this.grid[tile.x][tile.y] = null;
    this.grid[cell.x][cell.y] = tile;
    tile.updatePosition(cell);
  }

  private eachCell(callback: (x: number, y: number, tile: ClassicTile | null) => void): void {
    for (let x = 0; x < SIZE; x++)
      for (let y = 0; y < SIZE; y++)
        callback(x, y, this.grid[x][y]);
  }

  private availableCells(): Position[] {
    const cells: Position[] = [];
    this.eachCell((x, y, tile) => {
      if (!tile) cells.push({ x, y });
    });
    return cells;
  }

  private cellContent(cell: Position): ClassicTile | null {
    return this.withinBounds(cell) ? this.grid[cell.x][cell.y] : null;
  }

  private withinBounds(cell: Position): boolean {
    return cell.x >= 0 && cell.x < SIZE && cell.y >= 0 && cell.y < SIZE;
  }

  private static getVector(direction: number): Position {
    // 0: up, 1: right, 2: down, 3: left
    return [{ x: 0, y: -1 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: -1, y: 0 }][direction];
  }

  private static buildTraversals(vector: Position): { x: number[]; y: number[] } {
    const traversals = { x: [0, 1, 2, 3], y: [0, 1, 2, 3] };
    // Always traverse from the farthest cell in the chosen direction.
    if (vector.x === 1) traversals.x = traversals.x.reverse();
    if (vector.y === 1) traversals.y = traversals.y.reverse();
    return traversals;
  }

  private findFarthestPosition(cell: Position, vector: Position): { farthest: Position; next: Position } {
    let previous: Position;
    let current = cell;
    do {
      previous = current;
      current = { x: previous.x + vector.x, y: previous.y + vector.y };
    } while (this.withinBounds(current) && !this.cellContent(current));

    return { farthest: previous, next: current };
  }
}

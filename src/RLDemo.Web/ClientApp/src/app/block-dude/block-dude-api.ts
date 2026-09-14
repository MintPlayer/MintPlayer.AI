import { Injectable } from '@angular/core';

/** How the net produced a solution. `greedy` means the policy played it with no search at all. */
export type BlockDudeTier = 'greedy' | 'beam' | 'none';

export interface BlockDudeSolveResponse {
  solved: boolean;
  moves: number[];
  moveCount: number;
  tier: BlockDudeTier;
}

/**
 * Result of asking the server to play a level.
 *
 * A discriminated union rather than a thrown error, matching `CubeApi`: "the net is still loading" and "the net
 * could not solve it" are ordinary outcomes the page should render, not exceptions.
 */
export type BlockDudeSolveResult =
  | { kind: 'done'; value: BlockDudeSolveResponse }
  | { kind: 'loading' }
  | { kind: 'failed'; message: string };

@Injectable({ providedIn: 'root' })
export class BlockDudeApi {
  /**
   * Asks the shipped net to play `grid`.
   *
   * The GRID is sent rather than a level name, so a level added to the pack is playable without a server
   * change — the engine parses the same rows the browser twin does.
   */
  async solve(grid: string[]): Promise<BlockDudeSolveResult> {
    try {
      const response = await fetch('/api/blockdude/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grid }),
      });

      if (response.ok) return { kind: 'done', value: await response.json() as BlockDudeSolveResponse };
      if (response.status === 503) return { kind: 'loading' };
      return { kind: 'failed', message: `The server refused the board (HTTP ${response.status}).` };
    } catch (error) {
      return { kind: 'failed', message: error instanceof Error ? error.message : String(error) };
    }
  }
}

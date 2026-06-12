import { Injectable } from '@angular/core';
import { CubeState } from './cube-renderer';

export interface CubeSolveResponse {
  solution: string[];
  moveCount: number;
  solveTimeMs: number;
  error: string | null;
}

export type CubeSolveResult =
  | { kind: 'solved'; value: CubeSolveResponse }
  | { kind: 'invalid'; error: string };

@Injectable({ providedIn: 'root' })
export class CubeApi {
  /** Kociemba two-phase solve — the algorithmic oracle, always available. */
  async solve(state: CubeState): Promise<CubeSolveResult> {
    const response = await fetch('/api/cube/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state }),
    });
    if (response.ok) {
      return { kind: 'solved', value: await response.json() };
    }
    const body: CubeSolveResponse = await response.json();
    return { kind: 'invalid', error: body.error ?? 'Invalid cube.' };
  }
}

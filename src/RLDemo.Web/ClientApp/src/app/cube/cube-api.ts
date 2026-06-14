import { Injectable } from '@angular/core';
import { CubeState } from './cube-renderer';

export interface CubeSolveResponse {
  solution: string[];
  moveCount: number;
  solveTimeMs: number;
  error: string | null;
}

/** The AI's attempt — `solved` is false when even the lookahead ran out of budget (honest failure). */
export interface CubeSolveAiResponse {
  solved: boolean;
  solution: string[];
  moveCount: number;
  algorithmMoveCount: number;
  /**
   * 'greedy' = reactive policy rollout, 'search' = net-guided lookahead, 'dqn' = legacy fallback,
   * 'davi' = the teacher-free DAVI value net via batch-weighted A* (shortest-move "self-taught AI").
   */
  aiMode: 'greedy' | 'search' | 'dqn' | 'davi';
}

export interface CubeStatusResponse {
  status: 'loading' | 'training' | 'ready' | 'failed';
  trainingStep: number;
  trainingMaxSteps: number;
  lastEvalReturn: number;
  error: string | null;
}

export type CubeSolveResult =
  | { kind: 'solved'; value: CubeSolveResponse }
  | { kind: 'invalid'; error: string };

export type CubeSolveAiResult =
  | { kind: 'done'; value: CubeSolveAiResponse }
  | { kind: 'training'; status: CubeStatusResponse }
  | { kind: 'invalid'; error: string };

@Injectable({ providedIn: 'root' })
export class CubeApi {
  async status(): Promise<CubeStatusResponse> {
    const response = await fetch('/api/cube/status');
    return response.json();
  }

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

  /** Trained-DQN solve — greedy quarter-turn rollout, honest about failure on deep scrambles. */
  async solveAi(state: CubeState): Promise<CubeSolveAiResult> {
    const response = await fetch('/api/cube/solve-ai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state }),
    });
    if (response.ok) {
      return { kind: 'done', value: await response.json() };
    }
    if (response.status === 503) {
      return { kind: 'training', status: await response.json() };
    }
    const body: CubeSolveResponse = await response.json();
    return { kind: 'invalid', error: body.error ?? 'Invalid cube.' };
  }

  /** Teacher-free DAVI value net via batch-weighted A* — the shortest-move "self-taught AI". */
  async solveDavi(state: CubeState): Promise<CubeSolveAiResult> {
    const response = await fetch('/api/cube/solve-davi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state }),
    });
    if (response.ok) {
      return { kind: 'done', value: await response.json() };
    }
    if (response.status === 503) {
      return { kind: 'training', status: await response.json() };
    }
    const body: CubeSolveResponse = await response.json();
    return { kind: 'invalid', error: body.error ?? 'Invalid cube.' };
  }
}

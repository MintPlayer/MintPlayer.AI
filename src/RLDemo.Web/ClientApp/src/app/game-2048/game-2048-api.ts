import { Injectable } from '@angular/core';

export interface PlayoutStep {
  action: number;
  spawnIndex: number;
  spawnValue: number;
  scoreGained: number;
}

export interface SolveResponse2048 {
  initialCells: number[];
  steps: PlayoutStep[];
  finalCells: number[];
  score: number;
  maxTile: number;
  reached2048: boolean;
}

export interface Status2048 {
  status: 'loading' | 'ready' | 'failed';
  error: string | null;
}

export type Solve2048Result =
  | { kind: 'solved'; value: SolveResponse2048 }
  | { kind: 'invalid'; error: string }
  | { kind: 'loading'; status: Status2048 };

@Injectable({ providedIn: 'root' })
export class Game2048Api {
  async status(): Promise<Status2048> {
    const response = await fetch('/api/2048/status');
    return response.json();
  }

  async solve(cells: number[]): Promise<Solve2048Result> {
    const response = await fetch('/api/2048/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cells }),
    });
    if (response.ok) return { kind: 'solved', value: await response.json() };
    if (response.status === 503) return { kind: 'loading', status: await response.json() };
    const body = await response.json();
    return { kind: 'invalid', error: body.error ?? 'Invalid board.' };
  }
}

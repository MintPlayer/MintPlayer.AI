import { Injectable } from '@angular/core';

export interface VehicleDto {
  row: number;
  col: number;
  length: number;
  horizontal: boolean;
}

export interface AnalyzeResponse {
  valid: boolean;
  error: string | null;
  solvable: boolean;
  optimalMoves: number;
}

export interface TrajectoryStep {
  vehicle: number;
  direction: number;
  positions: number[];
}

export interface SolveResponse {
  solved: boolean;
  aiMoves: number;
  optimalMoves: number;
  trajectory: TrajectoryStep[];
  optimalTrajectory: TrajectoryStep[];
  /** 'greedy' = reactive policy, 'search' = policy-guided A*, 'dqn' = legacy fallback. */
  aiMode: 'greedy' | 'search' | 'dqn';
}

export interface StatusResponse {
  status: 'loading' | 'ready' | 'failed';
  error: string | null;
}

export type SolveResult =
  | { kind: 'solved'; value: SolveResponse }
  | { kind: 'invalid'; error: string }
  | { kind: 'loading'; status: StatusResponse };

/** A curated deck level: a named board plus its BFS-optimal move count. */
export interface DeckLevel {
  id: string;
  name: string;
  vehicles: VehicleDto[];
  optimalMoves: number;
}

export interface RushHourDeck {
  version: number;
  levels: DeckLevel[];
}

export type SaveLevelResult =
  | { kind: 'saved'; level: DeckLevel }
  | { kind: 'error'; error: string }
  | { kind: 'unavailable' }; // authoring endpoint is Development-only

@Injectable({ providedIn: 'root' })
export class RushHourApi {
  async status(): Promise<StatusResponse> {
    const response = await fetch('/api/rushhour/status');
    return response.json();
  }

  async analyze(vehicles: VehicleDto[]): Promise<AnalyzeResponse> {
    const response = await fetch('/api/rushhour/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vehicles }),
    });
    return response.json();
  }

  async solve(vehicles: VehicleDto[]): Promise<SolveResult> {
    const response = await fetch('/api/rushhour/solve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vehicles }),
    });
    if (response.ok) {
      return { kind: 'solved', value: await response.json() };
    }
    if (response.status === 503) {
      return { kind: 'loading', status: await response.json() };
    }
    const body: AnalyzeResponse = await response.json();
    return { kind: 'invalid', error: body.error ?? 'Invalid board.' };
  }

  /** The curated level deck (served everywhere; read-only in production). */
  async getDeck(): Promise<DeckLevel[]> {
    const response = await fetch('/api/rushhour/deck', { cache: 'no-store' });
    if (!response.ok) return [];
    const deck: RushHourDeck = await response.json();
    return deck.levels ?? [];
  }

  /** Save (insert or, with id, update) a level — Development only; production returns 'unavailable'. */
  async saveLevel(name: string, vehicles: VehicleDto[], id?: string): Promise<SaveLevelResult> {
    const response = await fetch('/api/rushhour/deck', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, name, vehicles }),
    });
    if (response.ok) return { kind: 'saved', level: await response.json() };
    if (response.status === 404) return { kind: 'unavailable' };
    const body = await response.json().catch(() => ({ error: 'Save failed.' }));
    return { kind: 'error', error: body.error ?? 'Save failed.' };
  }

  /** Remove a level — Development only. */
  async deleteLevel(id: string): Promise<boolean> {
    const response = await fetch(`/api/rushhour/deck/${id}`, { method: 'DELETE' });
    return response.ok;
  }
}

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
}

export interface StatusResponse {
  status: 'loading' | 'training' | 'ready' | 'failed';
  trainingStep: number;
  trainingMaxSteps: number;
  lastEvalReturn: number;
  error: string | null;
}

export type SolveResult =
  | { kind: 'solved'; value: SolveResponse }
  | { kind: 'invalid'; error: string }
  | { kind: 'training'; status: StatusResponse };

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
      return { kind: 'training', status: await response.json() };
    }
    const body: AnalyzeResponse = await response.json();
    return { kind: 'invalid', error: body.error ?? 'Invalid board.' };
  }
}

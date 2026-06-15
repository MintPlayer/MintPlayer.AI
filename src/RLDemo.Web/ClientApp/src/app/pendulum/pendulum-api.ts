import { Injectable } from '@angular/core';

/** One streamed frame of an AI Pendulum episode (PRD §7.1 principle B). `torque` is the continuous action applied. */
export interface PendulumFrame {
  cosTheta: number;
  sinTheta: number;
  angularVelocity: number;
  torque: number;
  reward: number;
  done: boolean;
}

export interface PendulumStatus {
  status: 'loading' | 'training' | 'ready' | 'failed';
  trainingStep: number;
  trainingMaxSteps: number;
  lastEvalReturn: number;
  error: string | null;
}

@Injectable({ providedIn: 'root' })
export class PendulumApi {
  async status(): Promise<PendulumStatus> {
    const response = await fetch('/api/pendulum/status');
    return response.json();
  }

  /** Server-authoritative live stream — the backend drives the pendulum and pushes frames; the caller renders. */
  connectLive(onFrame: (frame: PendulumFrame) => void, onClose: () => void): WebSocket {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${proto}://${location.host}/api/pendulum/live`);
    socket.onmessage = e => onFrame(JSON.parse(e.data) as PendulumFrame);
    socket.onclose = onClose;
    socket.onerror = onClose;
    return socket;
  }
}

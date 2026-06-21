import { Injectable } from '@angular/core';

/** One streamed frame of an AI MountainCar episode (PRD §7.1 principle B). `action = -1` marks a reset. */
export interface MountainCarFrame {
  position: number;
  velocity: number;
  action: number;
  reward: number;
  done: boolean;
}

export interface MountainCarStatus {
  status: 'loading' | 'ready' | 'failed';
  error: string | null;
}

@Injectable({ providedIn: 'root' })
export class MountainCarApi {
  async status(): Promise<MountainCarStatus> {
    const response = await fetch('/api/mountaincar/status');
    return response.json();
  }

  /** Server-authoritative live stream — the backend drives the car and pushes frames; the caller renders. */
  connectLive(onFrame: (frame: MountainCarFrame) => void, onClose: () => void): WebSocket {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${proto}://${location.host}/api/mountaincar/live`);
    socket.onmessage = e => onFrame(JSON.parse(e.data) as MountainCarFrame);
    socket.onclose = onClose;
    socket.onerror = onClose;
    return socket;
  }
}

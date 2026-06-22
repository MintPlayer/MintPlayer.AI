import { Injectable } from '@angular/core';

/** One streamed frame of an AI game (PRD §7.1 principle B). `action = -1` marks a freshly reset start. */
export interface SnakeFrame {
  body: number[]; // head first, cell indices
  food: number;
  action: number;
  reward: number;
  done: boolean;
  foodEaten: number;
  length: number;
}

export interface SnakeStatus {
  status: 'loading' | 'ready' | 'failed';
  error: string | null;
}

@Injectable({ providedIn: 'root' })
export class SnakeApi {
  async status(): Promise<SnakeStatus> {
    const response = await fetch('/api/snake/status');
    return response.json();
  }

  /**
   * Opens the server-authoritative live stream — the backend drives the game and pushes one frame per tick;
   * the caller just renders. Returns the socket so the caller can close it on teardown.
   */
  connectLive(onFrame: (frame: SnakeFrame) => void, onClose: () => void): WebSocket {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${proto}://${location.host}/api/snake/live`);
    socket.onmessage = e => onFrame(JSON.parse(e.data) as SnakeFrame);
    socket.onclose = onClose;
    socket.onerror = onClose;
    return socket;
  }
}

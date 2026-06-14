import { Component, DestroyRef, ElementRef, afterNextRender, computed, inject, signal, viewChild } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CubeApi, CubeSolveResponse, CubeStatusResponse } from './cube-api';
import { RubiksCube } from './cube-renderer';

const FACES = ['U', 'D', 'L', 'R', 'F', 'B'] as const;
const OPPOSITE: Record<string, string> = { U: 'D', D: 'U', R: 'L', L: 'R', F: 'B', B: 'F' };
const INVERSE: Record<string, string> = {
  U: "U'", "U'": 'U', U2: 'U2', D: "D'", "D'": 'D', D2: 'D2',
  R: "R'", "R'": 'R', R2: 'R2', L: "L'", "L'": 'L', L2: 'L2',
  F: "F'", "F'": 'F', F2: 'F2', B: "B'", "B'": 'B', B2: 'B2',
};

/** All 18 face-move buttons, grouped per face for the 3-column grid. */
const MOVE_BUTTONS = FACES.flatMap(f => [f, `${f}'`, `${f}2`]);

/** A move list armed for playback, with a caption saying who produced it. */
interface ArmedSolution {
  moves: string[];
  info: string;
}

@Component({
  selector: 'app-cube',
  templateUrl: './cube.html',
  styleUrl: './cube.scss',
  host: {
    '(window:keydown)': 'onKey($event)',
    '(window:resize)': 'onResize()',
  },
})
export class Cube {
  private readonly api = inject(CubeApi);
  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('cubeCanvas');

  protected readonly moveButtons = MOVE_BUTTONS;

  protected readonly status = signal('Ready');
  protected readonly animating = signal(false);
  protected readonly busy = signal(false);
  protected readonly moveHistory = signal<string[]>([]);
  protected readonly animationSpeed = signal(300);

  // --- solution playback, Kociemba or AI (index = last executed move, -1 = at start) ---
  protected readonly solution = signal<ArmedSolution | null>(null);
  protected readonly solutionIndex = signal(-1);
  protected readonly playing = signal(false);

  // --- model status (the AI button needs a trained DQN) ---
  protected readonly modelStatus = signal<CubeStatusResponse | null>(null);

  protected readonly locked = computed(() => this.animating() || this.busy());
  protected readonly atSolutionEnd = computed(() => {
    const s = this.solution();
    return s === null || this.solutionIndex() >= s.moves.length - 1;
  });

  // --- three.js scene (browser only) ---
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private cube!: RubiksCube;
  private frameHandle = 0;

  constructor() {
    const replayId = inject(ActivatedRoute).snapshot.queryParamMap.get('replay');
    this.pollStatus();
    afterNextRender(() => {
      this.initScene();
      this.animate();
      if (replayId) void this.loadGalleryEntry(replayId);
    });
    inject(DestroyRef).onDestroy(() => {
      cancelAnimationFrame(this.frameHandle);
      this.renderer?.dispose();
    });
  }

  private initScene(): void {
    const canvas = this.canvasRef().nativeElement;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x141823);

    this.camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
    this.camera.position.set(5, 5, 7);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 20;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 0.8);
    key.position.set(10, 10, 10);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4);
    fill.position.set(-10, -10, -10);
    this.scene.add(fill);

    this.cube = new RubiksCube();
    this.scene.add(this.cube.group);
  }

  private animate(): void {
    this.frameHandle = requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  protected onResize(): void {
    if (!this.renderer) return;
    const canvas = this.canvasRef().nativeElement;
    this.camera.aspect = canvas.clientWidth / canvas.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  }

  // ------------------------------------------------------------------ moves

  protected async executeMove(move: string): Promise<void> {
    if (this.locked() || !this.cube) return;
    this.animating.set(true);
    this.status.set('Rotating…');

    await this.cube.rotate(move, this.animationSpeed());

    this.moveHistory.update(h => [...h, move]);
    this.clearSolution(); // manual moves invalidate a computed solution
    this.animating.set(false);
    this.status.set('Ready');
  }

  protected onKey(event: KeyboardEvent): void {
    if (this.locked()) return;
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

    const key = event.key.toUpperCase();
    if ((FACES as readonly string[]).includes(key)) {
      event.preventDefault();
      void this.executeMove(event.shiftKey ? `${key}'` : key);
    }
  }

  protected setSpeed(value: string): void {
    this.animationSpeed.set(+value);
  }

  // ------------------------------------------------------------------ scramble / reset

  /** Full scramble (~20 moves incl. half turns) — Kociemba territory; the AI will mostly fail here, honestly. */
  protected scramble(): Promise<void> {
    return this.runScramble(this.randomMoves(20, ['', "'", '2']), 'Scrambled — ready to solve!');
  }

  /** Shallow quarter-turn scramble (≤ 6) — the trained agent's home turf (PRD §11). */
  protected easyScramble(): Promise<void> {
    return this.runScramble(this.randomMoves(6, ['', "'"]), 'Easy scramble — even the AI should manage this one.');
  }

  private randomMoves(length: number, modifiers: string[]): string[] {
    const moves: string[] = [];
    let last: string | null = null;
    let secondLast: string | null = null;
    for (let i = 0; i < length; i++) {
      let faces = FACES.filter(f => f !== last);
      if (last && secondLast && OPPOSITE[last] === secondLast) {
        faces = faces.filter(f => f !== OPPOSITE[last!]);
      }
      const face = faces[Math.floor(Math.random() * faces.length)];
      moves.push(face + modifiers[Math.floor(Math.random() * modifiers.length)]);
      secondLast = last;
      last = face;
    }
    return moves;
  }

  private async runScramble(moves: string[], doneStatus: string): Promise<void> {
    if (this.locked() || !this.cube) return;
    this.animating.set(true);
    this.status.set('Scrambling…');
    this.clearSolution();

    for (const move of moves) {
      await this.cube.rotate(move, 80);
      this.moveHistory.update(h => [...h, move]);
    }

    this.animating.set(false);
    this.status.set(doneStatus);
  }

  protected reset(): void {
    if (this.locked() || !this.cube) return;
    this.cube.reset();
    this.moveHistory.set([]);
    this.clearSolution();
    this.status.set('Reset — ready');
  }

  // ------------------------------------------------------------------ solve (algorithm)

  protected async solve(): Promise<void> {
    if (this.locked() || !this.cube) return;
    this.busy.set(true);
    this.status.set('Solving (Kociemba two-phase)…');

    try {
      const result = await this.api.solve(this.cube.getState());
      if (result.kind === 'invalid') {
        this.status.set(result.error);
        return;
      }
      if (result.value.solution.length === 0) {
        this.status.set('Already solved!');
        this.clearSolution();
        return;
      }
      this.armSolution(result.value.solution, `${result.value.moveCount} moves in ${result.value.solveTimeMs} ms (Kociemba)`);
      this.status.set(`Solution found: ${result.value.moveCount} moves (${result.value.solveTimeMs} ms)`);
    } catch {
      this.status.set('The solver is unreachable — is the backend running?');
    } finally {
      this.busy.set(false);
    }
  }

  // ------------------------------------------------------------------ solve (AI)

  /** The trained DQN's greedy attempt — armed for playback when it solves, reported honestly when it doesn't. */
  protected async solveAi(): Promise<void> {
    if (this.locked() || !this.cube) return;
    this.busy.set(true);
    this.status.set('Asking the trained AI…');

    try {
      const result = await this.api.solveAi(this.cube.getState());
      switch (result.kind) {
        case 'done': {
          const v = result.value;
          const how = v.aiMode === 'search' ? 'AI (with lookahead)' : 'AI';
          if (v.solved) {
            this.armSolution(v.solution, `${how}: ${v.moveCount} quarter-turns (Kociemba reference: ${v.algorithmMoveCount})`);
            this.status.set(`The ${how} solved it in ${v.moveCount} moves!`);
          } else {
            this.clearSolution();
            this.status.set(`The AI gave up, even with lookahead (Kociemba needs ${v.algorithmMoveCount} moves) — ` +
              'it is trained on shallow scrambles; try an easy scramble, or the algorithm.');
          }
          break;
        }
        case 'training':
          this.modelStatus.set(result.status);
          this.status.set('The AI model is still training — try again in a moment.');
          this.pollStatus();
          break;
        case 'invalid':
          this.status.set(result.error);
          break;
      }
    } catch {
      this.status.set('The solver is unreachable — is the backend running?');
    } finally {
      this.busy.set(false);
    }
  }

  /** The teacher-free DAVI value net via batch-weighted A* — shortest-move search, honest on failure. */
  protected async solveDavi(): Promise<void> {
    if (this.locked() || !this.cube) return;
    this.busy.set(true);
    this.status.set('Asking the self-taught AI (shortest-move search)…');

    try {
      const result = await this.api.solveDavi(this.cube.getState());
      switch (result.kind) {
        case 'done': {
          const v = result.value;
          if (v.solved) {
            const beats = v.moveCount <= v.algorithmMoveCount;
            this.armSolution(v.solution,
              `Self-taught AI: ${v.moveCount} quarter-turns (Kociemba QTM: ${v.algorithmMoveCount})` +
              (beats ? ' — shorter than Kociemba!' : ''));
            this.status.set(`The self-taught AI solved it in ${v.moveCount} quarter-turns!`);
          } else {
            this.clearSolution();
            this.status.set(`The self-taught AI ran out of search budget (Kociemba needs ${v.algorithmMoveCount} QTM) — ` +
              'it solves shortest-move up to ~15 quarter-turns deep; try a shallower scramble, or the algorithm.');
          }
          break;
        }
        case 'training':
          this.modelStatus.set(result.status);
          this.status.set('The self-taught AI net is not loaded yet — try again in a moment.');
          break;
        case 'invalid':
          this.status.set(result.error);
          break;
      }
    } catch {
      this.status.set('The solver is unreachable — is the backend running?');
    } finally {
      this.busy.set(false);
    }
  }

  private armSolution(moves: string[], info: string): void {
    this.solution.set({ moves, info });
    this.solutionIndex.set(-1);
  }

  // ------------------------------------------------------------------ model status

  private pollStatus(): void {
    void (async () => {
      try {
        const status = await this.api.status();
        this.modelStatus.set(status);
        if (status.status === 'loading' || status.status === 'training') {
          setTimeout(() => this.pollStatus(), 2000);
        }
      } catch {
        // Backend unreachable: leave the status unknown; the buttons stay usable.
      }
    })();
  }

  // ------------------------------------------------------------------ solution playback

  protected async stepNext(): Promise<void> {
    const s = this.solution();
    if (!s || this.locked() || this.atSolutionEnd()) return;

    const index = this.solutionIndex() + 1;
    const move = s.moves[index];
    this.animating.set(true);
    this.status.set(`Step ${index + 1}/${s.moves.length}: ${move}`);

    await this.cube.rotate(move, this.animationSpeed());

    this.solutionIndex.set(index);
    this.animating.set(false);
    if (index === s.moves.length - 1) {
      this.status.set('Solved!');
      this.moveHistory.set([]);
    }
  }

  protected async stepPrev(): Promise<void> {
    const s = this.solution();
    if (!s || this.locked() || this.solutionIndex() < 0) return;

    const move = s.moves[this.solutionIndex()];
    this.animating.set(true);
    this.status.set(`Undoing step ${this.solutionIndex() + 1}: ${move}`);

    await this.cube.rotate(INVERSE[move] ?? move, this.animationSpeed());

    this.solutionIndex.update(i => i - 1);
    this.animating.set(false);
    this.status.set(this.solutionIndex() < 0 ? 'Back to start' : `At step ${this.solutionIndex() + 1}/${s.moves.length}`);
  }

  protected async togglePlay(): Promise<void> {
    if (this.playing()) {
      this.playing.set(false);
      return;
    }
    if (this.atSolutionEnd()) return;

    this.playing.set(true);
    while (this.playing() && !this.atSolutionEnd()) {
      await this.stepNext();
      if (!this.atSolutionEnd()) await new Promise<void>(r => setTimeout(r, 50));
    }
    this.playing.set(false);
  }

  private clearSolution(): void {
    this.solution.set(null);
    this.solutionIndex.set(-1);
    this.playing.set(false);
  }

  protected moveClass(index: number): string {
    if (index < this.solutionIndex()) return 'move-item move-done';
    if (index === this.solutionIndex()) return 'move-item move-current';
    return 'move-item move-pending';
  }

  // ------------------------------------------------------------------ gallery replay

  /**
   * Rebuilds the submitted cube visually by applying the inverted solution to a solved
   * cube (state = solution⁻¹ ∘ solved, since the solution solves exactly that state),
   * then arms the solution for playback.
   */
  private async loadGalleryEntry(id: string): Promise<void> {
    const response = await fetch(`/api/gallery/${id}`);
    if (!response.ok) return;
    const entry = await response.json();
    if (entry.game !== 'cube') return;

    const value: CubeSolveResponse = entry.response;
    this.animating.set(true);
    this.status.set('Rebuilding the submitted cube…');
    for (const move of [...value.solution].reverse()) {
      await this.cube.rotate(INVERSE[move] ?? move, 30);
    }
    this.animating.set(false);
    this.armSolution(value.solution, entry.summary ?? `${value.moveCount} moves`);
    this.status.set(`Replaying: ${value.moveCount} moves`);
  }
}

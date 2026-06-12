// The 3D cube renderer + client-side sticker-state tracker, ported verbatim from the
// owner's Rubiksolver app (Scripts/rubiksCube.ts). Face arrays are indexed 0-8 row-major
// looking at the face, matching the backend DTO and the Kociemba facelet convention.
import * as THREE from 'three';

// Standard Rubik's Cube colors
const COLORS: Record<string, number> = {
    white: 0xffffff,   // U (Up)
    yellow: 0xffff00,  // D (Down)
    green: 0x00ff00,   // F (Front)
    blue: 0x0000ff,    // B (Back)
    red: 0xff0000,     // R (Right)
    orange: 0xff8c00,  // L (Left)
    black: 0x111111    // Inside faces
};

type FaceKey = 'U' | 'D' | 'F' | 'B' | 'R' | 'L';
type StickerColor = 'W' | 'Y' | 'G' | 'B' | 'R' | 'O';

export interface CubeState {
    U: StickerColor[];
    D: StickerColor[];
    F: StickerColor[];
    B: StickerColor[];
    R: StickerColor[];
    L: StickerColor[];
}

export class RubiksCube {
    public group: THREE.Group;
    private cubies: THREE.Mesh[];
    private cubeSize: number;
    private gap: number;
    private state: CubeState;

    constructor() {
        this.group = new THREE.Group();
        this.cubies = [];
        this.cubeSize = 1;
        this.gap = 0.05;

        // Track the cube state for solving
        this.state = this.createInitialState();

        this.createCube();
    }

    private createInitialState(): CubeState {
        // Each face has 9 stickers, indexed 0-8 (top-left to bottom-right)
        // U = white, D = yellow, F = green, B = blue, R = red, L = orange
        return {
            U: ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
            D: ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y'],
            F: ['G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'],
            B: ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
            R: ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R'],
            L: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        };
    }

    private createCubie(x: number, y: number, z: number): THREE.Mesh {
        const geometry = new THREE.BoxGeometry(
            this.cubeSize - this.gap,
            this.cubeSize - this.gap,
            this.cubeSize - this.gap
        );

        // Determine colors for each face based on position
        const materials = [
            new THREE.MeshPhongMaterial({ color: x === 1 ? COLORS['red'] : COLORS['black'] }),    // Right (+X)
            new THREE.MeshPhongMaterial({ color: x === -1 ? COLORS['orange'] : COLORS['black'] }), // Left (-X)
            new THREE.MeshPhongMaterial({ color: y === 1 ? COLORS['white'] : COLORS['black'] }),   // Up (+Y)
            new THREE.MeshPhongMaterial({ color: y === -1 ? COLORS['yellow'] : COLORS['black'] }), // Down (-Y)
            new THREE.MeshPhongMaterial({ color: z === 1 ? COLORS['green'] : COLORS['black'] }),   // Front (+Z)
            new THREE.MeshPhongMaterial({ color: z === -1 ? COLORS['blue'] : COLORS['black'] })    // Back (-Z)
        ];

        const cubie = new THREE.Mesh(geometry, materials);
        cubie.position.set(x, y, z);

        // Store initial position for reference
        cubie.userData['initialPosition'] = new THREE.Vector3(x, y, z);

        return cubie;
    }

    private createCube(): void {
        // Clear existing cubies
        this.cubies.forEach(cubie => this.group.remove(cubie));
        this.cubies = [];

        // Create 27 cubies (3x3x3)
        for (let x = -1; x <= 1; x++) {
            for (let y = -1; y <= 1; y++) {
                for (let z = -1; z <= 1; z++) {
                    const cubie = this.createCubie(x, y, z);
                    this.cubies.push(cubie);
                    this.group.add(cubie);
                }
            }
        }
    }

    private getCubiesForMove(move: string): THREE.Mesh[] {
        const face = move.replace("'", "").replace("2", "");
        const cubies: THREE.Mesh[] = [];

        this.cubies.forEach(cubie => {
            const pos = cubie.position;
            const threshold = 0.5;

            switch (face) {
                case 'U':
                    if (pos.y > threshold) cubies.push(cubie);
                    break;
                case 'D':
                    if (pos.y < -threshold) cubies.push(cubie);
                    break;
                case 'R':
                    if (pos.x > threshold) cubies.push(cubie);
                    break;
                case 'L':
                    if (pos.x < -threshold) cubies.push(cubie);
                    break;
                case 'F':
                    if (pos.z > threshold) cubies.push(cubie);
                    break;
                case 'B':
                    if (pos.z < -threshold) cubies.push(cubie);
                    break;
            }
        });

        return cubies;
    }

    private getRotationAxis(move: string): THREE.Vector3 {
        const face = move.replace("'", "").replace("2", "");
        // Standard Rubik's cube convention:
        // Clockwise is defined when looking AT the face
        switch (face) {
            case 'U': return new THREE.Vector3(0, 1, 0);   // +Y axis
            case 'D': return new THREE.Vector3(0, -1, 0);  // -Y axis
            case 'R': return new THREE.Vector3(1, 0, 0);   // +X axis
            case 'L': return new THREE.Vector3(-1, 0, 0);  // -X axis
            case 'F': return new THREE.Vector3(0, 0, 1);   // +Z axis
            case 'B': return new THREE.Vector3(0, 0, -1);  // -Z axis
            default: return new THREE.Vector3(0, 1, 0);
        }
    }

    private getRotationAngle(move: string): number {
        const isCounterClockwise = move.includes("'");
        const isDouble = move.includes("2");

        let angle = Math.PI / 2; // 90 degrees

        if (isDouble) angle = Math.PI; // 180 degrees
        if (isCounterClockwise) angle = -angle;

        // Negate to match standard convention (CW = negative rotation around axis)
        return -angle;
    }

    public async rotate(move: string, duration: number = 300): Promise<void> {
        return new Promise(resolve => {
            const cubies = this.getCubiesForMove(move);
            const axis = this.getRotationAxis(move);
            const totalAngle = this.getRotationAngle(move);

            // Create a temporary pivot point
            const pivot = new THREE.Object3D();
            this.group.add(pivot);

            // Attach cubies to pivot
            cubies.forEach(cubie => {
                pivot.attach(cubie);
            });

            // Animation
            const startTime = performance.now();
            const startRotation = pivot.rotation.clone();

            const animate = (currentTime: number): void => {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);

                // Easing function (ease-out)
                const easeProgress = 1 - Math.pow(1 - progress, 3);

                // Apply rotation
                if (axis.x !== 0) {
                    pivot.rotation.x = startRotation.x + totalAngle * easeProgress * axis.x;
                } else if (axis.y !== 0) {
                    pivot.rotation.y = startRotation.y + totalAngle * easeProgress * axis.y;
                } else if (axis.z !== 0) {
                    pivot.rotation.z = startRotation.z + totalAngle * easeProgress * axis.z;
                }

                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    // Animation complete - detach cubies and round positions
                    cubies.forEach(cubie => {
                        this.group.attach(cubie);

                        // Round position to nearest integer to prevent floating point drift
                        cubie.position.x = Math.round(cubie.position.x);
                        cubie.position.y = Math.round(cubie.position.y);
                        cubie.position.z = Math.round(cubie.position.z);
                    });

                    // Remove pivot
                    this.group.remove(pivot);

                    // Update state
                    this.updateState(move);

                    resolve();
                }
            };

            requestAnimationFrame(animate);
        });
    }

    private updateState(move: string): void {
        const face = move.replace("'", "").replace("2", "") as FaceKey;
        const counterClockwise = move.includes("'");
        const double = move.includes("2");

        const rotations = double ? 2 : 1;

        for (let i = 0; i < rotations; i++) {
            this.rotateFace(face, counterClockwise);
        }
    }

    private rotateFace(face: FaceKey, counterClockwise: boolean): void {
        const state = this.state;

        // Rotate the face itself
        // Standard convention: CW when looking at the face
        const f = state[face].slice();
        if (counterClockwise) {
            state[face] = [f[2], f[5], f[8], f[1], f[4], f[7], f[0], f[3], f[6]];
        } else {
            state[face] = [f[6], f[3], f[0], f[7], f[4], f[1], f[8], f[5], f[2]];
        }

        // Adjacent sticker cycles
        // Standard Rubik's cube convention for clockwise rotation
        let temp: StickerColor[];
        switch (face) {
            case 'U':
                // U CW (looking at U): F -> R -> B -> L -> F (pieces move this way)
                // So stickers: F <- R, R <- B, B <- L, L <- F
                if (counterClockwise) {
                    temp = [state.F[0], state.F[1], state.F[2]];
                    [state.F[0], state.F[1], state.F[2]] = [state.L[0], state.L[1], state.L[2]];
                    [state.L[0], state.L[1], state.L[2]] = [state.B[0], state.B[1], state.B[2]];
                    [state.B[0], state.B[1], state.B[2]] = [state.R[0], state.R[1], state.R[2]];
                    [state.R[0], state.R[1], state.R[2]] = temp;
                } else {
                    temp = [state.F[0], state.F[1], state.F[2]];
                    [state.F[0], state.F[1], state.F[2]] = [state.R[0], state.R[1], state.R[2]];
                    [state.R[0], state.R[1], state.R[2]] = [state.B[0], state.B[1], state.B[2]];
                    [state.B[0], state.B[1], state.B[2]] = [state.L[0], state.L[1], state.L[2]];
                    [state.L[0], state.L[1], state.L[2]] = temp;
                }
                break;

            case 'D':
                // D CW (looking at D): F -> L -> B -> R -> F (pieces move this way)
                // So stickers: F <- L, L <- B, B <- R, R <- F
                if (counterClockwise) {
                    temp = [state.F[6], state.F[7], state.F[8]];
                    [state.F[6], state.F[7], state.F[8]] = [state.R[6], state.R[7], state.R[8]];
                    [state.R[6], state.R[7], state.R[8]] = [state.B[6], state.B[7], state.B[8]];
                    [state.B[6], state.B[7], state.B[8]] = [state.L[6], state.L[7], state.L[8]];
                    [state.L[6], state.L[7], state.L[8]] = temp;
                } else {
                    temp = [state.F[6], state.F[7], state.F[8]];
                    [state.F[6], state.F[7], state.F[8]] = [state.L[6], state.L[7], state.L[8]];
                    [state.L[6], state.L[7], state.L[8]] = [state.B[6], state.B[7], state.B[8]];
                    [state.B[6], state.B[7], state.B[8]] = [state.R[6], state.R[7], state.R[8]];
                    [state.R[6], state.R[7], state.R[8]] = temp;
                }
                break;

            case 'F':
                // F CW: U[6,7,8] -> R[0,3,6] -> D[2,1,0] -> L[8,5,2] -> U[6,7,8]
                if (counterClockwise) {
                    temp = [state.U[6], state.U[7], state.U[8]];
                    [state.U[6], state.U[7], state.U[8]] = [state.R[0], state.R[3], state.R[6]];
                    [state.R[0], state.R[3], state.R[6]] = [state.D[2], state.D[1], state.D[0]];
                    [state.D[2], state.D[1], state.D[0]] = [state.L[8], state.L[5], state.L[2]];
                    [state.L[8], state.L[5], state.L[2]] = temp;
                } else {
                    temp = [state.U[6], state.U[7], state.U[8]];
                    [state.U[6], state.U[7], state.U[8]] = [state.L[8], state.L[5], state.L[2]];
                    [state.L[8], state.L[5], state.L[2]] = [state.D[2], state.D[1], state.D[0]];
                    [state.D[2], state.D[1], state.D[0]] = [state.R[0], state.R[3], state.R[6]];
                    [state.R[0], state.R[3], state.R[6]] = temp;
                }
                break;

            case 'B':
                // B CW: U[2,1,0] -> L[0,3,6] -> D[6,7,8] -> R[8,5,2] -> U[2,1,0]
                if (counterClockwise) {
                    temp = [state.U[2], state.U[1], state.U[0]];
                    [state.U[2], state.U[1], state.U[0]] = [state.L[0], state.L[3], state.L[6]];
                    [state.L[0], state.L[3], state.L[6]] = [state.D[6], state.D[7], state.D[8]];
                    [state.D[6], state.D[7], state.D[8]] = [state.R[8], state.R[5], state.R[2]];
                    [state.R[8], state.R[5], state.R[2]] = temp;
                } else {
                    temp = [state.U[2], state.U[1], state.U[0]];
                    [state.U[2], state.U[1], state.U[0]] = [state.R[8], state.R[5], state.R[2]];
                    [state.R[8], state.R[5], state.R[2]] = [state.D[6], state.D[7], state.D[8]];
                    [state.D[6], state.D[7], state.D[8]] = [state.L[0], state.L[3], state.L[6]];
                    [state.L[0], state.L[3], state.L[6]] = temp;
                }
                break;

            case 'R':
                // R CW: U[2,5,8] -> F[2,5,8] -> D[2,5,8] -> B[6,3,0] -> U[2,5,8]
                if (counterClockwise) {
                    temp = [state.U[2], state.U[5], state.U[8]];
                    [state.U[2], state.U[5], state.U[8]] = [state.B[6], state.B[3], state.B[0]];
                    [state.B[6], state.B[3], state.B[0]] = [state.D[2], state.D[5], state.D[8]];
                    [state.D[2], state.D[5], state.D[8]] = [state.F[2], state.F[5], state.F[8]];
                    [state.F[2], state.F[5], state.F[8]] = temp;
                } else {
                    temp = [state.U[2], state.U[5], state.U[8]];
                    [state.U[2], state.U[5], state.U[8]] = [state.F[2], state.F[5], state.F[8]];
                    [state.F[2], state.F[5], state.F[8]] = [state.D[2], state.D[5], state.D[8]];
                    [state.D[2], state.D[5], state.D[8]] = [state.B[6], state.B[3], state.B[0]];
                    [state.B[6], state.B[3], state.B[0]] = temp;
                }
                break;

            case 'L':
                // L CW: U[0,3,6] -> B[8,5,2] -> D[0,3,6] -> F[0,3,6] -> U[0,3,6]
                if (counterClockwise) {
                    temp = [state.U[0], state.U[3], state.U[6]];
                    [state.U[0], state.U[3], state.U[6]] = [state.F[0], state.F[3], state.F[6]];
                    [state.F[0], state.F[3], state.F[6]] = [state.D[0], state.D[3], state.D[6]];
                    [state.D[0], state.D[3], state.D[6]] = [state.B[8], state.B[5], state.B[2]];
                    [state.B[8], state.B[5], state.B[2]] = temp;
                } else {
                    temp = [state.U[0], state.U[3], state.U[6]];
                    [state.U[0], state.U[3], state.U[6]] = [state.B[8], state.B[5], state.B[2]];
                    [state.B[8], state.B[5], state.B[2]] = [state.D[0], state.D[3], state.D[6]];
                    [state.D[0], state.D[3], state.D[6]] = [state.F[0], state.F[3], state.F[6]];
                    [state.F[0], state.F[3], state.F[6]] = temp;
                }
                break;
        }
    }

    public isSolved(): boolean {
        for (const face of Object.keys(this.state) as FaceKey[]) {
            const color = this.state[face][4]; // Center piece
            for (const sticker of this.state[face]) {
                if (sticker !== color) return false;
            }
        }
        return true;
    }

    public reset(): void {
        this.state = this.createInitialState();
        this.createCube();
    }

    public getState(): CubeState {
        return JSON.parse(JSON.stringify(this.state));
    }

    public setState(state: CubeState): void {
        this.state = JSON.parse(JSON.stringify(state));
    }
}

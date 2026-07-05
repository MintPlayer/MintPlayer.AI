export type Option<T> = { tag: "Some"; value: T } | { tag: "None" };
export class PgMlpNet {
    sizes: number[];
    wFlat: number[];
    bFlat: number[];
    constructor(sizes: number[], wFlat: number[], bFlat: number[]) {
        this.sizes = sizes;
        this.wFlat = wFlat;
        this.bFlat = bFlat;
    }
    forward(input: number[]): number[] {
        let x = input;
        let wOff = 0;
        let bOff = 0;
        const layers = (this.sizes.length - 1 | 0);
        for (let l = 0; l < layers; l++) {
            const inDim = this.sizes[l];
            const outDim = this.sizes[(l + 1 | 0)];
            x = PgMlpNet.linear(x, this.wFlat, wOff, this.bFlat, bOff, inDim, outDim);
            wOff += Math.imul(inDim, outDim);
            bOff += outDim;
            if (l < (layers - 1 | 0)) {
                x = PgMlpNet.tanhv(x);
            }
        }
        return x;
    }
    static linear(x: number[], w: number[], wOff: number, b: number[], bOff: number, inDim: number, outDim: number): number[] {
        let out = [];
        for (let o = 0; o < outDim; o++) {
            let s = b[(bOff + o | 0)];
            for (let i = 0; i < inDim; i++) {
                s += x[i] * w[((wOff + Math.imul(i, outDim) | 0) + o | 0)];
            }
            out.push(s);
        }
        return out;
    }
    static tanhv(x: number[]): number[] {
        let out = [];
        for (const v of x) {
            out.push(Math.tanh(v));
        }
        return out;
    }
}
export class PgMountainCarEnv {
    static readonly MinPosition: number = -1.2;
    static readonly MaxPosition: number = 0.6;
    static readonly MaxSpeed: number = 0.07;
    static readonly GoalPosition: number = 0.5;
    static readonly Force: number = 0.001;
    static readonly Gravity: number = 0.0025;
    static readonly ShapeScale: number = 13.0;
    static readonly ActionCount: number = 3;
    maxEpisodeSteps: number;
    shapeReward: boolean;
    position: number;
    velocity: number;
    elapsedSteps: number;
    done: boolean;
    lastReward: number;
    lastTerminated: boolean;
    lastTruncated: boolean;
    constructor(maxEpisodeSteps: number, shapeReward: boolean) {
        this.maxEpisodeSteps = maxEpisodeSteps;
        this.shapeReward = shapeReward;
        this.position = 0.0;
        this.velocity = 0.0;
        this.elapsedSteps = 0;
        this.done = true;
        this.lastReward = 0.0;
        this.lastTerminated = false;
        this.lastTruncated = false;
    }
    reset(startPos: number): void {
        this.position = startPos;
        this.velocity = 0.0;
        this.elapsedSteps = 0;
        this.done = false;
    }
    setState(pos: number, vel: number): void {
        this.position = pos;
        this.velocity = vel;
        this.elapsedSteps = 0;
        this.done = false;
    }
    step(action: number): void {
        let velocity = this.velocity + (action - 1 | 0) * PgMountainCarEnv.Force + Math.cos(3.0 * this.position) * (0.0 - PgMountainCarEnv.Gravity);
        velocity = clampF(velocity, 0.0 - PgMountainCarEnv.MaxSpeed, PgMountainCarEnv.MaxSpeed);
        let position = clampF(this.position + velocity, PgMountainCarEnv.MinPosition, PgMountainCarEnv.MaxPosition);
        if (position <= PgMountainCarEnv.MinPosition && velocity < 0.0) {
            velocity = 0.0;
        }
        this.position = position;
        this.velocity = velocity;
        this.elapsedSteps = (this.elapsedSteps + 1 | 0);
        const terminated = position >= PgMountainCarEnv.GoalPosition;
        const truncated = !terminated && this.elapsedSteps >= this.maxEpisodeSteps;
        this.done = terminated || truncated;
        let reward = 0.0 - 1.0;
        if (this.shapeReward) {
            reward += PgMountainCarEnv.ShapeScale * ((a) => (a < (a - a) ? -a : a))(velocity);
        }
        this.lastReward = reward;
        this.lastTerminated = terminated;
        this.lastTruncated = truncated;
    }
    buildObservation(): number[] {
        let obs = [];
        obs.push((this.position + 0.3) / 0.9);
        obs.push(this.velocity / PgMountainCarEnv.MaxSpeed);
        return obs;
    }
    chooseAction(net: PgMlpNet): number {
        const logits = net.forward(this.buildObservation());
        let best = 0;
        for (let a = 1; a < logits.length; a++) {
            if (logits[a] > logits[best]) {
                best = a;
            }
        }
        return best;
    }
}
export function clampF(v: number, lo: number, hi: number): number {
    return ((a, b) => (a <= b ? a : b))(hi, ((a, b) => (a >= b ? a : b))(lo, v));
}

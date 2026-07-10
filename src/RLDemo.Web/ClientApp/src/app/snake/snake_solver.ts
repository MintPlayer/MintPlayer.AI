export type Option<T> = { tag: "Some"; value: T } | { tag: "None" };
export class PgSnakeBeamNode {
    env: PgSnakeEnv;
    firstMove: number;
    score: number;
    constructor(env: PgSnakeEnv, firstMove: number, score: number) {
        this.env = env;
        this.firstMove = firstMove;
        this.score = score;
    }
    equals(other: PgSnakeBeamNode): boolean {
        return this.env === other.env && this.firstMove === other.firstMove && this.score === other.score;
    }
}
export class PgSnakeNet {
    inputSize: number;
    actions: number;
    hidden: number[];
    trunkWFlat: number[];
    trunkBFlat: number[];
    valueW: number[];
    valueB: number[];
    advW: number[];
    advB: number[];
    constructor(inputSize: number, actions: number, hidden: number[], trunkWFlat: number[], trunkBFlat: number[], valueW: number[], valueB: number[], advW: number[], advB: number[]) {
        this.inputSize = inputSize;
        this.actions = actions;
        this.hidden = hidden;
        this.trunkWFlat = trunkWFlat;
        this.trunkBFlat = trunkBFlat;
        this.valueW = valueW;
        this.valueB = valueB;
        this.advW = advW;
        this.advB = advB;
    }
    forward(obs: number[]): number[] {
        let x = obs;
        let prev = this.inputSize;
        let wOff = 0;
        let bOff = 0;
        for (let l = 0; l < this.hidden.length; l++) {
            const h = this.hidden[l];
            x = PgSnakeNet.relu(PgSnakeNet.linear(x, this.trunkWFlat, wOff, this.trunkBFlat, bOff, prev, h));
            wOff += Math.imul(prev, h);
            bOff += h;
            prev = h;
        }
        const value = PgSnakeNet.linear(x, this.valueW, 0, this.valueB, 0, prev, 1);
        const adv = PgSnakeNet.linear(x, this.advW, 0, this.advB, 0, prev, this.actions);
        let sumA = 0.0;
        for (let k = 0; k < this.actions; k++) {
            sumA += adv[k];
        }
        const meanA = sumA / this.actions;
        const v = value[0];
        let q = [];
        for (let k = 0; k < this.actions; k++) {
            q.push(adv[k] + (v - meanA));
        }
        return q;
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
    static relu(x: number[]): number[] {
        let out = [];
        for (const v of x) {
            out.push(((a, b) => (a >= b ? a : b))(0.0, v));
        }
        return out;
    }
}
export class PgSnakeEnv {
    static readonly ActionCount: number = 4;
    static readonly PatchRadius: number = 4;
    static readonly PatchSide: number = 9;
    static readonly PatchPlane: number = 81;
    static readonly PatchSize: number = 162;
    static readonly ObservationSize: number = 177;
    static readonly FoodReward: number = 1.0;
    static readonly DeathReward: number = -1.0;
    size: number;
    cells: number;
    stepPenalty: number;
    safeMask: boolean;
    body: number[];
    occupied: boolean[];
    food: number;
    foodEaten: number;
    heading: number;
    elapsedSteps: number;
    stepsSinceFood: number;
    done: boolean;
    lastReward: number;
    lastTerminated: boolean;
    lastTruncated: boolean;
    needsFood: boolean;
    constructor(size: number, stepPenalty: number, safeMask: boolean) {
        this.size = size;
        this.cells = Math.imul(size, size);
        this.stepPenalty = stepPenalty;
        this.safeMask = safeMask;
        this.body = [];
        this.occupied = [];
        for (let i = 0; i < this.cells; i++) {
            this.occupied.push(false);
        }
        this.food = 0;
        this.foodEaten = 0;
        this.heading = 3;
        this.elapsedSteps = 0;
        this.stepsSinceFood = 0;
        this.done = true;
        this.lastReward = 0.0;
        this.lastTerminated = false;
        this.lastTruncated = false;
        this.needsFood = false;
    }
    static drOf(a: number): number {
        if (a === 0) {
            return (-1 | 0);
        }
        if (a === 1) {
            return 1;
        }
        return 0;
    }
    static dcOf(a: number): number {
        if (a === 2) {
            return (-1 | 0);
        }
        if (a === 3) {
            return 1;
        }
        return 0;
    }
    static iabs(x: number): number {
        if (x < 0) {
            return (0 - x | 0);
        }
        return x;
    }
    headCell(): number {
        return this.body[(this.body.length - 1 | 0)];
    }
    neckCell(): number {
        return this.body[(this.body.length - 2 | 0)];
    }
    tailCell(): number {
        return this.body[0];
    }
    get length(): number {
        return this.body.length;
    }
    freeCount(): number {
        return (this.cells - this.body.length | 0);
    }
    reset(): void {
        this.body = [];
        for (let i = 0; i < this.cells; i++) {
            this.occupied[i] = false;
        }
        const row = (this.size / 2 | 0);
        const headCol = (this.size / 2 | 0);
        for (let c = (headCol - 2 | 0); c < (headCol + 1 | 0); c++) {
            const cell = (Math.imul(row, this.size) + c | 0);
            this.body.push(cell);
            this.occupied[cell] = true;
        }
        this.heading = 3;
        this.foodEaten = 0;
        this.elapsedSteps = 0;
        this.stepsSinceFood = 0;
        this.done = false;
    }
    spawnFood(pick: number): void {
        let p = pick;
        for (let cell = 0; cell < this.cells; cell++) {
            if (this.occupied[cell]) {
                continue;
            }
            if (p === 0) {
                this.food = cell;
                return;
            }
            p = (p - 1 | 0);
        }
    }
    step(action: number): void {
        this.needsFood = false;
        const head = this.headCell();
        const headRow = (head / this.size | 0);
        const headCol = (head % this.size | 0);
        const newRow = (headRow + PgSnakeEnv.drOf(action) | 0);
        const newCol = (headCol + PgSnakeEnv.dcOf(action) | 0);
        if (newRow < 0 || newRow >= this.size || newCol < 0 || newCol >= this.size) {
            this.die();
            return;
        }
        const newHead = (Math.imul(newRow, this.size) + newCol | 0);
        const eating = newHead === this.food;
        const tail = this.tailCell();
        if (this.occupied[newHead] && !(newHead === tail && !eating)) {
            this.die();
            return;
        }
        this.body.push(newHead);
        this.heading = action;
        let terminated = false;
        if (eating) {
            this.occupied[newHead] = true;
            this.foodEaten = (this.foodEaten + 1 | 0);
            this.stepsSinceFood = 0;
            this.lastReward = PgSnakeEnv.FoodReward;
            if (this.body.length === this.cells) {
                terminated = true;
            } else {
                this.needsFood = true;
            }
        } else {
            this.occupied[tail] = false;
            this.body.splice(0, 1);
            this.occupied[newHead] = true;
            this.stepsSinceFood = (this.stepsSinceFood + 1 | 0);
            this.lastReward = this.stepPenalty;
        }
        this.elapsedSteps = (this.elapsedSteps + 1 | 0);
        const starveLimit = Math.imul(2, this.cells);
        const maxSteps = Math.imul(100, this.cells);
        const truncated = !terminated && (this.stepsSinceFood >= starveLimit || this.elapsedSteps >= maxSteps);
        this.done = terminated || truncated;
        this.lastTerminated = terminated;
        this.lastTruncated = truncated;
    }
    die(): void {
        this.done = true;
        this.lastReward = PgSnakeEnv.DeathReward;
        this.lastTerminated = true;
        this.lastTruncated = false;
    }
    obstacle(r: number, c: number, tail: number): boolean {
        if (r < 0 || r >= this.size || c < 0 || c >= this.size) {
            return true;
        }
        const cell = (Math.imul(r, this.size) + c | 0);
        return this.occupied[cell] && cell !== tail;
    }
    freeCell(r: number, c: number, tail: number): boolean {
        if (r < 0 || r >= this.size || c < 0 || c >= this.size) {
            return false;
        }
        const cell = (Math.imul(r, this.size) + c | 0);
        return !this.occupied[cell] || cell === tail;
    }
    reachableFreeSpace(r: number, c: number, tail: number): number {
        if (!this.freeCell(r, c, tail)) {
            return 0;
        }
        let seen = [];
        for (let i = 0; i < this.cells; i++) {
            seen.push(false);
        }
        seen[(Math.imul(r, this.size) + c | 0)] = true;
        let changed = true;
        for (let _pass = 0; _pass < this.cells; _pass++) {
            if (!changed) {
                break;
            }
            changed = false;
            for (let cell = 0; cell < this.cells; cell++) {
                if (!seen[cell]) {
                    continue;
                }
                const cr = (cell / this.size | 0);
                const cc = (cell % this.size | 0);
                for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
                    const nr = (cr + PgSnakeEnv.drOf(a) | 0);
                    const nc = (cc + PgSnakeEnv.dcOf(a) | 0);
                    if (!this.freeCell(nr, nc, tail)) {
                        continue;
                    }
                    const n = (Math.imul(nr, this.size) + nc | 0);
                    if (!seen[n]) {
                        seen[n] = true;
                        changed = true;
                    }
                }
            }
        }
        let count = 0;
        for (let cell = 0; cell < this.cells; cell++) {
            if (seen[cell]) {
                count = (count + 1 | 0);
            }
        }
        return count;
    }
    buildObservation(): number[] {
        const head = this.headCell();
        const hr = (head / this.size | 0);
        const hc = (head % this.size | 0);
        const tail = this.tailCell();
        const fr = (this.food / this.size | 0);
        const fc = (this.food % this.size | 0);
        let obs = [];
        for (let _pad = 0; _pad < PgSnakeEnv.ObservationSize; _pad++) {
            obs.push(0.0);
        }
        let i = 0;
        for (let dr = (0 - PgSnakeEnv.PatchRadius | 0); dr < (PgSnakeEnv.PatchRadius + 1 | 0); dr++) {
            for (let dc = (0 - PgSnakeEnv.PatchRadius | 0); dc < (PgSnakeEnv.PatchRadius + 1 | 0); dc++) {
                const r = (hr + dr | 0);
                const c = (hc + dc | 0);
                if ((dr !== 0 || dc !== 0) && this.obstacle(r, c, tail)) {
                    obs[i] = 1.0;
                }
                if (r === fr && c === fc) {
                    obs[(PgSnakeEnv.PatchPlane + i | 0)] = 1.0;
                }
                i = (i + 1 | 0);
            }
        }
        let s = PgSnakeEnv.PatchSize;
        obs[s] = (fc - hc | 0) / this.size;
        s = (s + 1 | 0);
        obs[s] = (fr - hr | 0) / this.size;
        s = (s + 1 | 0);
        obs[s] = (PgSnakeEnv.iabs((fr - hr | 0)) + PgSnakeEnv.iabs((fc - hc | 0)) | 0) / (2.0 * this.size);
        s = (s + 1 | 0);
        obs[s] = (this.heading === 0 ? 1.0 : 0.0);
        s = (s + 1 | 0);
        obs[s] = (this.heading === 1 ? 1.0 : 0.0);
        s = (s + 1 | 0);
        obs[s] = (this.heading === 2 ? 1.0 : 0.0);
        s = (s + 1 | 0);
        obs[s] = (this.heading === 3 ? 1.0 : 0.0);
        s = (s + 1 | 0);
        obs[s] = this.body.length / this.cells;
        s = (s + 1 | 0);
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            obs[s] = this.reachableFreeSpace((hr + PgSnakeEnv.drOf(a) | 0), (hc + PgSnakeEnv.dcOf(a) | 0), tail) / this.cells;
            s = (s + 1 | 0);
        }
        const tr = (tail / this.size | 0);
        const tc = (tail % this.size | 0);
        obs[s] = (tc - hc | 0) / this.size;
        s = (s + 1 | 0);
        obs[s] = (tr - hr | 0) / this.size;
        s = (s + 1 | 0);
        obs[s] = (PgSnakeEnv.iabs((tr - hr | 0)) + PgSnakeEnv.iabs((tc - hc | 0)) | 0) / (2.0 * this.size);
        s = (s + 1 | 0);
        return obs;
    }
    currentActionMask(): boolean[] {
        let mask = [];
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            mask.push(true);
        }
        if (this.body.length < 2) {
            return mask;
        }
        const head = this.headCell();
        const neck = this.neckCell();
        const headRow = (head / this.size | 0);
        const headCol = (head % this.size | 0);
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            const r = (headRow + PgSnakeEnv.drOf(a) | 0);
            const c = (headCol + PgSnakeEnv.dcOf(a) | 0);
            if (r >= 0 && r < this.size && c >= 0 && c < this.size && (Math.imul(r, this.size) + c | 0) === neck) {
                mask[a] = false;
            }
        }
        if (!this.safeMask) {
            return mask;
        }
        const tail = this.tailCell();
        const len = this.body.length;
        let safe = [];
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            safe.push(mask[a]);
        }
        let any = false;
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            if (!mask[a]) {
                continue;
            }
            if (this.reachableFreeSpace((headRow + PgSnakeEnv.drOf(a) | 0), (headCol + PgSnakeEnv.dcOf(a) | 0), tail) >= len) {
                any = true;
            } else {
                safe[a] = false;
            }
        }
        if (any) {
            return safe;
        }
        return mask;
    }
    chooseAction(net: PgSnakeNet): number {
        const q = net.forward(this.buildObservation());
        const mask = this.currentActionMask();
        let best = (-1 | 0);
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            if (!mask[a]) {
                continue;
            }
            if (best < 0) {
                best = a;
            } else {
                if (q[a] > q[best]) {
                    best = a;
                }
            }
        }
        return best;
    }
    clone(): PgSnakeEnv {
        const c = new PgSnakeEnv(this.size, this.stepPenalty, false);
        for (const x of this.body) {
            c.body.push(x);
        }
        for (let i = 0; i < this.cells; i++) {
            c.occupied[i] = this.occupied[i];
        }
        c.food = this.food;
        c.foodEaten = this.foodEaten;
        c.heading = this.heading;
        c.elapsedSteps = this.elapsedSteps;
        c.stepsSinceFood = this.stepsSinceFood;
        c.done = this.done;
        return c;
    }
    simSpawnFood(): void {
        for (let cell = 0; cell < this.cells; cell++) {
            if (!this.occupied[cell]) {
                this.food = cell;
                return;
            }
        }
    }
    freeSpaceAhead(): number {
        const head = this.headCell();
        const hr = (head / this.size | 0);
        const hc = (head % this.size | 0);
        const tail = this.tailCell();
        let best = 0;
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            const v = this.reachableFreeSpace((hr + PgSnakeEnv.drOf(a) | 0), (hc + PgSnakeEnv.dcOf(a) | 0), tail);
            if (v > best) {
                best = v;
            }
        }
        return best;
    }
    firstLegalAction(): number {
        const mask = this.currentActionMask();
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            if (mask[a]) {
                return a;
            }
        }
        return 0;
    }
    leafScoreSearch(rootFood: number, depth: number, foodWeight: number, trapPenalty: number, spaceWeight: number, foodDistWeight: number): number {
        if (this.lastTerminated && this.body.length === this.cells) {
            return 1000000000.0;
        }
        if (this.lastTerminated) {
            return -1000000.0 + depth * 1000.0;
        }
        const foodGained = (this.foodEaten - rootFood | 0);
        const free = this.freeSpaceAhead();
        let score = foodGained * foodWeight;
        if (free < this.body.length) {
            score = score - trapPenalty;
        }
        score = score + free * spaceWeight;
        const head = this.headCell();
        const hr = (head / this.size | 0);
        const hc = (head % this.size | 0);
        const fr = (this.food / this.size | 0);
        const fc = (this.food % this.size | 0);
        const foodDist = (PgSnakeEnv.iabs((fr - hr | 0)) + PgSnakeEnv.iabs((fc - hc | 0)) | 0);
        score = score - foodDist * foodDistWeight;
        return score;
    }
    chooseActionSearch(net: PgSnakeNet, maxDepth: number, beamWidth: number, foodWeight: number, trapPenalty: number, netWeight: number, spaceWeight: number, foodDistWeight: number): number {
        const rootFood = this.foodEaten;
        let bestByRoot = [];
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            bestByRoot.push(-1000000000000.0);
        }
        const root = this.clone();
        let beam = [];
        beam.push(new PgSnakeBeamNode(root, (-1 | 0), 0.0));
        for (let depth = 0; depth < maxDepth; depth++) {
            if (beam.length === 0) {
                break;
            }
            let next = [];
            for (const node of beam) {
                const mask = node.env.currentActionMask();
                for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
                    if (!mask[a]) {
                        continue;
                    }
                    const child = node.env.clone();
                    child.step(a);
                    if (child.needsFood) {
                        child.simSpawnFood();
                    }
                    const childFirst = (node.firstMove < 0 ? a : node.firstMove);
                    const score = child.leafScoreSearch(rootFood, (depth + 1 | 0), foodWeight, trapPenalty, spaceWeight, foodDistWeight);
                    if (score > bestByRoot[childFirst]) {
                        bestByRoot[childFirst] = score;
                    }
                    if (!child.done && (depth + 1 | 0) < maxDepth) {
                        next.push(new PgSnakeBeamNode(child, childFirst, score));
                    }
                }
            }
            beam = PgSnakeEnv.pruneBeam(next, beamWidth);
        }
        const rootQ = net.forward(this.buildObservation());
        let bestFinal = -1000000000000000.0;
        let bestFirst = root.firstLegalAction();
        for (let a = 0; a < PgSnakeEnv.ActionCount; a++) {
            if (bestByRoot[a] <= -1000000000000.0) {
                continue;
            }
            let v = bestByRoot[a];
            if (netWeight !== 0.0) {
                v = v + rootQ[a] * netWeight;
            }
            if (v > bestFinal) {
                bestFinal = v;
                bestFirst = a;
            }
        }
        return bestFirst;
    }
    static pruneBeam(nodes: PgSnakeBeamNode[], k: number): PgSnakeBeamNode[] {
        if (nodes.length <= k) {
            return nodes;
        }
        let kept = [];
        let used = [];
        for (let i = 0; i < nodes.length; i++) {
            used.push(false);
        }
        for (let _iter = 0; _iter < k; _iter++) {
            let bestIdx = (-1 | 0);
            let bestVal = 0.0;
            for (let i = 0; i < nodes.length; i++) {
                if (used[i]) {
                    continue;
                }
                if (bestIdx < 0 || nodes[i].score > bestVal) {
                    bestVal = nodes[i].score;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) {
                break;
            }
            used[bestIdx] = true;
            kept.push(nodes[bestIdx]);
        }
        return kept;
    }
}

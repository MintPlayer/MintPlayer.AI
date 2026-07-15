export type Option<T> = { tag: "Some"; value: T } | { tag: "None" };
export class PgDraughtsMove {
    from: number;
    to: number;
    captures: number[];
    constructor(from: number, to: number, captures: number[]) {
        this.from = from;
        this.to = to;
        this.captures = captures;
    }
    equals(other: PgDraughtsMove): boolean {
        return this.from === other.from && this.to === other.to && this.captures === other.captures;
    }
}
export class PgDraughtsNetOut {
    logits: number[];
    value: number;
    constructor(logits: number[], value: number) {
        this.logits = logits;
        this.value = value;
    }
    equals(other: PgDraughtsNetOut): boolean {
        return this.logits === other.logits && this.value === other.value;
    }
}
export class PgDraughtsLeafEval {
    priors: number[];
    value: number;
    constructor(priors: number[], value: number) {
        this.priors = priors;
        this.value = value;
    }
    equals(other: PgDraughtsLeafEval): boolean {
        return this.priors === other.priors && this.value === other.value;
    }
}
export class PgDraughtsState {
    size: number;
    flyingKings: boolean;
    menCaptureBackward: boolean;
    majorityCapture: boolean;
    noProgressLimit: number;
    squares: number[];
    whiteToMove: boolean;
    noProgress: number;
    constructor(size: number, flyingKings: boolean, menCaptureBackward: boolean, majorityCapture: boolean, noProgressLimit: number, squares: number[], whiteToMove: boolean, noProgress: number) {
        this.size = size;
        this.flyingKings = flyingKings;
        this.menCaptureBackward = menCaptureBackward;
        this.majorityCapture = majorityCapture;
        this.noProgressLimit = noProgressLimit;
        this.squares = squares;
        this.whiteToMove = whiteToMove;
        this.noProgress = noProgress;
    }
    static international(): PgDraughtsState {
        return PgDraughtsState.start(10, true, true, true, 80);
    }
    static english(): PgDraughtsState {
        return PgDraughtsState.start(8, false, false, false, 80);
    }
    static start(size: number, flyingKings: boolean, menCaptureBackward: boolean, majorityCapture: boolean, noProgressLimit: number): PgDraughtsState {
        let b: number[] = [];
        const cells = Math.imul(size, size);
        for (let sq = 0; sq < cells; sq++) {
            b.push(0);
        }
        const rows = ((size / 2 | 0) - 1 | 0);
        for (let sq = 0; sq < cells; sq++) {
            const f = (sq % size | 0);
            const r = (sq / size | 0);
            if ((((f + r | 0)) % 2 | 0) !== 0) {
                continue;
            }
            if (r < rows) {
                b[sq] = 1;
            }
            if (r >= (size - rows | 0)) {
                b[sq] = (-1 | 0);
            }
        }
        return new PgDraughtsState(size, flyingKings, menCaptureBackward, majorityCapture, noProgressLimit, b, true, 0);
    }
    fileOf(sq: number): number {
        return (sq % this.size | 0);
    }
    rankOf(sq: number): number {
        return (sq / this.size | 0);
    }
    onBoard(f: number, r: number): boolean {
        return f >= 0 && f < this.size && r >= 0 && r < this.size;
    }
    crownRank(white: boolean): number {
        return (white ? (this.size - 1 | 0) : 0);
    }
    static isWhitePiece(p: number): boolean {
        return p > 0;
    }
    static pieceKind(p: number): number {
        return ((a) => (a < (a - a) ? -a : a))(p);
    }
    static contains(xs: number[], v: number): boolean {
        for (const x of xs) {
            if (x === v) {
                return true;
            }
        }
        return false;
    }
    static diagDirs(): [number, number][] {
        let d: [number, number][] = [];
        d.push([1, 1]);
        d.push([(-1 | 0), 1]);
        d.push([1, (-1 | 0)]);
        d.push([(-1 | 0), (-1 | 0)]);
        return d;
    }
    legalMoves(): PgDraughtsMove[] {
        const caps = this.captureMoves();
        if (caps.length > 0) {
            return caps;
        }
        return this.quietMoves();
    }
    captureMoves(): PgDraughtsMove[] {
        let found: PgDraughtsMove[] = [];
        const cells = Math.imul(this.size, this.size);
        let b: number[] = [];
        for (const v of this.squares) {
            b.push(v);
        }
        for (let sq = 0; sq < cells; sq++) {
            const p = b[sq];
            if (p === 0 || PgDraughtsState.isWhitePiece(p) !== this.whiteToMove) {
                continue;
            }
            b[sq] = 0;
            this.captureDfs(b, sq, sq, PgDraughtsState.pieceKind(p) === 2, [], found);
            b[sq] = p;
        }
        const moves = PgDraughtsState.dedupe(found);
        if (this.majorityCapture && moves.length > 0) {
            let best = 0;
            for (const m of moves) {
                best = ((a, b) => (a >= b ? a : b))(best, m.captures.length);
            }
            let top: PgDraughtsMove[] = [];
            for (const m of moves) {
                if (m.captures.length === best) {
                    top.push(m);
                }
            }
            return top;
        }
        return moves;
    }
    captureDfs(b: number[], origin: number, sq: number, king: boolean, captured: number[], out: PgDraughtsMove[]): void {
        let extended = false;
        const f = this.fileOf(sq);
        const r = this.rankOf(sq);
        const fwd = (this.whiteToMove ? 1 : (-1 | 0));
        const dirs = PgDraughtsState.diagDirs();
        for (const [df, dr] of dirs) {
            if (!king && !this.menCaptureBackward && dr !== fwd) {
                continue;
            }
            if (king && this.flyingKings) {
                let cstep = (-1 | 0);
                let blocked = false;
                for (let step = 1; step < this.size; step++) {
                    if (blocked || cstep >= 0) {
                        continue;
                    }
                    const nf = (f + Math.imul(df, step) | 0);
                    const nr = (r + Math.imul(dr, step) | 0);
                    if (!this.onBoard(nf, nr)) {
                        blocked = true;
                        continue;
                    }
                    const cell = b[(Math.imul(nr, this.size) + nf | 0)];
                    if (cell === 0) {
                        continue;
                    }
                    if (PgDraughtsState.isWhitePiece(cell) === this.whiteToMove || PgDraughtsState.contains(captured, (Math.imul(nr, this.size) + nf | 0))) {
                        blocked = true;
                    } else {
                        cstep = step;
                    }
                }
                if (cstep < 0) {
                    continue;
                }
                const csq = (Math.imul((r + Math.imul(dr, cstep) | 0), this.size) + ((f + Math.imul(df, cstep) | 0)) | 0);
                let stop = false;
                for (let step = 1; step < this.size; step++) {
                    if (stop || step <= cstep) {
                        continue;
                    }
                    const lf = (f + Math.imul(df, step) | 0);
                    const lr = (r + Math.imul(dr, step) | 0);
                    if (!this.onBoard(lf, lr)) {
                        stop = true;
                        continue;
                    }
                    const land = (Math.imul(lr, this.size) + lf | 0);
                    if (b[land] !== 0) {
                        stop = true;
                        continue;
                    }
                    extended = true;
                    let c2: number[] = [];
                    for (const c of captured) {
                        c2.push(c);
                    }
                    c2.push(csq);
                    this.captureDfs(b, origin, land, king, c2, out);
                }
            } else {
                const lf = (f + Math.imul(df, 2) | 0);
                const lr = (r + Math.imul(dr, 2) | 0);
                if (!this.onBoard(lf, lr)) {
                    continue;
                }
                const mid = (Math.imul((r + dr | 0), this.size) + ((f + df | 0)) | 0);
                const cell = b[mid];
                if (cell === 0 || PgDraughtsState.isWhitePiece(cell) === this.whiteToMove) {
                    continue;
                }
                if (PgDraughtsState.contains(captured, mid)) {
                    continue;
                }
                const land = (Math.imul(lr, this.size) + lf | 0);
                if (b[land] !== 0) {
                    continue;
                }
                extended = true;
                let c2: number[] = [];
                for (const c of captured) {
                    c2.push(c);
                }
                c2.push(mid);
                this.captureDfs(b, origin, land, king, c2, out);
            }
        }
        if (!extended && captured.length > 0) {
            out.push(new PgDraughtsMove(origin, sq, captured));
        }
    }
    quietMoves(): PgDraughtsMove[] {
        let moves: PgDraughtsMove[] = [];
        const cells = Math.imul(this.size, this.size);
        const fwd = (this.whiteToMove ? 1 : (-1 | 0));
        for (let sq = 0; sq < cells; sq++) {
            const p = this.squares[sq];
            if (p === 0 || PgDraughtsState.isWhitePiece(p) !== this.whiteToMove) {
                continue;
            }
            const f = this.fileOf(sq);
            const r = this.rankOf(sq);
            const king = PgDraughtsState.pieceKind(p) === 2;
            const dirs = PgDraughtsState.diagDirs();
            for (const [df, dr] of dirs) {
                if (!king && dr !== fwd) {
                    continue;
                }
                if (king && this.flyingKings) {
                    let blocked = false;
                    for (let step = 1; step < this.size; step++) {
                        if (blocked) {
                            continue;
                        }
                        const nf = (f + Math.imul(df, step) | 0);
                        const nr = (r + Math.imul(dr, step) | 0);
                        if (!this.onBoard(nf, nr) || this.squares[(Math.imul(nr, this.size) + nf | 0)] !== 0) {
                            blocked = true;
                            continue;
                        }
                        moves.push(new PgDraughtsMove(sq, (Math.imul(nr, this.size) + nf | 0), []));
                    }
                } else {
                    const nf = (f + df | 0);
                    const nr = (r + dr | 0);
                    if (!this.onBoard(nf, nr)) {
                        continue;
                    }
                    const to = (Math.imul(nr, this.size) + nf | 0);
                    if (this.squares[to] === 0) {
                        moves.push(new PgDraughtsMove(sq, to, []));
                    }
                }
            }
        }
        return moves;
    }
    static dedupe(moves: PgDraughtsMove[]): PgDraughtsMove[] {
        let out: PgDraughtsMove[] = [];
        for (const m of moves) {
            let dup = false;
            for (const k of out) {
                if (k.from === m.from && k.to === m.to && PgDraughtsState.sameSet(k.captures, m.captures)) {
                    dup = true;
                }
            }
            if (!dup) {
                out.push(m);
            }
        }
        return out;
    }
    static sameSet(a: number[], b: number[]): boolean {
        if (a.length !== b.length) {
            return false;
        }
        for (const x of a) {
            if (!PgDraughtsState.contains(b, x)) {
                return false;
            }
        }
        return true;
    }
    makeMove(move: PgDraughtsMove): PgDraughtsState {
        let b: number[] = [];
        for (const v of this.squares) {
            b.push(v);
        }
        const piece = b[move.from];
        b[move.from] = 0;
        for (const c of move.captures) {
            b[c] = 0;
        }
        let placed = piece;
        if (PgDraughtsState.pieceKind(piece) === 1 && this.rankOf(move.to) === this.crownRank(this.whiteToMove)) {
            placed = (this.whiteToMove ? 2 : (-2 | 0));
        }
        b[move.to] = placed;
        const progress = (move.captures.length > 0 || PgDraughtsState.pieceKind(piece) === 1 ? 0 : (this.noProgress + 1 | 0));
        return new PgDraughtsState(this.size, this.flyingKings, this.menCaptureBackward, this.majorityCapture, this.noProgressLimit, b, !this.whiteToMove, progress);
    }
    result(): number {
        if (this.noProgress >= this.noProgressLimit) {
            return 2;
        }
        if (this.legalMoves().length === 0) {
            return 1;
        }
        return 0;
    }
    perft(depth: number): number {
        if (depth === 0) {
            return 1;
        }
        const moves = this.legalMoves();
        if (depth === 1) {
            return moves.length;
        }
        let nodes = 0;
        for (const m of moves) {
            nodes += this.makeMove(m).perft((depth - 1 | 0));
        }
        return nodes;
    }
    moverSq(sq: number): number {
        return (this.whiteToMove ? sq : ((Math.imul(this.size, this.size) - 1 | 0) - sq | 0));
    }
    actionSpace(): number {
        return Math.imul((Math.imul(this.size, this.size) / 2 | 0), (Math.imul(this.size, this.size) / 2 | 0));
    }
    moveIndex(move: PgDraughtsMove): number {
        const half = (Math.imul(this.size, this.size) / 2 | 0);
        return (Math.imul((this.moverSq(move.from) / 2 | 0), half) + (this.moverSq(move.to) / 2 | 0) | 0);
    }
    legalMoveIndices(): number[] {
        let out: number[] = [];
        const moves = this.legalMoves();
        for (const m of moves) {
            const idx = this.moveIndex(m);
            if (!PgDraughtsState.contains(out, idx)) {
                out.push(idx);
            }
        }
        return out;
    }
    applyIndex(index: number): PgDraughtsState {
        const moves = this.legalMoves();
        let chosen = moves[0];
        let found = false;
        for (const m of moves) {
            if (this.moveIndex(m) !== index) {
                continue;
            }
            if (!found) {
                chosen = m;
                found = true;
                continue;
            }
            if (PgDraughtsState.canonicalBefore(m, chosen)) {
                chosen = m;
            }
        }
        return this.makeMove(chosen);
    }
    static canonicalBefore(a: PgDraughtsMove, b: PgDraughtsMove): boolean {
        if (a.captures.length !== b.captures.length) {
            return a.captures.length > b.captures.length;
        }
        const sa = PgDraughtsState.sortedCopy(a.captures);
        const sb = PgDraughtsState.sortedCopy(b.captures);
        for (let i = 0; i < sa.length; i++) {
            if (sa[i] < sb[i]) {
                return true;
            }
            if (sa[i] > sb[i]) {
                return false;
            }
        }
        return false;
    }
    static sortedCopy(xs: number[]): number[] {
        let out: number[] = [];
        for (const x of xs) {
            out.push(x);
        }
        for (let i = 0; i < out.length; i++) {
            for (let j = 0; j < out.length; j++) {
                if (j <= i) {
                    continue;
                }
                if (out[j] < out[i]) {
                    const t = out[i];
                    out[i] = out[j];
                    out[j] = t;
                }
            }
        }
        return out;
    }
    writeObservation(): number[] {
        const cells = Math.imul(this.size, this.size);
        const total = Math.imul(5, cells);
        let obs: number[] = [];
        for (let i = 0; i < total; i++) {
            obs.push(0.0);
        }
        for (let sq = 0; sq < cells; sq++) {
            const p = this.squares[sq];
            if (p === 0) {
                continue;
            }
            const base = (PgDraughtsState.isWhitePiece(p) === this.whiteToMove ? 0 : 2);
            const plane = ((base + PgDraughtsState.pieceKind(p) | 0) - 1 | 0);
            obs[(Math.imul(plane, cells) + this.moverSq(sq) | 0)] = 1.0;
        }
        let clock = this.noProgress * 1.0 / this.noProgressLimit;
        if (clock > 1.0) {
            clock = 1.0;
        }
        for (let sq = 0; sq < cells; sq++) {
            obs[(Math.imul(4, cells) + sq | 0)] = clock;
        }
        return obs;
    }
}
export class PgDraughtsConvNet {
    planes: number;
    boardH: number;
    boardW: number;
    filters: number;
    blocks: number;
    actions: number;
    stemW: number[];
    stemB: number[];
    stemNG: number[];
    stemNB: number[];
    b1W: number[];
    b1B: number[];
    n1G: number[];
    n1B: number[];
    b2W: number[];
    b2B: number[];
    n2G: number[];
    n2B: number[];
    pConvW: number[];
    pConvB: number[];
    pNG: number[];
    pNB: number[];
    pHeadW: number[];
    pHeadB: number[];
    vConvW: number[];
    vConvB: number[];
    vNG: number[];
    vNB: number[];
    vHidW: number[];
    vHidB: number[];
    vHeadW: number[];
    vHeadB: number[];
    constructor(planes: number, boardH: number, boardW: number, filters: number, blocks: number, actions: number) {
        this.planes = planes;
        this.boardH = boardH;
        this.boardW = boardW;
        this.filters = filters;
        this.blocks = blocks;
        this.actions = actions;
        this.stemW = [];
        this.stemB = [];
        this.stemNG = [];
        this.stemNB = [];
        this.b1W = [];
        this.b1B = [];
        this.n1G = [];
        this.n1B = [];
        this.b2W = [];
        this.b2B = [];
        this.n2G = [];
        this.n2B = [];
        this.pConvW = [];
        this.pConvB = [];
        this.pNG = [];
        this.pNB = [];
        this.pHeadW = [];
        this.pHeadB = [];
        this.vConvW = [];
        this.vConvB = [];
        this.vNG = [];
        this.vNB = [];
        this.vHidW = [];
        this.vHidB = [];
        this.vHeadW = [];
        this.vHeadB = [];
    }
    forward(obs: number[]): PgDraughtsNetOut {
        const hw = Math.imul(this.boardH, this.boardW);
        const f = this.filters;
        let x = PgDraughtsConvNet.relu(PgDraughtsConvNet.layerNorm(PgDraughtsConvNet.conv2d(obs, this.stemW, 0, this.stemB, 0, this.planes, f, 3, 1, this.boardH, this.boardW), this.stemNG, 0, this.stemNB, 0));
        const wStride = Math.imul(Math.imul(f, 9), f);
        for (let i = 0; i < this.blocks; i++) {
            const wOff = Math.imul(i, wStride);
            const bOff = Math.imul(i, f);
            const nOff = Math.imul(i, Math.imul(f, hw));
            let h = PgDraughtsConvNet.relu(PgDraughtsConvNet.layerNorm(PgDraughtsConvNet.conv2d(x, this.b1W, wOff, this.b1B, bOff, f, f, 3, 1, this.boardH, this.boardW), this.n1G, nOff, this.n1B, nOff));
            h = PgDraughtsConvNet.layerNorm(PgDraughtsConvNet.conv2d(h, this.b2W, wOff, this.b2B, bOff, f, f, 3, 1, this.boardH, this.boardW), this.n2G, nOff, this.n2B, nOff);
            x = PgDraughtsConvNet.relu(PgDraughtsConvNet.addVec(x, h));
        }
        const p = PgDraughtsConvNet.relu(PgDraughtsConvNet.layerNorm(PgDraughtsConvNet.conv2d(x, this.pConvW, 0, this.pConvB, 0, f, 2, 1, 0, this.boardH, this.boardW), this.pNG, 0, this.pNB, 0));
        const logits = PgDraughtsConvNet.linear(p, this.pHeadW, 0, this.pHeadB, 0, Math.imul(2, hw), this.actions);
        const vc = PgDraughtsConvNet.relu(PgDraughtsConvNet.layerNorm(PgDraughtsConvNet.conv2d(x, this.vConvW, 0, this.vConvB, 0, f, 1, 1, 0, this.boardH, this.boardW), this.vNG, 0, this.vNB, 0));
        const vh = PgDraughtsConvNet.relu(PgDraughtsConvNet.linear(vc, this.vHidW, 0, this.vHidB, 0, hw, f));
        const value = PgDraughtsConvNet.linear(vh, this.vHeadW, 0, this.vHeadB, 0, f, 1);
        return new PgDraughtsNetOut(logits, value[0]);
    }
    static conv2d(x: number[], w: number[], wOff: number, b: number[], bOff: number, inC: number, outC: number, k: number, pad: number, h: number, w2: number): number[] {
        const hw = Math.imul(h, w2);
        let out: number[] = [];
        for (let oc = 0; oc < outC; oc++) {
            const bias = b[(bOff + oc | 0)];
            for (let oh = 0; oh < h; oh++) {
                for (let ow = 0; ow < w2; ow++) {
                    let s = bias;
                    for (let c = 0; c < inC; c++) {
                        for (let kh = 0; kh < k; kh++) {
                            const ih = ((oh - pad | 0) + kh | 0);
                            if (ih < 0) {
                                continue;
                            }
                            if (ih >= h) {
                                continue;
                            }
                            for (let kw = 0; kw < k; kw++) {
                                const iw = ((ow - pad | 0) + kw | 0);
                                if (iw < 0) {
                                    continue;
                                }
                                if (iw >= w2) {
                                    continue;
                                }
                                s += x[((Math.imul(c, hw) + Math.imul(ih, w2) | 0) + iw | 0)] * w[((wOff + Math.imul((Math.imul((Math.imul(c, k) + kh | 0), k) + kw | 0), outC) | 0) + oc | 0)];
                            }
                        }
                    }
                    out.push(s);
                }
            }
        }
        return out;
    }
    static layerNorm(x: number[], g: number[], gOff: number, be: number[], beOff: number): number[] {
        const n = x.length;
        const nf = n * 1.0;
        let mean = 0.0;
        for (const v of x) {
            mean += v;
        }
        mean = mean / nf;
        let vs = 0.0;
        for (const v of x) {
            const d = v - mean;
            vs += d * d;
        }
        const denom = Math.sqrt(vs / nf + 0.00001);
        let out: number[] = [];
        for (let i = 0; i < n; i++) {
            out.push(g[(gOff + i | 0)] * (x[i] - mean) / denom + be[(beOff + i | 0)]);
        }
        return out;
    }
    static linear(x: number[], w: number[], wOff: number, b: number[], bOff: number, inDim: number, outDim: number): number[] {
        let out: number[] = [];
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
        let out: number[] = [];
        for (const v of x) {
            out.push(((a, b) => (a >= b ? a : b))(0.0, v));
        }
        return out;
    }
    static addVec(a: number[], b: number[]): number[] {
        let out: number[] = [];
        for (let i = 0; i < a.length; i++) {
            out.push(a[i] + b[i]);
        }
        return out;
    }
}
export class PgDraughtsNet {
    inputSize: number;
    actions: number;
    hidden: number[];
    trunkWFlat: number[];
    trunkBFlat: number[];
    policyW: number[];
    policyB: number[];
    valueW: number[];
    valueB: number[];
    conv: (PgDraughtsConvNet | null);
    constructor(inputSize: number, actions: number, hidden: number[], trunkWFlat: number[], trunkBFlat: number[], policyW: number[], policyB: number[], valueW: number[], valueB: number[]) {
        this.inputSize = inputSize;
        this.actions = actions;
        this.hidden = hidden;
        this.trunkWFlat = trunkWFlat;
        this.trunkBFlat = trunkBFlat;
        this.policyW = policyW;
        this.policyB = policyB;
        this.valueW = valueW;
        this.valueB = valueB;
        this.conv = null;
    }
    static withConv(conv: PgDraughtsConvNet): PgDraughtsNet {
        const n = new PgDraughtsNet(Math.imul(Math.imul(conv.planes, conv.boardH), conv.boardW), conv.actions, [], [], [], [], [], [], []);
        n.conv = conv;
        return n;
    }
    forward(obs: number[]): PgDraughtsNetOut {
        if (this.conv !== null) {
            return this.conv.forward(obs);
        }
        let x = obs;
        let prev = this.inputSize;
        let wOff = 0;
        let bOff = 0;
        for (let l = 0; l < this.hidden.length; l++) {
            const h = this.hidden[l];
            x = PgDraughtsNet.relu(PgDraughtsNet.linear(x, this.trunkWFlat, wOff, this.trunkBFlat, bOff, prev, h));
            wOff += Math.imul(prev, h);
            bOff += h;
            prev = h;
        }
        const logits = PgDraughtsNet.linear(x, this.policyW, 0, this.policyB, 0, prev, this.actions);
        const value = PgDraughtsNet.linear(x, this.valueW, 0, this.valueB, 0, prev, 1);
        return new PgDraughtsNetOut(logits, value[0]);
    }
    static linear(x: number[], w: number[], wOff: number, b: number[], bOff: number, inDim: number, outDim: number): number[] {
        let out: number[] = [];
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
        let out: number[] = [];
        for (const v of x) {
            out.push(((a, b) => (a >= b ? a : b))(0.0, v));
        }
        return out;
    }
}
export class PgDraughtsMctsNode {
    moves: number[];
    p: number[];
    n: number[];
    w: number[];
    children: (PgDraughtsMctsNode | null)[];
    expanded: boolean;
    terminal: boolean;
    terminalValue: number;
    constructor(moves: number[], terminal: boolean, terminalValue: number) {
        this.moves = moves;
        this.terminal = terminal;
        this.terminalValue = terminalValue;
        this.p = [];
        this.n = [];
        this.w = [];
        this.children = [];
        this.expanded = false;
    }
}
export class PgDraughtsMcts {
    static search(net: PgDraughtsNet, root: PgDraughtsState, sims: number, cpuct: number): number[] {
        const rootNode = PgDraughtsMcts.newNode(root);
        if (!rootNode.terminal) {
            PgDraughtsMcts.expandLeaf(rootNode, net, root);
            for (let s = 0; s < sims; s++) {
                PgDraughtsMcts.simulate(rootNode, net, root, cpuct);
            }
        }
        const space = root.actionSpace();
        let pi: number[] = [];
        for (let i = 0; i < space; i++) {
            pi.push(0.0);
        }
        let total = 0;
        for (let i = 0; i < rootNode.moves.length; i++) {
            total += rootNode.n[i];
        }
        if (total === 0) {
            for (let i = 0; i < rootNode.moves.length; i++) {
                pi[rootNode.moves[i]] = rootNode.p[i];
            }
            return pi;
        }
        for (let i = 0; i < rootNode.moves.length; i++) {
            pi[rootNode.moves[i]] = rootNode.n[i] * 1.0 / total;
        }
        return pi;
    }
    static chooseMove(net: PgDraughtsNet, root: PgDraughtsState, sims: number, cpuct: number): number {
        const pi = PgDraughtsMcts.search(net, root, sims, cpuct);
        let best = (-1 | 0);
        let bestV = -1.0;
        for (let i = 0; i < pi.length; i++) {
            if (pi[i] > bestV) {
                bestV = pi[i];
                best = i;
            }
        }
        return best;
    }
    static simulate(node: PgDraughtsMctsNode, net: PgDraughtsNet, state: PgDraughtsState, cpuct: number): number {
        if (node.terminal) {
            return node.terminalValue;
        }
        if (!node.expanded) {
            return PgDraughtsMcts.expandLeaf(node, net, state);
        }
        const edge = PgDraughtsMcts.selectChild(node, cpuct);
        const childState = state.applyIndex(node.moves[edge]);
        if (node.children[edge] === null) {
            node.children[edge] = PgDraughtsMcts.newNode(childState);
        }
        const child = node.children[edge];
        const value = -PgDraughtsMcts.simulate(child, net, childState, cpuct);
        node.n[edge] = (node.n[edge] + 1 | 0);
        node.w[edge] = node.w[edge] + value;
        return value;
    }
    static selectChild(node: PgDraughtsMctsNode, cpuct: number): number {
        let sumN = 0;
        for (let i = 0; i < node.n.length; i++) {
            sumN += node.n[i];
        }
        const sqrtSum = Math.sqrt(sumN * 1.0);
        let best = 0;
        let bestScore = -1.0e30;
        for (let i = 0; i < node.moves.length; i++) {
            const q = (node.n[i] > 0 ? node.w[i] / node.n[i] : 0.0);
            const u = cpuct * node.p[i] * sqrtSum / (1.0 + node.n[i]);
            const score = q + u;
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        return best;
    }
    static newNode(state: PgDraughtsState): PgDraughtsMctsNode {
        const r = state.result();
        if (r !== 0) {
            const tv = (r === 1 ? -1.0 : 0.0);
            return new PgDraughtsMctsNode([], true, tv);
        }
        return new PgDraughtsMctsNode(state.legalMoveIndices(), false, 0.0);
    }
    static expandLeaf(node: PgDraughtsMctsNode, net: PgDraughtsNet, state: PgDraughtsState): number {
        const ev = PgDraughtsMcts.evaluate(net, state, node.moves);
        const k = node.moves.length;
        node.p = [];
        node.n = [];
        node.w = [];
        node.children = [];
        for (let i = 0; i < k; i++) {
            node.p.push(ev.priors[i]);
            node.n.push(0);
            node.w.push(0.0);
            node.children.push(null);
        }
        node.expanded = true;
        return ev.value;
    }
    static evaluate(net: PgDraughtsNet, state: PgDraughtsState, moves: number[]): PgDraughtsLeafEval {
        const out = net.forward(state.writeObservation());
        let mx = -1.0e30;
        for (const m of moves) {
            if (out.logits[m] > mx) {
                mx = out.logits[m];
            }
        }
        let pr: number[] = [];
        let sum = 0.0;
        for (const m of moves) {
            const e = Math.exp(out.logits[m] - mx);
            pr.push(e);
            sum += e;
        }
        if (sum > 0.0) {
            for (let i = 0; i < pr.length; i++) {
                pr[i] = pr[i] / sum;
            }
        } else {
            for (let i = 0; i < pr.length; i++) {
                pr[i] = 1.0 / pr.length;
            }
        }
        return new PgDraughtsLeafEval(pr, Math.tanh(out.value));
    }
}

export type Option<T> = { tag: "Some"; value: T } | { tag: "None" };
export class PgChessMove {
    from: number;
    to: number;
    promotion: number;
    constructor(from: number, to: number, promotion: number) {
        this.from = from;
        this.to = to;
        this.promotion = promotion;
    }
    equals(other: PgChessMove): boolean {
        return this.from === other.from && this.to === other.to && this.promotion === other.promotion;
    }
}
export class PgNetOut {
    logits: number[];
    value: number;
    constructor(logits: number[], value: number) {
        this.logits = logits;
        this.value = value;
    }
    equals(other: PgNetOut): boolean {
        return this.logits === other.logits && this.value === other.value;
    }
}
export class PgLeafEval {
    priors: number[];
    value: number;
    constructor(priors: number[], value: number) {
        this.priors = priors;
        this.value = value;
    }
    equals(other: PgLeafEval): boolean {
        return this.priors === other.priors && this.value === other.value;
    }
}
export class PgChessState {
    static readonly PromoKnight: number = 2;
    static readonly PromoBishop: number = 3;
    static readonly PromoRook: number = 4;
    static readonly PromoQueen: number = 5;
    static readonly CastleWK: number = 1;
    static readonly CastleWQ: number = 2;
    static readonly CastleBK: number = 4;
    static readonly CastleBQ: number = 8;
    static readonly Size: number = 4672;
    static readonly PlanesPerSquare: number = 73;
    squares: number[];
    whiteToMove: boolean;
    castling: number;
    enPassant: number;
    halfmoveClock: number;
    constructor(squares: number[], whiteToMove: boolean, castling: number, enPassant: number, halfmoveClock: number) {
        this.squares = squares;
        this.whiteToMove = whiteToMove;
        this.castling = castling;
        this.enPassant = enPassant;
        this.halfmoveClock = halfmoveClock;
    }
    static fileOf(sq: number): number {
        return (sq % 8 | 0);
    }
    static rankOf(sq: number): number {
        return (sq / 8 | 0);
    }
    static isign(x: number): number {
        return (x > 0 ? 1 : (x < 0 ? (-1 | 0) : 0));
    }
    static onBoard(f: number, r: number): boolean {
        return f >= 0 && f < 8 && r >= 0 && r < 8;
    }
    static isWhite(piece: number): boolean {
        return piece > 0;
    }
    static pieceType(piece: number): number {
        return (piece < 0 ? (-piece | 0) : piece);
    }
    static clearBits(mask: number, bits: number): number {
        return (mask - (mask & bits) | 0);
    }
    static knightDeltas(): [number, number][] {
        let d = [];
        d.push([1, 2]);
        d.push([2, 1]);
        d.push([2, (-1 | 0)]);
        d.push([1, (-2 | 0)]);
        d.push([(-1 | 0), (-2 | 0)]);
        d.push([(-2 | 0), (-1 | 0)]);
        d.push([(-2 | 0), 1]);
        d.push([(-1 | 0), 2]);
        return d;
    }
    static kingDeltas(): [number, number][] {
        let d = [];
        d.push([1, 0]);
        d.push([1, 1]);
        d.push([0, 1]);
        d.push([(-1 | 0), 1]);
        d.push([(-1 | 0), 0]);
        d.push([(-1 | 0), (-1 | 0)]);
        d.push([0, (-1 | 0)]);
        d.push([1, (-1 | 0)]);
        return d;
    }
    static bishopDirs(): [number, number][] {
        let d = [];
        d.push([1, 1]);
        d.push([(-1 | 0), 1]);
        d.push([1, (-1 | 0)]);
        d.push([(-1 | 0), (-1 | 0)]);
        return d;
    }
    static rookDirs(): [number, number][] {
        let d = [];
        d.push([1, 0]);
        d.push([(-1 | 0), 0]);
        d.push([0, 1]);
        d.push([0, (-1 | 0)]);
        return d;
    }
    static encDirs(): [number, number][] {
        let d = [];
        d.push([0, 1]);
        d.push([1, 1]);
        d.push([1, 0]);
        d.push([1, (-1 | 0)]);
        d.push([0, (-1 | 0)]);
        d.push([(-1 | 0), (-1 | 0)]);
        d.push([(-1 | 0), 0]);
        d.push([(-1 | 0), 1]);
        return d;
    }
    static encKnights(): [number, number][] {
        let d = [];
        d.push([1, 2]);
        d.push([2, 1]);
        d.push([2, (-1 | 0)]);
        d.push([1, (-2 | 0)]);
        d.push([(-1 | 0), (-2 | 0)]);
        d.push([(-2 | 0), (-1 | 0)]);
        d.push([(-2 | 0), 1]);
        d.push([(-1 | 0), 2]);
        return d;
    }
    clone(): PgChessState {
        let sq = [];
        for (const v of this.squares) {
            sq.push(v);
        }
        return new PgChessState(sq, this.whiteToMove, this.castling, this.enPassant, this.halfmoveClock);
    }
    isSquareAttacked(square: number, byWhite: boolean): boolean {
        const tf = PgChessState.fileOf(square);
        const tr = PgChessState.rankOf(square);
        const pawnDir = (byWhite ? 1 : (-1 | 0));
        const pawn = (byWhite ? 1 : (-1 | 0));
        for (let k = 0; k < 2; k++) {
            const df = (Math.imul(k, 2) - 1 | 0);
            const f = (tf + df | 0);
            const r = (tr - pawnDir | 0);
            if (PgChessState.onBoard(f, r) && this.squares[(Math.imul(r, 8) + f | 0)] === pawn) {
                return true;
            }
        }
        const knight = (byWhite ? 2 : (-2 | 0));
        const kn = PgChessState.knightDeltas();
        for (const [df, dr] of kn) {
            const f = (tf + df | 0);
            const r = (tr + dr | 0);
            if (PgChessState.onBoard(f, r) && this.squares[(Math.imul(r, 8) + f | 0)] === knight) {
                return true;
            }
        }
        const king = (byWhite ? 6 : (-6 | 0));
        const kg = PgChessState.kingDeltas();
        for (const [df, dr] of kg) {
            const f = (tf + df | 0);
            const r = (tr + dr | 0);
            if (PgChessState.onBoard(f, r) && this.squares[(Math.imul(r, 8) + f | 0)] === king) {
                return true;
            }
        }
        if (this.rayHits(tf, tr, PgChessState.bishopDirs(), byWhite, 3)) {
            return true;
        }
        if (this.rayHits(tf, tr, PgChessState.rookDirs(), byWhite, 4)) {
            return true;
        }
        return false;
    }
    rayHits(tf: number, tr: number, dirs: [number, number][], byWhite: boolean, slider: number): boolean {
        for (const [df, dr] of dirs) {
            let found = false;
            let blocked = false;
            for (let step = 1; step < 8; step++) {
                if (blocked) {
                    continue;
                }
                const f = (tf + Math.imul(df, step) | 0);
                const r = (tr + Math.imul(dr, step) | 0);
                if (!PgChessState.onBoard(f, r)) {
                    blocked = true;
                    continue;
                }
                const piece = this.squares[(Math.imul(r, 8) + f | 0)];
                if (piece !== 0) {
                    if (PgChessState.isWhite(piece) === byWhite) {
                        const t = PgChessState.pieceType(piece);
                        if (t === slider || t === 5) {
                            found = true;
                        }
                    }
                    blocked = true;
                }
            }
            if (found) {
                return true;
            }
        }
        return false;
    }
    kingSquare(white: boolean): number {
        const king = (white ? 6 : (-6 | 0));
        for (let sq = 0; sq < 64; sq++) {
            if (this.squares[sq] === king) {
                return sq;
            }
        }
        return (-1 | 0);
    }
    inCheck(white: boolean): boolean {
        return this.isSquareAttacked(this.kingSquare(white), !white);
    }
    legalMoves(): PgChessMove[] {
        let legal = [];
        const pseudo = this.pseudoLegal();
        for (const m of pseudo) {
            const next = this.makeMove(m);
            if (!next.inCheck(this.whiteToMove)) {
                legal.push(m);
            }
        }
        return legal;
    }
    pseudoLegal(): PgChessMove[] {
        let moves = [];
        const white = this.whiteToMove;
        for (let sq = 0; sq < 64; sq++) {
            const piece = this.squares[sq];
            if (piece === 0 || PgChessState.isWhite(piece) !== white) {
                continue;
            }
            const f = PgChessState.fileOf(sq);
            const r = PgChessState.rankOf(sq);
            const t = PgChessState.pieceType(piece);
            if (t === 1) {
                this.pawnMoves(sq, f, r, white, moves);
            } else {
                if (t === 2) {
                    this.stepMoves(sq, f, r, white, PgChessState.knightDeltas(), moves);
                } else {
                    if (t === 6) {
                        this.stepMoves(sq, f, r, white, PgChessState.kingDeltas(), moves);
                        this.castleMoves(white, moves);
                    } else {
                        if (t === 3) {
                            this.slideMoves(sq, f, r, white, PgChessState.bishopDirs(), moves);
                        } else {
                            if (t === 4) {
                                this.slideMoves(sq, f, r, white, PgChessState.rookDirs(), moves);
                            } else {
                                if (t === 5) {
                                    this.slideMoves(sq, f, r, white, PgChessState.bishopDirs(), moves);
                                    this.slideMoves(sq, f, r, white, PgChessState.rookDirs(), moves);
                                }
                            }
                        }
                    }
                }
            }
        }
        return moves;
    }
    stepMoves(sq: number, f: number, r: number, white: boolean, deltas: [number, number][], moves: PgChessMove[]): void {
        for (const [df, dr] of deltas) {
            const nf = (f + df | 0);
            const nr = (r + dr | 0);
            if (!PgChessState.onBoard(nf, nr)) {
                continue;
            }
            const to = (Math.imul(nr, 8) + nf | 0);
            const target = this.squares[to];
            if (target === 0 || PgChessState.isWhite(target) !== white) {
                moves.push(new PgChessMove(sq, to, 0));
            }
        }
    }
    slideMoves(sq: number, f: number, r: number, white: boolean, dirs: [number, number][], moves: PgChessMove[]): void {
        for (const [df, dr] of dirs) {
            let blocked = false;
            for (let step = 1; step < 8; step++) {
                if (blocked) {
                    continue;
                }
                const nf = (f + Math.imul(df, step) | 0);
                const nr = (r + Math.imul(dr, step) | 0);
                if (!PgChessState.onBoard(nf, nr)) {
                    blocked = true;
                    continue;
                }
                const to = (Math.imul(nr, 8) + nf | 0);
                const target = this.squares[to];
                if (target === 0) {
                    moves.push(new PgChessMove(sq, to, 0));
                } else {
                    if (PgChessState.isWhite(target) !== white) {
                        moves.push(new PgChessMove(sq, to, 0));
                    }
                    blocked = true;
                }
            }
        }
    }
    pawnMoves(sq: number, f: number, r: number, white: boolean, moves: PgChessMove[]): void {
        const dir = (white ? 1 : (-1 | 0));
        const startRank = (white ? 1 : 6);
        const lastRank = (white ? 7 : 0);
        const one = (Math.imul((r + dir | 0), 8) + f | 0);
        if (this.squares[one] === 0) {
            this.addPawn(sq, one, (r + dir | 0) === lastRank, moves);
            if (r === startRank) {
                const two = (Math.imul((r + Math.imul(2, dir) | 0), 8) + f | 0);
                if (this.squares[two] === 0) {
                    moves.push(new PgChessMove(sq, two, 0));
                }
            }
        }
        for (let k = 0; k < 2; k++) {
            const df = (Math.imul(k, 2) - 1 | 0);
            const nf = (f + df | 0);
            const nr = (r + dir | 0);
            if (!PgChessState.onBoard(nf, nr)) {
                continue;
            }
            const to = (Math.imul(nr, 8) + nf | 0);
            const target = this.squares[to];
            if (target !== 0 && PgChessState.isWhite(target) !== white) {
                this.addPawn(sq, to, nr === lastRank, moves);
            } else {
                if (to === this.enPassant) {
                    moves.push(new PgChessMove(sq, to, 0));
                }
            }
        }
    }
    addPawn(from: number, to: number, promotion: boolean, moves: PgChessMove[]): void {
        if (promotion) {
            moves.push(new PgChessMove(from, to, PgChessState.PromoQueen));
            moves.push(new PgChessMove(from, to, PgChessState.PromoRook));
            moves.push(new PgChessMove(from, to, PgChessState.PromoBishop));
            moves.push(new PgChessMove(from, to, PgChessState.PromoKnight));
        } else {
            moves.push(new PgChessMove(from, to, 0));
        }
    }
    castleMoves(white: boolean, moves: PgChessMove[]): void {
        const rank = (white ? 0 : 7);
        const kingSq = (Math.imul(rank, 8) + 4 | 0);
        const ownKing = (white ? 6 : (-6 | 0));
        if (this.squares[kingSq] !== ownKing) {
            return;
        }
        if (this.isSquareAttacked(kingSq, !white)) {
            return;
        }
        const kSide = (white ? PgChessState.CastleWK : PgChessState.CastleBK);
        const qSide = (white ? PgChessState.CastleWQ : PgChessState.CastleBQ);
        const kEmpty = this.squares[(Math.imul(rank, 8) + 5 | 0)] === 0 && this.squares[(Math.imul(rank, 8) + 6 | 0)] === 0;
        const kSafe = !this.isSquareAttacked((Math.imul(rank, 8) + 5 | 0), !white) && !this.isSquareAttacked((Math.imul(rank, 8) + 6 | 0), !white);
        if ((this.castling & kSide) !== 0 && kEmpty && kSafe) {
            moves.push(new PgChessMove(kingSq, (Math.imul(rank, 8) + 6 | 0), 0));
        }
        const qEmpty = this.squares[(Math.imul(rank, 8) + 1 | 0)] === 0 && this.squares[(Math.imul(rank, 8) + 2 | 0)] === 0 && this.squares[(Math.imul(rank, 8) + 3 | 0)] === 0;
        const qSafe = !this.isSquareAttacked((Math.imul(rank, 8) + 3 | 0), !white) && !this.isSquareAttacked((Math.imul(rank, 8) + 2 | 0), !white);
        if ((this.castling & qSide) !== 0 && qEmpty && qSafe) {
            moves.push(new PgChessMove(kingSq, (Math.imul(rank, 8) + 2 | 0), 0));
        }
    }
    makeMove(move: PgChessMove): PgChessState {
        let b = [];
        for (const v of this.squares) {
            b.push(v);
        }
        const white = this.whiteToMove;
        const piece = b[move.from];
        const t = PgChessState.pieceType(piece);
        let capture = b[move.to] !== 0;
        let newEp = (-1 | 0);
        let castling = this.castling;
        b[move.from] = 0;
        if (t === 1 && move.to === this.enPassant && this.enPassant >= 0) {
            const capturedSq = (move.to + (white ? (-8 | 0) : 8) | 0);
            b[capturedSq] = 0;
            capture = true;
        }
        if (move.promotion !== 0) {
            b[move.to] = Math.imul((white ? 1 : (-1 | 0)), move.promotion);
        } else {
            b[move.to] = piece;
        }
        if (t === 1 && ((a) => (a < (a - a) ? -a : a))((PgChessState.rankOf(move.to) - PgChessState.rankOf(move.from) | 0)) === 2) {
            newEp = (((move.from + move.to | 0)) / 2 | 0);
        }
        if (t === 6 && ((a) => (a < (a - a) ? -a : a))((PgChessState.fileOf(move.to) - PgChessState.fileOf(move.from) | 0)) === 2) {
            const rank = PgChessState.rankOf(move.from);
            if (PgChessState.fileOf(move.to) === 6) {
                b[(Math.imul(rank, 8) + 5 | 0)] = b[(Math.imul(rank, 8) + 7 | 0)];
                b[(Math.imul(rank, 8) + 7 | 0)] = 0;
            } else {
                b[(Math.imul(rank, 8) + 3 | 0)] = b[(Math.imul(rank, 8) + 0 | 0)];
                b[(Math.imul(rank, 8) + 0 | 0)] = 0;
            }
        }
        if (t === 6) {
            if (white) {
                castling = PgChessState.clearBits(castling, (PgChessState.CastleWK + PgChessState.CastleWQ | 0));
            } else {
                castling = PgChessState.clearBits(castling, (PgChessState.CastleBK + PgChessState.CastleBQ | 0));
            }
        }
        castling = PgChessState.clearBits(castling, PgChessState.cornerRight(move.from));
        castling = PgChessState.clearBits(castling, PgChessState.cornerRight(move.to));
        const halfmove = (t === 1 || capture ? 0 : (this.halfmoveClock + 1 | 0));
        return new PgChessState(b, !white, castling, newEp, halfmove);
    }
    static cornerRight(sq: number): number {
        if (sq === 0) {
            return PgChessState.CastleWQ;
        }
        if (sq === 7) {
            return PgChessState.CastleWK;
        }
        if (sq === 56) {
            return PgChessState.CastleBQ;
        }
        if (sq === 63) {
            return PgChessState.CastleBK;
        }
        return 0;
    }
    isFiftyMove(): boolean {
        return this.halfmoveClock >= 100;
    }
    isInsufficientMaterial(): boolean {
        let knights = 0;
        let bishops = 0;
        for (const p of this.squares) {
            const t = PgChessState.pieceType(p);
            if (t === 0 || t === 6) {
                continue;
            } else {
                if (t === 2) {
                    knights += 1;
                } else {
                    if (t === 3) {
                        bishops += 1;
                    } else {
                        return false;
                    }
                }
            }
        }
        return (knights + bishops | 0) <= 1;
    }
    result(): number {
        if (this.legalMoves().length === 0) {
            return (this.inCheck(this.whiteToMove) ? 1 : 2);
        }
        if (this.isFiftyMove() || this.isInsufficientMaterial()) {
            return 2;
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
    legalMoveIndices(): number[] {
        let out = [];
        const moves = this.legalMoves();
        for (const m of moves) {
            out.push(PgChessState.encode(m));
        }
        return out;
    }
    writeObservation(): number[] {
        let obs = [];
        for (let i = 0; i < 1152; i++) {
            obs.push(0.0);
        }
        for (let sq = 0; sq < 64; sq++) {
            const piece = this.squares[sq];
            if (piece === 0) {
                continue;
            }
            const t = (PgChessState.pieceType(piece) - 1 | 0);
            const plane = (piece > 0 ? t : (6 + t | 0));
            obs[(Math.imul(plane, 64) + sq | 0)] = 1.0;
        }
        if (this.whiteToMove) {
            this.fillPlane(obs, 12);
        }
        if ((this.castling & PgChessState.CastleWK) !== 0) {
            this.fillPlane(obs, 13);
        }
        if ((this.castling & PgChessState.CastleWQ) !== 0) {
            this.fillPlane(obs, 14);
        }
        if ((this.castling & PgChessState.CastleBK) !== 0) {
            this.fillPlane(obs, 15);
        }
        if ((this.castling & PgChessState.CastleBQ) !== 0) {
            this.fillPlane(obs, 16);
        }
        if (this.enPassant >= 0) {
            obs[(Math.imul(17, 64) + this.enPassant | 0)] = 1.0;
        }
        return obs;
    }
    fillPlane(obs: number[], plane: number): void {
        for (let sq = 0; sq < 64; sq++) {
            obs[(Math.imul(plane, 64) + sq | 0)] = 1.0;
        }
    }
    applyIndex(index: number): PgChessState {
        const m = PgChessState.decode(index);
        let chosen = m;
        if (m.promotion === 0 && PgChessState.pieceType(this.squares[m.from]) === 1) {
            const toRank = PgChessState.rankOf(m.to);
            if (toRank === 0 || toRank === 7) {
                chosen = new PgChessMove(m.from, m.to, PgChessState.PromoQueen);
            }
        }
        return this.makeMove(chosen);
    }
    static encode(move: PgChessMove): number {
        const df = (PgChessState.fileOf(move.to) - PgChessState.fileOf(move.from) | 0);
        const dr = (PgChessState.rankOf(move.to) - PgChessState.rankOf(move.from) | 0);
        let plane = 0;
        if (move.promotion === PgChessState.PromoKnight || move.promotion === PgChessState.PromoBishop || move.promotion === PgChessState.PromoRook) {
            const pieceIdx = (move.promotion === PgChessState.PromoKnight ? 0 : (move.promotion === PgChessState.PromoBishop ? 1 : 2));
            plane = ((64 + Math.imul(pieceIdx, 3) | 0) + ((df + 1 | 0)) | 0);
        } else {
            if (PgChessState.isKnightMove(df, dr)) {
                plane = (56 + PgChessState.knightIndex(df, dr) | 0);
            } else {
                const dir = PgChessState.dirIndex(PgChessState.isign(df), PgChessState.isign(dr));
                const dist = ((a, b) => (a >= b ? a : b))(((a) => (a < (a - a) ? -a : a))(df), ((a) => (a < (a - a) ? -a : a))(dr));
                plane = (Math.imul(dir, 7) + ((dist - 1 | 0)) | 0);
            }
        }
        return (Math.imul(move.from, PgChessState.PlanesPerSquare) + plane | 0);
    }
    static decode(index: number): PgChessMove {
        const from = (index / PgChessState.PlanesPerSquare | 0);
        const plane = (index % PgChessState.PlanesPerSquare | 0);
        const f = PgChessState.fileOf(from);
        const r = PgChessState.rankOf(from);
        if (plane >= 64) {
            const p = (plane - 64 | 0);
            const promo = ((p / 3 | 0) === 0 ? PgChessState.PromoKnight : ((p / 3 | 0) === 1 ? PgChessState.PromoBishop : PgChessState.PromoRook));
            const df = ((p % 3 | 0) - 1 | 0);
            const dr = (r === 6 ? 1 : (-1 | 0));
            return new PgChessMove(from, (Math.imul((r + dr | 0), 8) + ((f + df | 0)) | 0), promo);
        }
        if (plane >= 56) {
            const target = (plane - 56 | 0);
            let i = 0;
            let rdf = 0;
            let rdr = 0;
            for (const [df, dr] of PgChessState.encKnights()) {
                if (i === target) {
                    rdf = df;
                    rdr = dr;
                }
                i += 1;
            }
            return new PgChessMove(from, (Math.imul((r + rdr | 0), 8) + ((f + rdf | 0)) | 0), 0);
        }
        const dirTarget = (plane / 7 | 0);
        const dist = ((plane % 7 | 0) + 1 | 0);
        let j = 0;
        let ddf = 0;
        let ddr = 0;
        for (const [df, dr] of PgChessState.encDirs()) {
            if (j === dirTarget) {
                ddf = df;
                ddr = dr;
            }
            j += 1;
        }
        return new PgChessMove(from, (Math.imul((r + Math.imul(ddr, dist) | 0), 8) + ((f + Math.imul(ddf, dist) | 0)) | 0), 0);
    }
    static isKnightMove(df: number, dr: number): boolean {
        const a = ((a) => (a < (a - a) ? -a : a))(df);
        const b = ((a) => (a < (a - a) ? -a : a))(dr);
        return a === 1 && b === 2 || a === 2 && b === 1;
    }
    static knightIndex(df: number, dr: number): number {
        let i = 0;
        let found = (-1 | 0);
        for (const [kdf, kdr] of PgChessState.encKnights()) {
            if (kdf === df && kdr === dr) {
                found = i;
            }
            i += 1;
        }
        return found;
    }
    static dirIndex(sdf: number, sdr: number): number {
        let i = 0;
        let found = (-1 | 0);
        for (const [ddf, ddr] of PgChessState.encDirs()) {
            if (ddf === sdf && ddr === sdr) {
                found = i;
            }
            i += 1;
        }
        return found;
    }
}
export class PgPolicyValueNet {
    inputSize: number;
    actions: number;
    hidden: number[];
    trunkWFlat: number[];
    trunkBFlat: number[];
    policyW: number[];
    policyB: number[];
    valueW: number[];
    valueB: number[];
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
    }
    forward(obs: number[]): PgNetOut {
        let x = obs;
        let prev = this.inputSize;
        let wOff = 0;
        let bOff = 0;
        for (let l = 0; l < this.hidden.length; l++) {
            const h = this.hidden[l];
            x = PgPolicyValueNet.relu(PgPolicyValueNet.linear(x, this.trunkWFlat, wOff, this.trunkBFlat, bOff, prev, h));
            wOff += Math.imul(prev, h);
            bOff += h;
            prev = h;
        }
        const logits = PgPolicyValueNet.linear(x, this.policyW, 0, this.policyB, 0, prev, this.actions);
        const value = PgPolicyValueNet.linear(x, this.valueW, 0, this.valueB, 0, prev, 1);
        return new PgNetOut(logits, value[0]);
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
export class PgMctsNode {
    moves: number[];
    p: number[];
    n: number[];
    w: number[];
    children: PgMctsNode | null[];
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
export class PgChessMcts {
    static search(net: PgPolicyValueNet, root: PgChessState, sims: number, cpuct: number): number[] {
        const rootNode = PgChessMcts.newNode(root);
        if (!rootNode.terminal) {
            PgChessMcts.expandLeaf(rootNode, net, root);
            for (let s = 0; s < sims; s++) {
                PgChessMcts.simulate(rootNode, net, root, cpuct);
            }
        }
        let pi = [];
        for (let i = 0; i < 4672; i++) {
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
    static chooseMove(net: PgPolicyValueNet, root: PgChessState, sims: number, cpuct: number): number {
        const pi = PgChessMcts.search(net, root, sims, cpuct);
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
    static simulate(node: PgMctsNode, net: PgPolicyValueNet, state: PgChessState, cpuct: number): number {
        if (node.terminal) {
            return node.terminalValue;
        }
        if (!node.expanded) {
            return PgChessMcts.expandLeaf(node, net, state);
        }
        const edge = PgChessMcts.selectChild(node, cpuct);
        const childState = state.applyIndex(node.moves[edge]);
        if (node.children[edge] === null) {
            node.children[edge] = PgChessMcts.newNode(childState);
        }
        const child = node.children[edge];
        const value = -PgChessMcts.simulate(child, net, childState, cpuct);
        node.n[edge] = (node.n[edge] + 1 | 0);
        node.w[edge] = node.w[edge] + value;
        return value;
    }
    static selectChild(node: PgMctsNode, cpuct: number): number {
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
    static newNode(state: PgChessState): PgMctsNode {
        const r = state.result();
        if (r !== 0) {
            const tv = (r === 1 ? -1.0 : 0.0);
            return new PgMctsNode([], true, tv);
        }
        return new PgMctsNode(state.legalMoveIndices(), false, 0.0);
    }
    static expandLeaf(node: PgMctsNode, net: PgPolicyValueNet, state: PgChessState): number {
        const ev = PgChessMcts.evaluate(net, state, node.moves);
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
    static evaluate(net: PgPolicyValueNet, state: PgChessState, moves: number[]): PgLeafEval {
        const out = net.forward(state.writeObservation());
        let mx = -1.0e30;
        for (const m of moves) {
            if (out.logits[m] > mx) {
                mx = out.logits[m];
            }
        }
        let pr = [];
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
        return new PgLeafEval(pr, Math.tanh(out.value));
    }
}

export type Option<T> = { tag: "Some"; value: T } | { tag: "None" };
export class PgFruitDef {
    radiusPx: number;
    mergePoints: number;
    constructor(radiusPx: number, mergePoints: number) {
        this.radiusPx = radiusPx;
        this.mergePoints = mergePoints;
    }
    equals(other: PgFruitDef): boolean {
        return this.radiusPx === other.radiusPx && this.mergePoints === other.mergePoints;
    }
}
export class PgContact {
    a: PgFruitBody;
    b: PgFruitBody | null;
    nx: number;
    ny: number;
    pen: number;
    constructor(a: PgFruitBody, b: PgFruitBody | null, nx: number, ny: number, pen: number) {
        this.a = a;
        this.b = b;
        this.nx = nx;
        this.ny = ny;
        this.pen = pen;
    }
    equals(other: PgContact): boolean {
        return this.a === other.a && this.b === other.b && this.nx === other.nx && this.ny === other.ny && this.pen === other.pen;
    }
}
export class PgMergeEvent {
    sourceTier: number;
    resultTier: number | null;
    x: number;
    y: number;
    points: number;
    constructor(sourceTier: number, resultTier: number | null, x: number, y: number, points: number) {
        this.sourceTier = sourceTier;
        this.resultTier = resultTier;
        this.x = x;
        this.y = y;
        this.points = points;
    }
    equals(other: PgMergeEvent): boolean {
        return this.sourceTier === other.sourceTier && this.resultTier === other.resultTier && this.x === other.x && this.y === other.y && this.points === other.points;
    }
}
export class PgFruitBody {
    x: number;
    y: number;
    vx: number;
    vy: number;
    angle: number;
    angularVel: number;
    r: number;
    tier: number;
    invMass: number;
    invI: number;
    pendingMerge: boolean;
    removed: boolean;
    constructor(x: number, y: number, r: number, tier: number, invMass: number, invI: number) {
        this.x = x;
        this.y = y;
        this.r = r;
        this.tier = tier;
        this.invMass = invMass;
        this.invI = invI;
        this.vx = 0.0;
        this.vy = 0.0;
        this.angle = 0.0;
        this.angularVel = 0.0;
        this.pendingMerge = false;
        this.removed = false;
    }
    get speed(): number {
        return Math.sqrt(this.vx * this.vx + this.vy * this.vy);
    }
}
export class PgFruitCakeWorld {
    static readonly Width: number = 620.0;
    static readonly Height: number = 850.0;
    static readonly DangerLineY: number = 150.0;
    static readonly Gravity: number = 9.8 * 64.0;
    static readonly Restitution: number = 0.1;
    static readonly Friction: number = 0.3;
    static readonly RestitutionThresholdPx: number = 64.0;
    static readonly VelocityIterations: number = 12;
    static readonly PositionIterations: number = 4;
    static readonly Slop: number = 0.5;
    static readonly CorrectionPercent: number = 0.8;
    static readonly AngularDamping: number = 0.995;
    rotation: boolean;
    bodies: PgFruitBody[] = [];
    mergeQueue: [PgFruitBody, PgFruitBody][] = [];
    contacts: PgContact[] = [];
    lastMerges: PgMergeEvent[] = [];
    constructor(enableRotation: boolean = false) {
        this.rotation = enableRotation;
    }
    get count(): number {
        return this.bodies.length;
    }
    spawnFruit(tier: number, xPx: number, yPx: number): PgFruitBody {
        const def = byTier(tier);
        const r = def.radiusPx;
        const mass = Math.PI * r * r;
        const inertia = 0.5 * mass * r * r;
        const body = new PgFruitBody(xPx, yPx, r, tier, 1.0 / mass, (this.rotation ? 1.0 / inertia : 0.0));
        this.bodies.push(body);
        return body;
    }
    clear(): void {
        this.bodies = [];
        this.mergeQueue = [];
    }
    step(dt: number): number {
        this.lastMerges = [];
        for (const b of this.bodies) {
            b.vy += PgFruitCakeWorld.Gravity * dt;
        }
        this.buildContacts(true);
        for (let _it = 0; _it < PgFruitCakeWorld.VelocityIterations; _it++) {
            for (const c of this.contacts) {
                PgFruitCakeWorld.resolveVelocity(c);
            }
        }
        for (const b of this.bodies) {
            b.x += b.vx * dt;
            b.y += b.vy * dt;
            if (this.rotation) {
                b.angle += b.angularVel * dt;
                b.angularVel *= PgFruitCakeWorld.AngularDamping;
            }
        }
        for (let _it = 0; _it < PgFruitCakeWorld.PositionIterations; _it++) {
            this.buildContacts(false);
            for (const c of this.contacts) {
                PgFruitCakeWorld.correctPosition(c);
            }
        }
        return this.flushMerges();
    }
    maxSpeed(): number {
        let m = 0.0;
        for (const b of this.bodies) {
            m = ((a, b) => (a >= b ? a : b))(m, b.speed);
        }
        return m;
    }
    settleAfterDrop(settleSpeedPx: number, minSubsteps: number, maxSubsteps: number, dt: number = 1.0 / 60.0): number {
        let points = 0;
        for (let sub = 0; sub < maxSubsteps; sub++) {
            const gained = this.step(dt);
            points += gained;
            if (sub >= minSubsteps && gained === 0 && this.maxSpeed() < settleSpeedPx) {
                break;
            }
        }
        return points;
    }
    clone(enableRotation: boolean): PgFruitCakeWorld {
        const copy = new PgFruitCakeWorld(enableRotation);
        for (const b of this.bodies) {
            const nb = copy.spawnFruit(b.tier, b.x, b.y);
            nb.vx = b.vx;
            nb.vy = b.vy;
            nb.angle = (enableRotation ? b.angle : 0.0);
            nb.angularVel = (enableRotation ? b.angularVel : 0.0);
        }
        return copy;
    }
    anyEjected(): boolean {
        for (const b of this.bodies) {
            if (b.y < 0.0) {
                return true;
            }
        }
        return false;
    }
    anyRestingAboveDangerLine(restSpeedPx: number): boolean {
        for (const b of this.bodies) {
            if (b.y < PgFruitCakeWorld.DangerLineY && b.speed < restSpeedPx) {
                return true;
            }
        }
        return false;
    }
    pileHeight(): number {
        let minTop = PgFruitCakeWorld.Height;
        for (const b of this.bodies) {
            minTop = ((a, b) => (a <= b ? a : b))(minTop, b.y - b.r);
        }
        return PgFruitCakeWorld.Height - minTop;
    }
    buildContacts(detect: boolean): void {
        this.contacts = [];
        for (const b of this.bodies) {
            const left = b.r - b.x;
            if (left > 0.0) {
                this.contacts.push(new PgContact(b, null, -1.0, 0.0, left));
            }
            const right = b.x + b.r - PgFruitCakeWorld.Width;
            if (right > 0.0) {
                this.contacts.push(new PgContact(b, null, 1.0, 0.0, right));
            }
            const floor = b.y + b.r - PgFruitCakeWorld.Height;
            if (floor > 0.0) {
                this.contacts.push(new PgContact(b, null, 0.0, 1.0, floor));
            }
        }
        for (let i = 0; i < this.bodies.length; i++) {
            const a = this.bodies[i];
            for (let j = (i + 1 | 0); j < this.bodies.length; j++) {
                const b = this.bodies[j];
                const dx = b.x - a.x;
                const dy = b.y - a.y;
                const rsum = a.r + b.r;
                const d2 = dx * dx + dy * dy;
                if (d2 >= rsum * rsum) {
                    continue;
                }
                let d = Math.sqrt(d2);
                if (d === 0.0) {
                    d = 0.0001;
                }
                if (detect && a.tier === b.tier && !a.pendingMerge && !b.pendingMerge) {
                    a.pendingMerge = true;
                    b.pendingMerge = true;
                    this.mergeQueue.push([a, b]);
                    continue;
                }
                this.contacts.push(new PgContact(a, b, dx / d, dy / d, rsum - d));
            }
        }
    }
    static resolveVelocity(c: PgContact): void {
        const a = c.a;
        const b = c.b;
        const nx = c.nx;
        const ny = c.ny;
        const invMa = a.invMass;
        const invMb = b?.invMass ?? 0.0;
        const invIa = a.invI;
        const invIb = b?.invI ?? 0.0;
        if (invMa + invMb === 0.0) {
            return;
        }
        const rax = a.r * nx;
        const ray = a.r * ny;
        const rbx = (b === null ? 0.0 : -b.r * nx);
        const rby = (b === null ? 0.0 : -b.r * ny);
        const wa = a.angularVel;
        const wb = b?.angularVel ?? 0.0;
        let rvx = (b === null ? 0.0 : b.vx - wb * rby) - (a.vx - wa * ray);
        let rvy = (b === null ? 0.0 : b.vy + wb * rbx) - (a.vy + wa * rax);
        const relN = rvx * nx + rvy * ny;
        if (relN > 0.0) {
            return;
        }
        const e = (relN < -PgFruitCakeWorld.RestitutionThresholdPx ? PgFruitCakeWorld.Restitution : 0.0);
        const rnA = rax * ny - ray * nx;
        const rnB = rbx * ny - rby * nx;
        const kn = invMa + invMb + invIa * rnA * rnA + invIb * rnB * rnB;
        if (kn <= 0.0) {
            return;
        }
        const jn = -(1.0 + e) * relN / kn;
        PgFruitCakeWorld.applyImpulse(a, b, rax, ray, rbx, rby, jn * nx, jn * ny);
        const wa2 = a.angularVel;
        const wb2 = b?.angularVel ?? 0.0;
        rvx = (b === null ? 0.0 : b.vx - wb2 * rby) - (a.vx - wa2 * ray);
        rvy = (b === null ? 0.0 : b.vy + wb2 * rbx) - (a.vy + wa2 * rax);
        const rvn = rvx * nx + rvy * ny;
        let tx = rvx - rvn * nx;
        let ty = rvy - rvn * ny;
        const tlen = Math.sqrt(tx * tx + ty * ty);
        if (tlen < 0.000001) {
            return;
        }
        tx /= tlen;
        ty /= tlen;
        const rtA = rax * ty - ray * tx;
        const rtB = rbx * ty - rby * tx;
        const kt = invMa + invMb + invIa * rtA * rtA + invIb * rtB * rtB;
        if (kt <= 0.0) {
            return;
        }
        let jt = -(rvx * tx + rvy * ty) / kt;
        const maxJt = PgFruitCakeWorld.Friction * jn;
        jt = ((a, b) => (a <= b ? a : b))(maxJt, ((a, b) => (a >= b ? a : b))(-maxJt, jt));
        PgFruitCakeWorld.applyImpulse(a, b, rax, ray, rbx, rby, jt * tx, jt * ty);
    }
    static applyImpulse(a: PgFruitBody, b: PgFruitBody | null, rax: number, ray: number, rbx: number, rby: number, jx: number, jy: number): void {
        a.vx -= a.invMass * jx;
        a.vy -= a.invMass * jy;
        a.angularVel -= a.invI * (rax * jy - ray * jx);
        if (b !== null) {
            b.vx += b.invMass * jx;
            b.vy += b.invMass * jy;
            b.angularVel += b.invI * (rbx * jy - rby * jx);
        }
    }
    static correctPosition(c: PgContact): void {
        const a = c.a;
        const b = c.b;
        const invSum = a.invMass + (b?.invMass ?? 0.0);
        if (invSum === 0.0) {
            return;
        }
        const corr = ((a, b) => (a >= b ? a : b))(c.pen - PgFruitCakeWorld.Slop, 0.0) / invSum * PgFruitCakeWorld.CorrectionPercent;
        a.x -= a.invMass * corr * c.nx;
        a.y -= a.invMass * corr * c.ny;
        if (b !== null) {
            b.x += b.invMass * corr * c.nx;
            b.y += b.invMass * corr * c.ny;
        }
    }
    flushMerges(): number {
        let points = 0;
        let removed = false;
        for (const [a, b] of this.mergeQueue) {
            if (a.removed || b.removed) {
                continue;
            }
            a.removed = true;
            b.removed = true;
            removed = true;
            const tier = a.tier;
            const cx = (a.x + b.x) * 0.5;
            const cy = (a.y + b.y) * 0.5;
            const rt = mergeResultTier(tier);
            if (rt !== null) {
                this.spawnFruit(rt, cx, cy);
            }
            const mp = byTier(tier).mergePoints;
            points += mp;
            this.lastMerges.push(new PgMergeEvent(tier, rt, cx, cy, mp));
        }
        this.mergeQueue = [];
        if (removed) {
            this.bodies = this.bodies.filter(__e => !(((b) => b.removed)(__e)));
        }
        return points;
    }
}
export let Catalog: PgFruitDef[] = [new PgFruitDef(24.0, 1), new PgFruitDef(32.0, 3), new PgFruitDef(40.0, 6), new PgFruitDef(56.0, 10), new PgFruitDef(64.0, 15), new PgFruitDef(72.0, 21), new PgFruitDef(84.0, 28), new PgFruitDef(96.0, 36), new PgFruitDef(128.0, 45), new PgFruitDef(160.0, 55), new PgFruitDef(192.0, 66)];
export function byTier(tier: number): PgFruitDef {
    return Catalog[(tier - 1 | 0)];
}
export function mergeResultTier(tier: number): number | null {
    return (tier < Catalog.length ? (tier + 1 | 0) : null);
}

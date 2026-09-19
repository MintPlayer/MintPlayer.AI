import { describe, expect, it } from 'vitest';

import { CHESS_ACTIONS, CHESS_INPUT_SIZE, parsePolicyValueNet } from './chess-net';

// Chess' half of the C#/TS checkpoint pair, kind "selfplay-pv". Same drift risk as the dueling-q readers
// (see `snake/snake-net.spec.ts` for the full rationale), but a different layout, and two properties that
// are specific to it and easy to break:
//
//   * there is NO noisy flag and NO inputSize/actions in the stream — the header is magic, kind, version,
//     trunkCount, the widths, and then straight into the parameters. Consuming one stray byte here would
//     misalign every float without raising anything.
//   * the version is range-checked (1..2), unlike the dueling-q readers which accept any version.
//
// The bytes are written from the format description, not from the reader's constants: the magic is emitted
// as the ASCII characters 'R','L','N','C' in on-disk order, lengths use BinaryWriter's 7-bit encoding, and
// every float is exactly representable in float32 so values compare without tolerance.

/** Writes a checkpoint by hand, from the documented layout rather than from the reader's constants. */
class CkptWriter {
  private readonly bytes: number[] = [];

  private scalar(v: number, put: (dv: DataView, value: number) => void): this {
    const u = new Uint8Array(4);
    put(new DataView(u.buffer), v);
    this.bytes.push(...u);
    return this;
  }

  magic(tag = 'RLNC'): this {
    for (const ch of tag) this.bytes.push(ch.charCodeAt(0));
    return this;
  }

  u8(v: number): this { this.bytes.push(v & 0xff); return this; }
  i32(v: number): this { return this.scalar(v, (dv, value) => dv.setInt32(0, value, true)); }

  str(s: string): this {
    const utf8 = new TextEncoder().encode(s);
    let n = utf8.length;
    do {
      const b = n & 0x7f;
      n >>>= 7;
      this.u8(n > 0 ? b | 0x80 : b);
    } while (n > 0);
    this.bytes.push(...utf8);
    return this;
  }

  floats(values: number[]): this {
    this.i32(values.length);
    for (const v of values) this.scalar(v, (dv, value) => dv.setFloat32(0, value, true));
    return this;
  }

  buffer(): ArrayBuffer { return new Uint8Array(this.bytes).buffer as ArrayBuffer; }
}

// A tiny stand-in for the real 1152→…→4672 net, so every count is hand-computable.
const INPUT = 3;
const HIDDEN = [4, 2];
const ACTIONS = 2;

const seq = (tag: number, n: number) => Array.from({ length: n }, (_, i) => tag + i / 4);

const L0W = seq(10, INPUT * HIDDEN[0]);       // 3 × 4 = 12
const L0B = seq(20, HIDDEN[0]);               // 4
const L1W = seq(30, HIDDEN[0] * HIDDEN[1]);   // 4 × 2 = 8
const L1B = seq(40, HIDDEN[1]);               // 2
const POLICY_W = seq(50, HIDDEN[1] * ACTIONS); // 2 × 2 = 4
const POLICY_B = seq(60, ACTIONS);
const VALUE_W = seq(70, HIDDEN[1] * 1);       // 2 × 1 = 2
const VALUE_B = seq(80, 1);

interface CkptOptions { magic?: string; kind?: string; version?: number; }

function pvCkpt(o: CkptOptions = {}): ArrayBuffer {
  const w = new CkptWriter()
    .magic(o.magic)
    .str(o.kind ?? 'selfplay-pv')
    .i32(o.version ?? 2)
    .i32(HIDDEN.length);
  for (const h of HIDDEN) w.i32(h);
  w.floats(L0W).floats(L0B).floats(L1W).floats(L1B)
    .floats(POLICY_W).floats(POLICY_B).floats(VALUE_W).floats(VALUE_B);
  return w.buffer();
}

describe('parsePolicyValueNet', () => {
  it('round-trips a hand-written selfplay-pv checkpoint', () => {
    const net = parsePolicyValueNet(pvCkpt(), INPUT, ACTIONS);

    expect(net.inputSize).toBe(INPUT);
    expect(net.actions).toBe(ACTIONS);
    expect(net.hidden).toEqual(HIDDEN);

    // W,b alternate per trunk layer in the stream; the reader emits one flat W array and one flat b
    // array, concatenated in layer order. Hand-computed: 3×4 + 4×2 weights, 4 + 2 biases.
    expect(net.trunkWFlat).toEqual([...L0W, ...L1W]);
    expect(net.trunkBFlat).toEqual([...L0B, ...L1B]);
    expect(net.trunkWFlat).toHaveLength(20);
    expect(net.trunkBFlat).toHaveLength(6);

    // Head order is policy before value — swapping them would put a 4-long array on a 2-long field.
    expect(net.policyW).toEqual(POLICY_W);
    expect(net.policyB).toEqual(POLICY_B);
    expect(net.valueW).toEqual(VALUE_W);
    expect(net.valueB).toEqual(VALUE_B);

    // This reader builds the MLP tier, never the conv one.
    expect(net.conv).toBeNull();
  });

  it('takes its input and action dimensions from the caller, not from the file', () => {
    // Neither number is stored in the checkpoint, so the same bytes must yield whatever dimensions are
    // asked for, with the weights untouched — and the defaults must be the exported game constants.
    const bytes = pvCkpt();

    const defaults = parsePolicyValueNet(bytes);
    expect(defaults.inputSize).toBe(CHESS_INPUT_SIZE);
    expect(defaults.actions).toBe(CHESS_ACTIONS);

    const overridden = parsePolicyValueNet(bytes, 7, 11);
    expect(overridden.inputSize).toBe(7);
    expect(overridden.actions).toBe(11);
    expect(overridden.trunkWFlat).toEqual(defaults.trunkWFlat);
  });

  it('accepts versions 1 and 2 and rejects anything outside that range', () => {
    expect(parsePolicyValueNet(pvCkpt({ version: 1 }), INPUT, ACTIONS).hidden).toEqual(HIDDEN);
    expect(parsePolicyValueNet(pvCkpt({ version: 2 }), INPUT, ACTIONS).hidden).toEqual(HIDDEN);

    expect(() => parsePolicyValueNet(pvCkpt({ version: 0 }), INPUT, ACTIONS)).toThrow(/version/);
    expect(() => parsePolicyValueNet(pvCkpt({ version: 3 }), INPUT, ACTIONS)).toThrow(/version/);
  });

  it('rejects a buffer whose magic is not RLNC', () => {
    expect(() => parsePolicyValueNet(pvCkpt({ magic: 'RLNX' }), INPUT, ACTIONS)).toThrow(/RLNC/);
  });

  it('rejects a checkpoint written for a different net kind', () => {
    // "dueling-q" is a real kind in this repo, written by the same C# writer — rejecting it is what
    // keeps a mis-wired model path from loading silently.
    expect(() => parsePolicyValueNet(pvCkpt({ kind: 'dueling-q' }), INPUT, ACTIONS)).toThrow(/kind/);
  });

  it('throws rather than returning a short net when the buffer is truncated', () => {
    const full = pvCkpt();
    expect(() => parsePolicyValueNet(full.slice(0, full.byteLength - 8), INPUT, ACTIONS)).toThrow();
  });
});

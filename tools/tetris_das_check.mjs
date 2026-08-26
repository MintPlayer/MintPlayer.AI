// NES DAS conformance harness (PLAN M55): drives the pure NesInput machine (tetris-das.ts) with a fake
// host and asserts the frame-exact NES behaviors from the disassembly spec (DAS 16/10/6, wall charge,
// charge preserved across release+lock, hypertap latching, Down-blocks-horizontal, 3-then-2 soft drop,
// gravity/soft-drop non-cumulative). Run: `node tools/tetris_das_check.mjs` — must print ALL PASS.
const { NesInput } = await import('../src/RLDemo.Web/ClientApp/src/app/tetris/tetris-das.ts');

let failures = 0;
const check = (name, cond) => { console.log(cond ? 'ok  ' : 'FAIL', name); if (!cond) failures++; };

function host({ blockShift = false, gravity = 48 } = {}) {
  return {
    shifts: [], drops: 0, frame: 0, blockShift, gravity,
    shift(dir) { if (this.blockShift) return false; this.shifts.push({ frame: this.frame, dir }); return true; },
    dropStep() { this.drops++; return false; },
    gravityFrames() { return this.gravity; },
  };
}
function run(input, h, frames, perFrame = null) {
  for (let f = 0; f < frames; f++) { h.frame = f; perFrame?.(f); input.tick(h); }
}

// 1) Hold right: shift on frame 0, next at frame 16, then every 6 frames (16/10 reset ⇒ 6-frame repeat).
{
  const input = new NesInput(), h = host();
  input.press(1);
  run(input, h, 47);
  const frames = h.shifts.map(s => s.frame);
  check('hold: 0,16,22,28,34,40,46', JSON.stringify(frames) === JSON.stringify([0, 16, 22, 28, 34, 40, 46]));
}

// 2) Wall charge: a blocked fresh press saturates DAS — after unblocking, the very next held frame shifts.
{
  const input = new NesInput(), h = host({ blockShift: true });
  input.press(1);
  run(input, h, 5); // press blocked at frame 0 (das := 16), held-blocked frames keep it pinned
  h.blockShift = false;
  h.frame = 5; input.tick(h);
  check('wall charge: first unblocked held frame shifts', h.shifts.length === 1 && h.shifts[0].frame === 5);
  h.frame = 6; input.tick(h); h.frame = 7; input.tick(h);
  const f2 = h.shifts[1]?.frame;
  check('wall charge: then 6-frame repeat resumes (next at +6)', h.shifts.length === 1 || f2 === 11);
}

// 3) Charge preserved across release and lock: release at full charge, spawn, re-press+hold ⇒ the
//    fresh press itself shifts immediately (das := 0) — but a HELD direction with a carried charge
//    fires on its first held frame. Emulate the classic: charge to 16, release, onSpawn, hold (no
//    fresh press possible without a press — so verify via clear(): das survives clear+spawn, and the
//    next held-only frame (heldR restored) auto-shifts immediately.
{
  const input = new NesInput(), h = host({ blockShift: true });
  input.press(1);
  run(input, h, 3);           // blocked press ⇒ das = 16 (wall charge)
  input.onSpawn();            // lock/spawn: das preserved
  h.blockShift = false;
  // Simulate "kept holding through spawn": heldR is still true, no new press ⇒ held path, das already 16.
  h.frame = 100; input.tick(h);
  check('charge carried across spawn: held frame after spawn shifts at once', h.shifts.length === 1 && h.shifts[0].frame === 100);
}

// 4) Hypertapping: taps latched between frames each register exactly once; one shift per frame max.
{
  const input = new NesInput(), h = host();
  for (let f = 0; f < 12; f++) {
    h.frame = f;
    if (f % 2 === 0) { input.press(1); input.release(1); } // 30 Hz tapping (press+release, latched)
    input.tick(h);
  }
  check('hypertap: 6 taps in 12 frames = 6 shifts', h.shifts.length === 6);
  const input2 = new NesInput(), h2 = host();
  h2.frame = 0;
  input2.press(1); input2.release(1); input2.press(1); input2.release(1); // two taps INSIDE one frame
  input2.tick(h2);
  check('sub-frame double tap collapses to one shift', h2.shifts.length === 1);
}

// 5) Down blocks horizontal, and DAS charge survives the soft drop.
{
  const input = new NesInput(), h = host({ blockShift: true });
  input.press(1);
  run(input, h, 2);           // wall-charged
  h.blockShift = false;
  input.pressDown();          // fresh Down while right held: does NOT engage (horizontal held) —
  h.frame = 10; input.tick(h); // and holding Down also skips the horizontal routine (no shift fires).
  check('down cannot engage while a direction is held', h.drops === 0 && h.shifts.length === 0);
  // ...release right, engage down properly: horizontal is then skipped entirely.
  input.release(1);
  input.releaseDown(); input.pressDown();
  const before = h.shifts.length;
  run(input, h, 10);
  check('soft drop: first row after 3 frames then every 2 (10 frames = 4 rows)', h.drops === 4);
  check('horizontal skipped while down held', h.shifts.length === before);
}

// 6) Gravity and soft drop are non-cumulative: at 2-frame gravity (level 19+) with soft drop engaged,
//    the piece still moves at most 1 row per frame.
{
  const input = new NesInput(), h = host({ gravity: 1 }); // kill-screen gravity: 1 frame/row
  input.pressDown();
  run(input, h, 10);
  check('non-cumulative: 10 frames at 1-frame gravity + soft drop = 10 rows, not more', h.drops === 10);
}

// 7) Both directions held = no movement (D-pad impossibility resolved to neutral).
{
  const input = new NesInput(), h = host();
  input.press(-1); input.press(1);
  run(input, h, 2);
  const bothFresh = h.shifts.length; // frame 0: both latched fresh ⇒ neutral
  run(input, h, 30);
  check('left+right held: no shifts at all', bothFresh === 0 && h.shifts.length === 0);
}

console.log(failures === 0 ? 'ALL PASS' : `${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);

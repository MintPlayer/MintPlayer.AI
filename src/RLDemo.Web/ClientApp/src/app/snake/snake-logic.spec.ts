import { describe, expect, it } from 'vitest';

import { SIZE, SnakeGame, type Dir } from './snake-logic';

// M63.x — the client-side Snake engine that human play runs on.
//
// It exists to mirror the server's `SnakeEnv` rules exactly, so that a human and the AI are playing
// the same game. The two rules worth guarding are the ones a naive port gets wrong: the 180°
// reversal is IGNORED rather than fatal, and stepping onto the cell the tail is vacating is LEGAL
// (unless the snake is growing that step, in which case the tail stays put and it is fatal).
//
// Food placement is random, so every fixture below either pins `food` to a cell out of the way or
// asserts a property of the spawn rather than its location. No DOM, no Angular, no timers.

const UP: Dir = 0, DOWN: Dir = 1, LEFT: Dir = 2, RIGHT: Dir = 3;

/** Board-local helpers for the default board edge. */
const cell = (row: number, col: number): number => row * SIZE + col;
const rowOf = (index: number): number => Math.floor(index / SIZE);
const colOf = (index: number): number => index % SIZE;

/** Replaces the whole snake, so a fixture can express a shape `reset()` never produces. */
function place(game: SnakeGame, body: number[], food: number): void {
  game.body = [...body];
  game.occupied = new Set(body);
  game.food = food;
}

describe('SnakeGame board size', () => {
  it('refuses a board too small to hold its own starting snake', () => {
    // reset() seeds a three-cell snake by walking the head column down by two, so a narrower board
    // yields negative cells and a body longer than the board — a corrupt game, not an error. The UI
    // clamps to MIN_SIZE = 6, so this guards direct construction only.
    expect(() => new SnakeGame(2)).toThrow(/at least 3/);
    expect(() => new SnakeGame(0)).toThrow(/at least 3/);
    expect(() => new SnakeGame(4.5)).toThrow(/integer/);
  });

  it('accepts the smallest board that works', () => {
    expect(() => new SnakeGame(3)).not.toThrow();
  });
});

describe('reset', () => {
  it('starts as a three-cell snake lying along one row, facing right', () => {
    const game = new SnakeGame();

    expect(game.body).toHaveLength(3);
    expect(game.occupied.size).toBe(3);
    expect(game.body.every(c => rowOf(c) === rowOf(game.body[0]))).toBe(true);
    // Head first, each further segment one column to the left of the previous one.
    expect(game.body.map(colOf)).toEqual([0, 1, 2].map(i => colOf(game.body[0]) - i));
    expect(game.dead).toBe(false);
    expect(game.foodEaten).toBe(0);
  });

  it('spawns food on a free cell of the board', () => {
    const game = new SnakeGame();

    expect(game.food).toBeGreaterThanOrEqual(0);
    expect(game.food).toBeLessThan(SIZE * SIZE);
    expect(game.occupied.has(game.food)).toBe(false);
  });

  it('honours a custom board edge', () => {
    const game = new SnakeGame(8);

    expect(game.body).toHaveLength(3);
    expect(game.food).toBeLessThan(8 * 8);
    expect(game.body.every(c => c < 8 * 8)).toBe(true);
  });
});

describe('setDirection', () => {
  it('ignores the 180° reversal onto the neck', () => {
    const game = new SnakeGame();
    const head = game.body[0];
    game.food = 0; // out of the way of everything this test touches

    game.setDirection(LEFT); // the snake is facing right, so this is the reversal
    game.tick();

    expect(game.dead).toBe(false);
    expect(game.body[0]).toBe(head + 1); // still travelling right
  });

  it('accepts a perpendicular turn', () => {
    const game = new SnakeGame();
    const head = game.body[0];
    game.food = 0;

    game.setDirection(UP);
    game.tick();

    expect(game.body[0]).toBe(head - SIZE);
    expect(game.dead).toBe(false);
  });
});

describe('tick', () => {
  it('advances without growing while there is no food to eat', () => {
    const game = new SnakeGame();
    game.food = 0;
    const tail = game.body[game.body.length - 1];

    game.tick();

    expect(game.body).toHaveLength(3);
    expect(game.occupied.size).toBe(3);
    expect(game.occupied.has(tail)).toBe(false); // the vacated tail is released
    expect(new Set(game.body)).toEqual(game.occupied); // the two views never drift apart
  });

  it('grows and scores when the head reaches the food, then respawns it elsewhere', () => {
    const game = new SnakeGame();
    game.food = game.body[0] + 1; // directly ahead

    game.tick();

    expect(game.foodEaten).toBe(1);
    expect(game.body).toHaveLength(4);
    expect(new Set(game.body)).toEqual(game.occupied);
    expect(game.occupied.has(game.food)).toBe(false); // the new food is on a free cell
    expect(game.dead).toBe(false);
  });

  it('dies on the wall, and only on the step that leaves the board', () => {
    const game = new SnakeGame();
    game.food = 0;
    const startRow = rowOf(game.body[0]);
    game.setDirection(UP);

    for (let step = 0; step < startRow; step++) game.tick();
    expect(game.dead).toBe(false);
    expect(rowOf(game.body[0])).toBe(0); // pressed against the top edge

    game.tick();
    expect(game.dead).toBe(true);
  });

  it('is inert once dead', () => {
    const game = new SnakeGame();
    game.food = 0;
    const startRow = rowOf(game.body[0]);
    game.setDirection(UP);
    for (let step = 0; step <= startRow; step++) game.tick();
    expect(game.dead).toBe(true);

    const body = [...game.body];
    game.tick();

    expect(game.body).toEqual(body);
  });

  it('dies when the head runs into its own body', () => {
    const game = new SnakeGame();
    // A hook shape: head at (5,5), then left, down, right, right — so the cell below the head is
    // occupied by a middle segment, not by the neck and not by the tail.
    place(game, [cell(5, 5), cell(5, 4), cell(6, 4), cell(6, 5), cell(6, 6)], cell(0, 0));

    game.setDirection(DOWN);
    game.tick();

    expect(game.dead).toBe(true);
  });

  it('survives stepping onto the cell the tail is vacating', () => {
    const game = new SnakeGame();
    // A closed square: head at (5,5) and the TAIL at (5,6), the very cell the head is about to
    // enter. The tail moves out of the way on the same step, so this is legal.
    place(game, [cell(5, 5), cell(6, 5), cell(6, 6), cell(5, 6)], cell(0, 0));

    game.setDirection(RIGHT);
    game.tick();

    expect(game.dead).toBe(false);
    expect(game.body[0]).toBe(cell(5, 6));
    expect(game.body).toHaveLength(4);
    expect(new Set(game.body)).toEqual(game.occupied); // the tail-follow case is where this drifts
  });

  it('dies stepping onto the tail when it is growing that same step', () => {
    const game = new SnakeGame();
    // Same square, but the food sits on the tail cell: eating keeps the tail in place, so the cell
    // the head wants is still occupied when it arrives.
    place(game, [cell(5, 5), cell(6, 5), cell(6, 6), cell(5, 6)], cell(5, 6));

    game.setDirection(RIGHT);
    game.tick();

    expect(game.dead).toBe(true);
    expect(game.foodEaten).toBe(0);
  });
});

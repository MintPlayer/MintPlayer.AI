"""Author the shipped Lunar Lockout level ladder, and prove every entry with an exhaustive BFS.

Why this exists: the level pack inherited from the WebGames port was unusable — 13 of its 15 grids had no
legal first move at all (no two robots shared a row or column), and one more disagreed with its authored
move count. See docs/prd/WEBGAMES_RETIREMENT_PRD.md §5.3. So the shipped ladder is generated here instead,
and every entry's OptimalMoves is a BFS result rather than a hand-authored guess.

Deterministic: fixed seed, so re-running reproduces the committed ladder byte for byte.

    python tools/lunarlockout_levels.py

Rules implemented here are the published ThinkFun rules, written independently of the .pg engine on purpose
— LunarLockoutOracleTests re-derives every OptimalMoves through the .pg oracle, so the two implementations
cross-check each other.
"""
from __future__ import annotations

import random
from collections import deque

SIZE = 5
CENTRE = (2, 2)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SEED = 58  # M58


def slide(pos, d, occupied):
    """Landing cell for a slide, or None when illegal (nothing ahead, or already resting against a robot)."""
    dr, dc = DIRS[d]
    r, c = pos
    blocked = False
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < SIZE and 0 <= nc < SIZE):
            break  # the edge is NOT a backstop
        if (nr, nc) in occupied:
            blocked = True
            break
        r, c = nr, nc
    if not blocked or (r, c) == pos:
        return None
    return (r, c)


def analyse(target, helpers):
    """(optimal_moves, states_explored, robots_used_on_one_optimal_path). optimal is -1 when unsolvable."""
    start = (target, tuple(sorted(helpers)))
    prev: dict = {start: None}
    dist = {start: 0}
    q = deque([start])
    goal = None
    while q:
        state = q.popleft()
        tgt, helps = state
        if tgt == CENTRE:
            goal = state
            break
        occupied = {tgt} | set(helps)
        for i in range(1 + len(helps)):
            pos = tgt if i == 0 else helps[i - 1]
            for d in range(4):
                land = slide(pos, d, occupied)
                if land is None:
                    continue
                if i == 0:
                    nxt = (land, helps)
                else:
                    lst = list(helps)
                    lst[i - 1] = land
                    nxt = (tgt, tuple(sorted(lst)))
                if nxt not in dist:
                    dist[nxt] = dist[state] + 1
                    prev[nxt] = (state, i)
                    q.append(nxt)
    if goal is None:
        return -1, len(dist), 0

    movers, cur = set(), goal
    while prev[cur] is not None:
        parent, mover = prev[cur]
        movers.add(mover)
        cur = parent
    return dist[goal], len(dist), len(movers)


def canonical(target, helpers):
    """Canonical form under the 8 symmetries of the square (all of which fix the centre goal), so the ladder
    never ships two rotations of the same puzzle."""
    forms = []
    for flip in (False, True):
        for rot in range(4):
            def xf(p):
                r, c = p
                if flip:
                    c = SIZE - 1 - c
                for _ in range(rot):
                    r, c = c, SIZE - 1 - r
                return (r, c)
            forms.append((xf(target), tuple(sorted(xf(h) for h in helpers))))
    return min(forms)


def to_grid(target, helpers):
    rows = [['.'] * SIZE for _ in range(SIZE)]
    rows[target[0]][target[1]] = 'R'
    for i, (r, c) in enumerate(sorted(helpers)):
        rows[r][c] = 'BGYPO'[i % 5]
    return [''.join(r) for r in rows]


def main():
    rng = random.Random(SEED)
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]

    # Bucket candidate puzzles by optimal length. Require genuine interaction: from level 3 up, an optimal
    # solution must move more than one robot, otherwise the puzzle is just "slide the red robot".
    buckets: dict[int, list] = {}
    seen: set = set()
    for _ in range(400_000):
        k = rng.choice([2, 2, 3, 3, 3, 4, 4])
        picks = rng.sample(cells, k + 1)
        target, helpers = picks[0], picks[1:]
        if target == CENTRE:
            continue
        form = canonical(target, helpers)
        if form in seen:
            continue
        seen.add(form)

        optimal, states, movers = analyse(target, helpers)
        if optimal < 1 or optimal > 12:
            continue
        if optimal >= 3 and movers < 2:
            continue
        if optimal >= 5 and states < 50:
            continue
        buckets.setdefault(optimal, []).append((states, target, tuple(helpers), len(helpers)))

    # One level per optimal length, easiest first. Difficulty must ramp on BOARD complexity as well as solution
    # length, so each rung asks for a target helper count (2 robots early, 5 late) and only then prefers the
    # richest state graph — more ways to go wrong makes a more interesting puzzle. Falling back to any available
    # count keeps the ladder complete if a rung has no candidate at the preferred width.
    def preferred_helpers(optimal: int) -> int:
        if optimal <= 2:
            return 2
        if optimal <= 5:
            return 3
        return 4

    ladder = []
    for optimal in sorted(buckets):
        want = preferred_helpers(optimal)
        candidates = [t for t in buckets[optimal] if t[3] == want] or buckets[optimal]
        states, target, helpers, _ = sorted(candidates, key=lambda t: -t[0])[0]
        ladder.append((optimal, target, helpers, states))

    print(f'{"name":16s} {"optimal":>7} {"robots":>6} {"states":>7}')
    levels_out = []
    for i, (optimal, target, helpers, states) in enumerate(ladder, 1):
        name = f'Level {i}'
        print(f'{name:16s} {optimal:>7} {1 + len(helpers):>6} {states:>7}')
        levels_out.append({'name': name, 'grid': to_grid(target, helpers), 'optimalMoves': optimal})

    doc = {
        '_generator': 'tools/lunarlockout_levels.py (deterministic, seed 58) — do not hand-edit; re-run the tool.',
        '_format': 'Five rows, top row first. R = target robot, . = empty, any other letter = helper robot.',
        '_verified': 'Every optimalMoves is an exhaustive-BFS result, not a hand-authored guess. '
                     'LunarLockoutOracleTests re-derives all of them through the .pg oracle as a cross-check.',
        '_provenance': 'The level pack inherited from the WebGames port was unusable — 13 of its 15 grids had no '
                       'legal first move at all, because no two robots shared a row or column — so this ladder '
                       'is authored here instead. See docs/prd/WEBGAMES_RETIREMENT_PRD.md section 5.3.',
        'version': 1,
        'levels': levels_out,
    }

    # Canonical, single source of truth: embedded into the Environments assembly for training AND linked into
    # wwwroot/levels for the browser, so the C# campaign and the Angular app read the SAME bytes.
    dest = (r'C:\Repos\MintPlayer.AI\src\MintPlayer.AI.ReinforcementLearning.Environments'
            r'\LunarLockout\levels\lunarlockout-levels.json')
    import json
    import os
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w', newline='\n', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'\nwrote {len(ladder)} levels -> {dest}')


if __name__ == '__main__':
    main()

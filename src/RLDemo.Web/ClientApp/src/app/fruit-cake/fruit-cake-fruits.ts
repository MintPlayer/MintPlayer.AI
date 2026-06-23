/** One link in the merge chain. Tier is 1-based (1 = cherry … 11 = watermelon). `color` is packed 0xAARRGGBB. */
export interface FruitDef {
  tier: number;
  name: string;
  radiusPx: number;
  color: number;
  droppable: boolean;
  mergePoints: number;
}

/**
 * The canonical 11-fruit Suika chain (radii, colors, scores) and the merge rules over it —
 * the single source of truth physics, rendering, and scoring all read from. Only tiers 1–5 are
 * player-droppable; 6–11 exist only as merge products. Score series is the triangular numbers.
 */
export const FRUITS: readonly FruitDef[] = [
  { tier: 1, name: 'Cherry', radiusPx: 24, color: 0xffe8273a, droppable: true, mergePoints: 1 },
  { tier: 2, name: 'Strawberry', radiusPx: 32, color: 0xfff4607a, droppable: true, mergePoints: 3 },
  { tier: 3, name: 'Grape', radiusPx: 40, color: 0xff8b4fbe, droppable: true, mergePoints: 6 },
  { tier: 4, name: 'Dekopon', radiusPx: 56, color: 0xfff97316, droppable: true, mergePoints: 10 },
  { tier: 5, name: 'Persimmon', radiusPx: 64, color: 0xffd4752a, droppable: true, mergePoints: 15 },
  { tier: 6, name: 'Apple', radiusPx: 72, color: 0xffdc2626, droppable: false, mergePoints: 21 },
  { tier: 7, name: 'Pear', radiusPx: 84, color: 0xffd4c547, droppable: false, mergePoints: 28 },
  { tier: 8, name: 'Peach', radiusPx: 96, color: 0xfff9a06a, droppable: false, mergePoints: 36 },
  { tier: 9, name: 'Pineapple', radiusPx: 128, color: 0xffd4af37, droppable: false, mergePoints: 45 },
  { tier: 10, name: 'Melon', radiusPx: 160, color: 0xffa8d55a, droppable: false, mergePoints: 55 },
  { tier: 11, name: 'Watermelon', radiusPx: 192, color: 0xff3db560, droppable: false, mergePoints: 66 },
];

/** Highest tier a player may drop; tiers above this only appear as merge products. */
export const MAX_DROPPABLE_TIER = 5;
/** Top tier (watermelon) — a pair of these vanishes instead of producing a new fruit. */
export const TOP_TIER = FRUITS.length;

export const DROPPABLE: readonly FruitDef[] = FRUITS.filter(f => f.droppable);

export function byTier(tier: number): FruitDef {
  return FRUITS[tier - 1];
}

/**
 * Result of merging two fruit of `tier`: the next tier up, or `null` when two top-tier fruit
 * merge (both vanish). Defining the top-tier case as null keeps the "watermelon pair disappears"
 * rule out of every caller's path.
 */
export function mergeResultTier(tier: number): number | null {
  return tier >= TOP_TIER ? null : tier + 1;
}

/** A visual skin: just a backdrop and a container-wall color (fruit keep their natural colors). */
export interface FruitTheme {
  name: string;
  background: number;
  wall: number;
}

/** The available skins. Index 0 is the default. */
export const THEMES: readonly FruitTheme[] = [
  { name: 'Classic', background: 0xff1c0b2b, wall: 0xff3d1f5c },
  { name: 'Candy', background: 0xff2a1a3a, wall: 0xff5a3f7c },
  { name: 'Mono', background: 0xff161616, wall: 0xff444444 },
];

/** A packed 0xAARRGGBB color as a CSS `rgba(...)`. `alpha` (0..255) overrides the packed alpha. */
export function cssColor(packed: number, alpha?: number): string {
  const a = alpha ?? (packed >>> 24) & 0xff;
  const r = (packed >>> 16) & 0xff;
  const g = (packed >>> 8) & 0xff;
  const b = packed & 0xff;
  return `rgba(${r},${g},${b},${(a / 255).toFixed(3)})`;
}

import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', loadComponent: () => import('./home/home').then(m => m.Home) },
  { path: 'rushhour', loadComponent: () => import('./rush-hour/rush-hour').then(m => m.RushHour) },
  { path: 'lunarlockout', loadComponent: () => import('./lunar-lockout/lunar-lockout').then(m => m.LunarLockout) },
  { path: 'blockdude', loadComponent: () => import('./block-dude/block-dude').then(m => m.BlockDude) },
  { path: '2048', loadComponent: () => import('./game-2048/game-2048').then(m => m.Game2048) },
  { path: 'cube', loadComponent: () => import('./cube/cube').then(m => m.Cube) },
  { path: 'snake', loadComponent: () => import('./snake/snake').then(m => m.Snake) },
  { path: 'mountaincar', loadComponent: () => import('./mountaincar/mountaincar').then(m => m.MountainCar) },
  { path: 'fruitcake', redirectTo: 'suika' }, // pre-rename URL — keep old links working
  { path: 'suika', loadComponent: () => import('./suika/suika').then(m => m.Suika) },
  { path: 'crazyfruits', loadComponent: () => import('./crazy-fruits/crazy-fruits').then(m => m.CrazyFruits) },
  { path: 'tetris', loadComponent: () => import('./tetris/tetris').then(m => m.Tetris) },
  { path: 'chess', loadComponent: () => import('./chess/chess').then(m => m.Chess) },
  { path: 'draughts', loadComponent: () => import('./draughts/draughts').then(m => m.Draughts) },
  { path: 'gallery', loadComponent: () => import('./gallery/gallery').then(m => m.Gallery) },
  { path: '**', redirectTo: '' },
];

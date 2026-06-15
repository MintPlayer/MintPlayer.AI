import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', loadComponent: () => import('./home/home').then(m => m.Home) },
  { path: 'rushhour', loadComponent: () => import('./rush-hour/rush-hour').then(m => m.RushHour) },
  { path: '2048', loadComponent: () => import('./game-2048/game-2048').then(m => m.Game2048) },
  { path: 'cube', loadComponent: () => import('./cube/cube').then(m => m.Cube) },
  { path: 'snake', loadComponent: () => import('./snake/snake').then(m => m.Snake) },
  { path: 'mountaincar', loadComponent: () => import('./mountaincar/mountaincar').then(m => m.MountainCar) },
  { path: 'gallery', loadComponent: () => import('./gallery/gallery').then(m => m.Gallery) },
  { path: '**', redirectTo: '' },
];

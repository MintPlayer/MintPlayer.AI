import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', loadComponent: () => import('./home/home').then(m => m.Home) },
  { path: 'rushhour', loadComponent: () => import('./rush-hour/rush-hour').then(m => m.RushHour) },
  { path: '**', redirectTo: '' },
];

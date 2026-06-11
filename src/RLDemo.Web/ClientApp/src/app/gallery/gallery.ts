import { DatePipe } from '@angular/common';
import { Component, signal } from '@angular/core';
import { RouterLink } from '@angular/router';

interface GalleryItem {
  id: string;
  game: string;
  createdUtc: string;
  summary: string;
}

@Component({
  selector: 'app-gallery',
  imports: [DatePipe, RouterLink],
  template: `
    <h1>Gallery</h1>
    <p class="subtitle">
      Every board submitted to the AI, with its solution — click one to replay it.
    </p>

    @if (loaded() && items().length === 0) {
      <p class="empty">Nothing here yet. Draw a puzzle and let the AI solve it!</p>
    }

    <div class="entries">
      @for (item of items(); track item.id) {
        <a class="entry" [routerLink]="['/', item.game === 'rushhour' ? 'rushhour' : '2048']"
           [queryParams]="{ replay: item.id }">
          <span class="game" [class.rushhour]="item.game === 'rushhour'">
            {{ item.game === 'rushhour' ? '🚗 Rush Hour' : '🔢 2048' }}
          </span>
          <span class="summary">{{ item.summary }}</span>
          <span class="date">{{ item.createdUtc | date: 'medium' }}</span>
        </a>
      }
    </div>
  `,
  styles: `
    .subtitle {
      color: #aab2c5;
      margin-top: -0.5rem;
    }

    .empty {
      color: #5b6378;
      margin-top: 2rem;
    }

    .entries {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      margin-top: 1rem;
    }

    .entry {
      display: grid;
      grid-template-columns: 9rem 1fr auto;
      gap: 1rem;
      align-items: center;
      padding: 0.7rem 1rem;
      background: #1a1f2b;
      border: 1px solid #2b3245;
      border-radius: 10px;
      color: inherit;
      text-decoration: none;
      transition: border-color 0.15s;

      &:hover {
        border-color: #6ea8fe;
      }

      .game {
        font-weight: 600;
        color: #fbbf24;

        &.rushhour {
          color: #f87171;
        }
      }

      .summary {
        color: #aab2c5;
      }

      .date {
        color: #5b6378;
        font-size: 0.85rem;
        white-space: nowrap;
      }
    }
  `,
})
export class Gallery {
  protected readonly items = signal<GalleryItem[]>([]);
  protected readonly loaded = signal(false);

  constructor() {
    void (async () => {
      const response = await fetch('/api/gallery');
      this.items.set(await response.json());
      this.loaded.set(true);
    })();
  }
}

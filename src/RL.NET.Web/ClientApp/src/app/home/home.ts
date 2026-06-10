import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-home',
  imports: [RouterLink],
  template: `
    <h1>RL.NET Playground</h1>
    <p class="intro">
      Draw a puzzle, try it yourself, then let an agent trained with <strong>RL.NET</strong> —
      a reinforcement-learning library written from scratch in C# — solve it for you.
    </p>

    <div class="cards">
      <a class="card" routerLink="/rushhour">
        <h2>Rush Hour</h2>
        <p>
          Slide the cars to free the red one. Drawn puzzles are solved by a masked
          Double DQN and compared against the BFS-optimal solution.
        </p>
        <span class="cta">Play →</span>
      </a>

      <a class="card" routerLink="/2048">
        <h2>2048</h2>
        <p>
          Set up a board and let the n-tuple network (84% win rate) play it out,
          move by move — same board, same playout, every time.
        </p>
        <span class="cta">Play →</span>
      </a>

      <a class="card" routerLink="/gallery">
        <h2>Gallery</h2>
        <p>
          Every board anyone submitted to the AI, with its solution — browse
          and replay them.
        </p>
        <span class="cta">Browse →</span>
      </a>
    </div>
  `,
  styles: `
    .intro {
      color: #aab2c5;
      max-width: 46rem;
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.25rem;
      margin-top: 1.5rem;
    }

    .card {
      display: block;
      padding: 1.25rem 1.5rem;
      background: #1a1f2b;
      border: 1px solid #2b3245;
      border-radius: 12px;
      color: inherit;
      text-decoration: none;
      transition: border-color 0.15s, transform 0.15s;

      h2 {
        margin: 0 0 0.5rem;
      }

      p {
        color: #aab2c5;
        min-height: 4.2em;
      }

      .cta {
        color: #6ea8fe;
        font-weight: 600;
      }

      &:not(.disabled):hover {
        border-color: #6ea8fe;
        transform: translateY(-2px);
      }

      &.disabled {
        opacity: 0.55;

        .cta {
          color: #5b6378;
        }
      }
    }
  `,
})
export class Home {}

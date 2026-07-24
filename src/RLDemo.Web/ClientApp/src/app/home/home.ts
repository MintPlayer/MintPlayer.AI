import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-home',
  imports: [RouterLink],
  template: `
    <h1>MintPlayer.AI Playground</h1>
    <p class="intro">
      Draw a puzzle, try it yourself, then let an agent trained with
      <strong>MintPlayer.AI.ReinforcementLearning</strong> — a reinforcement-learning
      library written from scratch in C# — solve it for you.
    </p>

    <div class="cards">
      <a class="card" routerLink="/rushhour">
        <h2>Rush Hour</h2>
        <p>
          Slide the cars to free the red one. Drawn puzzles are solved by the trained
          AI (with lookahead when needed) and compared against the BFS-optimal solution.
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

      <a class="card" routerLink="/cube">
        <h2>Rubik's Cube</h2>
        <p>
          Scramble a 3D cube and watch Kociemba's two-phase algorithm solve it
          in 22 moves or fewer, step by step.
        </p>
        <span class="cta">Play →</span>
      </a>

      <a class="card" routerLink="/snake">
        <h2>Snake</h2>
        <p>
          A masked Double + Dueling DQN that learned Snake from scratch. Watch it play
          or play it yourself — both run entirely in your browser (no server).
        </p>
        <span class="cta">Play →</span>
      </a>

      <a class="card" routerLink="/mountaincar">
        <h2>Mountain Car</h2>
        <p>
          A PPO agent learned to swing an underpowered car up to the flag. Watch it drive
          or drive it yourself — both run entirely in your browser (no server).
        </p>
        <span class="cta">Play →</span>
      </a>

      <a class="card" routerLink="/fruitcake">
        <h2>FruitCake</h2>
        <p>
          A Suika-style drop-and-merge physics game — aim, drop, and merge fruit from cherry to
          watermelon. Runs entirely in your browser; no AI (yet). Play it fullscreen.
        </p>
        <span class="cta">Play →</span>
      </a>

      <a class="card" routerLink="/crazyfruits">
        <h2>Crazy Fruits</h2>
        <p>
          A match-3 in the spirit of the Flash-era KidCity classic — swap adjacent fruits, line up three or
          more, chain the cascades. Works with mouse or touch, entirely in your browser.
        </p>
        <span class="cta">Play →</span>
      </a>

      <a class="card" routerLink="/chess">
        <h2>Chess</h2>
        <p>
          A network that taught itself chess from scratch via AlphaZero-style self-play. Play it or watch it play
          itself — the engine, network, and search all run in your browser (no server).
        </p>
        <span class="cta">Play →</span>
      </a>

      <a class="card" routerLink="/draughts">
        <h2>Draughts</h2>
        <p>
          A network that taught itself checkers by self-play — in one hour of training it beats a material
          minimax it never saw. Play it or watch it play itself, entirely in your browser.
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
    }
  `,
})
export class Home {}

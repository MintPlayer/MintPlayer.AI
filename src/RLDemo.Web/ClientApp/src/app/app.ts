import { Component, computed, inject, signal } from '@angular/core';
import { RouterLink, RouterOutlet } from '@angular/router';
import { VersionApi, VersionInfo } from './version-api';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  private readonly versionApi = inject(VersionApi);

  protected readonly version = signal<VersionInfo | null>(null);

  // Only a real 40-hex commit SHA gets linked to its GitHub commit; "dev"/unknown stays plain text.
  protected readonly commitUrl = computed(() => {
    const sha = this.version()?.commitSha;
    return sha && /^[0-9a-f]{40}$/i.test(sha)
      ? `https://github.com/MintPlayer/MintPlayer.AI/commit/${sha}`
      : null;
  });

  constructor() {
    // Best-effort: a failed/absent endpoint just leaves the footer empty.
    this.versionApi.get().then(v => this.version.set(v)).catch(() => { });
  }

  protected shortSha(sha: string): string {
    return /^[0-9a-f]{40}$/i.test(sha) ? sha.slice(0, 7) : sha;
  }
}

import { Injectable } from '@angular/core';

/** Build identity returned by GET /api/version (all "dev" when running locally / undeployed). */
export interface VersionInfo {
  commitSha: string;
  imageDigest: string;
  deployTime: string;
}

@Injectable({ providedIn: 'root' })
export class VersionApi {
  async get(): Promise<VersionInfo> {
    const response = await fetch('/api/version');
    return response.json();
  }
}

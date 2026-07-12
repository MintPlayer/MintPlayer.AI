/**
 * Poll a game's `/status` endpoint until the model stops loading, pushing each reply to a signal.
 *
 * Hides the self-rescheduling loop and — crucially — swallows a transient fetch failure so a backend
 * blip during startup can never surface as an unhandled promise rejection (the bug two of the three
 * hand-rolled copies had). Re-schedules only while `status === 'loading'`; a `ready`/`failed` reply
 * ends the poll. Generic over the status shape so each game keeps its own typed reply.
 */
export function pollModelStatus<T extends { status: string }>(
  fetchStatus: () => Promise<T>,
  set: (status: T) => void,
  intervalMs = 2000,
): void {
  void (async () => {
    try {
      const status = await fetchStatus();
      set(status);
      if (status.status === 'loading') {
        setTimeout(() => pollModelStatus(fetchStatus, set, intervalMs), intervalMs);
      }
    } catch {
      // Backend unreachable: leave the status as-is; the UI stays usable and a later poll can recover.
    }
  })();
}

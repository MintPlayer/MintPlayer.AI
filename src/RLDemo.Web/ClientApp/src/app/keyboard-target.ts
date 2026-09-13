/**
 * True while the caret is in a text field.
 *
 * Game pages bind their key handlers at DOCUMENT level, so the board is playable the moment the page loads with
 * no click to focus it first. The cost of that reach is that the handler also sees keystrokes meant for inputs —
 * the Rush Hour page has a level-name field, and arrow keys inside it must move the caret, not a vehicle. Every
 * document-level handler guards with this before calling `preventDefault`.
 */
export function isTypingTarget(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  if (!element) return false;
  const tag = element.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || element.isContentEditable;
}

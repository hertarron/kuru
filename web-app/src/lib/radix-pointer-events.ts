/**
 * Radix can leave `pointer-events: none` on <body> after a modal closes.
 *
 * The modal and whatever layer it was opened from (a dropdown item, a context
 * menu) both manage that one style. The outer layer's cleanup runs first, so
 * the modal's restore writes nothing back and the style survives the close.
 * The window then looks normal and ignores every click until a reload.
 *
 * Clearing it ourselves once the last modal is gone is the only reliable undo.
 * The guard keeps a still-open modal (one dialog opened from another) working.
 */
export function releaseBodyPointerEvents() {
  if (
    document.querySelector(
      '[data-state="open"][role="dialog"], [data-state="open"][role="alertdialog"]'
    )
  )
    return
  document.body.style.pointerEvents = ''
}

/** Delay covering the close animation, so the guard sees the settled DOM. */
export const POINTER_EVENTS_RELEASE_DELAY = 250

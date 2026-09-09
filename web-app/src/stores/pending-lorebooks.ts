import { create } from 'zustand'
import type { ThreadLorebookState } from '@/lib/lorebook-selection'

type State = {
  /** Per-chat world-info picks made on the home screen, before a thread exists. */
  pending: ThreadLorebookState
  patch: (patch: ThreadLorebookState) => void
  clear: () => void
}

/**
 * The home screen has no thread to write `metadata.lorebooks` onto, so the
 * lorebook pill stages picks here and ChatInput copies them onto the thread it
 * creates. Deliberately not persisted: an abandoned home selection should not
 * outlive the app session.
 */
export const usePendingLorebooks = create<State>()((set) => ({
  pending: {},
  patch: (patch) => set((s) => ({ pending: { ...s.pending, ...patch } })),
  clear: () => set({ pending: {} }),
}))

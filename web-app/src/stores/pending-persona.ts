import { create } from 'zustand'
import type { ThreadPersonaState } from '@/lib/persona-selection'

type State = {
  /** A persona pinned on the home screen, before a thread exists. */
  pending: ThreadPersonaState
  patch: (patch: ThreadPersonaState) => void
  clear: () => void
}

/**
 * Mirrors `usePendingLorebooks`: the home screen has no thread to write
 * `metadata.persona` onto, so the pill stages the pick here and ChatInput
 * copies it onto the thread it creates. Not persisted — an abandoned home
 * selection should not outlive the session.
 */
export const usePendingPersona = create<State>()((set) => ({
  pending: {},
  patch: (patch) => set((s) => ({ pending: { ...s.pending, ...patch } })),
  clear: () => set({ pending: {} }),
}))

import { create } from 'zustand'
import type { ThreadMemoryMeta } from '@/lib/thread-memory'

type State = {
  /** Per-chat memory picks made on the home screen, before a thread exists. */
  pending: ThreadMemoryMeta
  patch: (patch: ThreadMemoryMeta) => void
  clear: () => void
}

/**
 * The home screen has no thread to write `metadata.memory` onto, so the
 * context view stages auto/sizes here and ChatInput copies them onto the
 * thread it creates. Deliberately not persisted: an abandoned home selection
 * should not outlive the app session.
 */
export const usePendingMemory = create<State>()((set) => ({
  pending: {},
  patch: (patch) => set((s) => ({ pending: { ...s.pending, ...patch } })),
  clear: () => set({ pending: {} }),
}))

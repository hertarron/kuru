import { create } from 'zustand'

export type MemoryRunPhase = 'chapter' | 'canon'

interface MemoryRunsState {
  /** Threads with a fold running right now, and what phase each is in. */
  active: Record<string, MemoryRunPhase>
  begin: (threadId: string, phase: MemoryRunPhase) => void
  end: (threadId: string) => void
}

/**
 * Reactive mirror of the runner's in-flight map: the chatbox strip reads
 * this to show "Writing chapter…" while a background fold runs. Session
 * state only, never persisted.
 */
export const useMemoryRuns = create<MemoryRunsState>()((set) => ({
  active: {},
  begin: (threadId, phase) =>
    set((s) => ({ active: { ...s.active, [threadId]: phase } })),
  end: (threadId) =>
    set((s) => {
      if (!(threadId in s.active)) return s
      const active = { ...s.active }
      delete active[threadId]
      return { active }
    }),
}))

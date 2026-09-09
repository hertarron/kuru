import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { localStorageKey } from '@/constants/localStorage'
import { backendStorage } from '@/lib/backendStorage'
import {
  emptyThreadMemory,
  type RootMemory,
  type ThreadMemoryData,
} from '@/lib/thread-memory'

interface ThreadMemoryState {
  /** Per thread, the chapters/canon/pins it has accrued. */
  records: Record<string, ThreadMemoryData>
  get: (threadId: string) => ThreadMemoryData | undefined
  put: (threadId: string, data: ThreadMemoryData) => void
  /**
   * Rewrite one root scope (one story universe) and stamp the record.
   * Creates the thread record and the scope when absent.
   */
  updateRoot: (
    threadId: string,
    rootId: string,
    fn: (scope: RootMemory) => RootMemory
  ) => void
  togglePin: (threadId: string, messageId: string) => void
  /** Thread delete cascade and full resets land here. */
  clearThread: (threadId: string) => void
  clearAll: () => void
}

// NOTE: chapter/canon text lives here until it migrates to per-thread
// sidecar files (`file://memory/<threadId>.json`, the lorebook pattern).
// settings.json stays honest for a handful of roleplay threads; a library
// migration is owed before this holds very large or very many chats.
export const useThreadMemory = create<ThreadMemoryState>()(
  persist(
    (set, get) => ({
      records: {},

      get: (threadId) => get().records[threadId],

      put: (threadId, data) =>
        set((s) => ({
          records: { ...s.records, [threadId]: data },
        })),

      updateRoot: (threadId, rootId, fn) =>
        set((s) => {
          const prev = s.records[threadId] ?? emptyThreadMemory()
          const scope: RootMemory = prev.roots[rootId] ?? {
            chapters: [],
            canon: [],
            passes: [],
          }
          return {
            records: {
              ...s.records,
              [threadId]: {
                ...prev,
                roots: { ...prev.roots, [rootId]: fn(scope) },
                updatedAt: Date.now(),
              },
            },
          }
        }),

      togglePin: (threadId, messageId) =>
        set((s) => {
          const prev = s.records[threadId] ?? emptyThreadMemory()
          const pins = prev.pins.includes(messageId)
            ? prev.pins.filter((id) => id !== messageId)
            : [...prev.pins, messageId]
          return {
            records: {
              ...s.records,
              [threadId]: { ...prev, pins, updatedAt: Date.now() },
            },
          }
        }),

      clearThread: (threadId) =>
        set((s) => {
          if (!(threadId in s.records)) return s
          const records = { ...s.records }
          delete records[threadId]
          return { records }
        }),

      clearAll: () => set({ records: {} }),
    }),
    {
      name: localStorageKey.threadMemory,
      storage: createJSONStorage(() => backendStorage),
      skipHydration: true,
      partialize: (state) => ({ records: state.records }),
    }
  )
)

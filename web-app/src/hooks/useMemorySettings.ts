import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { localStorageKey } from '@/constants/localStorage'
import { backendStorage } from '@/lib/backendStorage'
import {
  DEFAULT_CANON_PROMPT,
  DEFAULT_CHAPTER_PROMPT,
} from '@/lib/thread-memory'

export interface MemoryGlobalSettings {
  /** "Auto by default": untouched chats fold on their own. Default off. */
  autoDefault: boolean
  chapterPrompt: string
  canonPrompt: string
}

export const defaultMemorySettings: MemoryGlobalSettings = {
  autoDefault: false,
  chapterPrompt: DEFAULT_CHAPTER_PROMPT,
  canonPrompt: DEFAULT_CANON_PROMPT,
}

type MemorySettingsState = MemoryGlobalSettings & {
  set: <K extends keyof MemoryGlobalSettings>(
    key: K,
    value: MemoryGlobalSettings[K]
  ) => void
  /** Prompt templates only: the auto default is a separate decision. */
  resetPrompts: () => void
}

/**
 * App-wide memory behaviour: the auto default new chats inherit and the
 * fold prompt templates. Sizes are per chat (the context view), falling
 * back to the medium bundle; per-chat auto overrides live on the thread
 * (`thread.metadata.memory`); chapters/canon live in the per-thread store.
 */
export const useMemorySettings = create<MemorySettingsState>()(
  persist(
    (set) => ({
      ...defaultMemorySettings,
      set: (key, value) =>
        set({ [key]: value } as Partial<MemorySettingsState>),
      resetPrompts: () =>
        set({
          chapterPrompt: DEFAULT_CHAPTER_PROMPT,
          canonPrompt: DEFAULT_CANON_PROMPT,
        }),
    }),
    {
      name: localStorageKey.settingMemory,
      storage: createJSONStorage(() => backendStorage),
      skipHydration: true,
      partialize: (state) => ({
        autoDefault: state.autoDefault,
        chapterPrompt: state.chapterPrompt,
        canonPrompt: state.canonPrompt,
      }),
    }
  )
)

/** Live global settings snapshot for non-React call sites. */
export const memoryGlobalSettings = (): MemoryGlobalSettings => {
  const s = useMemorySettings.getState()
  return {
    autoDefault: s.autoDefault,
    chapterPrompt: s.chapterPrompt,
    canonPrompt: s.canonPrompt,
  }
}

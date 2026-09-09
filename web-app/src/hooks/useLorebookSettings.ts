import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { localStorageKey } from '@/constants/localStorage'
import { backendStorage } from '@/lib/backendStorage'
import {
  defaultLorebookRuntimeSettings,
  type LorebookRuntimeSettings,
} from '@/lib/lorebook-runtime'

type LorebookSettingState = LorebookRuntimeSettings & {
  set: <K extends keyof LorebookRuntimeSettings>(
    key: K,
    value: LorebookRuntimeSettings[K]
  ) => void
  reset: () => void
}

/**
 * App-wide world-info behaviour. Per-book overrides live on the book and
 * per-chat switches live on the thread; this is the floor both build on.
 */
export const useLorebookSettings = create<LorebookSettingState>()(
  persist(
    (set) => ({
      ...defaultLorebookRuntimeSettings,
      set: (key, value) => set({ [key]: value } as Partial<LorebookSettingState>),
      reset: () => set({ ...defaultLorebookRuntimeSettings }),
    }),
    {
      name: localStorageKey.settingLorebook,
      storage: createJSONStorage(() => backendStorage),
      skipHydration: true,
      partialize: (state) => ({
        enabled: state.enabled,
        scanDepth: state.scanDepth,
        budgetPercent: state.budgetPercent,
        maxRecursionSteps: state.maxRecursionSteps,
      }),
    }
  )
)

/** The runtime slice, without the setters, for the scanner. */
export const lorebookRuntimeSettings = (): LorebookRuntimeSettings => {
  const s = useLorebookSettings.getState()
  return {
    enabled: s.enabled,
    scanDepth: s.scanDepth,
    budgetPercent: s.budgetPercent,
    maxRecursionSteps: s.maxRecursionSteps,
  }
}

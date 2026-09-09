import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { localStorageKey } from '@/constants/localStorage'
import { backendStorage } from '@/lib/backendStorage'
import { defaultRpTextSettings, type RpTextSettings } from '@/lib/rp-text'

type RpTextSettingState = RpTextSettings & {
  set: <K extends keyof RpTextSettings>(key: K, value: RpTextSettings[K]) => void
  reset: () => void
}

/**
 * App-wide roleplay text cleanups. Global rather than per-character: this is a
 * preference about how you read and prompt, not a property of one card. Every
 * read is gated on the chat's character being a roleplay one, so an assistant
 * ignores all of it.
 */
export const useRpTextSettings = create<RpTextSettingState>()(
  persist(
    (set) => ({
      ...defaultRpTextSettings,
      set: (key, value) => set({ [key]: value } as Partial<RpTextSettingState>),
      reset: () => set({ ...defaultRpTextSettings }),
    }),
    {
      name: localStorageKey.settingRpText,
      storage: createJSONStorage(() => backendStorage),
      skipHydration: true,
      partialize: (state) => ({
        hideOocFromPrompt: state.hideOocFromPrompt,
        trimIncompleteSentence: state.trimIncompleteSentence,
      }),
    }
  )
)

/** The settings slice, without the setters, for the transport. */
export const rpTextSettings = (): RpTextSettings => {
  const s = useRpTextSettings.getState()
  return {
    hideOocFromPrompt: s.hideOocFromPrompt,
    trimIncompleteSentence: s.trimIncompleteSentence,
  }
}

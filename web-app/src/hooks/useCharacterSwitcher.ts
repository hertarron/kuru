import { create } from 'zustand'

interface CharacterSwitcherState {
  open: boolean
  setOpen: (open: boolean) => void
  toggle: () => void
  cycleHandler: (() => void) | null
  setCycleHandler: (handler: (() => void) | null) => void
}

/**
 * Shared open-state + global shortcut hook for the character switcher pill.
 * Cmd/Ctrl+J advances to the next character via the registered handler.
 */
export const useCharacterSwitcher = create<CharacterSwitcherState>((set, get) => ({
  open: false,
  setOpen: (open) => set({ open }),
  toggle: () => set({ open: !get().open }),
  cycleHandler: null,
  setCycleHandler: (handler) => set({ cycleHandler: handler }),
}))

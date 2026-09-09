import { create } from 'zustand'
import { ulid } from 'ulidx'
import { getServiceHub } from '@/hooks/useServiceHub'
import type { Lorebook, LorebookEntry } from '@/lib/lorebook'
import { LorebookPosition, LorebookLogic } from '@/lib/lorebook'

interface LorebookState {
  lorebooks: Lorebook[]
  loading: boolean
  setLorebooks: (lorebooks: Lorebook[]) => void
  /** Persists and returns the stored book, so callers can route to it. */
  addLorebook: (lorebook: Lorebook) => Lorebook
  updateLorebook: (lorebook: Lorebook) => void
  deleteLorebook: (id: string) => void
  setGlobal: (id: string, global: boolean) => void
}

const persist = (lorebook: Lorebook) => {
  getServiceHub()
    .lorebooks()
    .saveLorebook(lorebook)
    .catch((error) => console.error('Failed to save lorebook:', error))
}

export const useLorebooks = create<LorebookState>((set, get) => ({
  lorebooks: [],
  loading: true,

  setLorebooks: (lorebooks) => set({ lorebooks, loading: false }),

  addLorebook: (lorebook) => {
    const stored: Lorebook = { ...lorebook, id: lorebook.id || ulid() }
    set({ lorebooks: [...get().lorebooks, stored] })
    persist(stored)
    return stored
  },

  updateLorebook: (lorebook) => {
    set({
      lorebooks: get().lorebooks.map((b) =>
        b.id === lorebook.id ? lorebook : b
      ),
    })
    persist(lorebook)
  },

  deleteLorebook: (id) => {
    set({ lorebooks: get().lorebooks.filter((b) => b.id !== id) })
    getServiceHub()
      .lorebooks()
      .deleteLorebook(id)
      .catch((error) => console.error('Failed to delete lorebook:', error))
  },

  setGlobal: (id, global) => {
    const book = get().lorebooks.find((b) => b.id === id)
    if (!book || book.global === global) return
    get().updateLorebook({ ...book, global })
  },
}))

/**
 * Stores a book that arrived embedded in a character card and returns the id
 * to attach. Cards carry no stable identity for their book, so re-importing a
 * character makes a second copy rather than silently rewriting the first — the
 * user may well have edited it.
 */
export function adoptCardLorebook(book: Lorebook): string {
  return useLorebooks.getState().addLorebook({ ...book, id: ulid() }).id
}

/** A book with no entries yet, ready for the editor. */
export function createEmptyLorebook(name = 'New lorebook'): Lorebook {
  return {
    id: ulid(),
    name,
    description: '',
    entries: [],
    scanDepth: null,
    tokenBudget: null,
    recursiveScanning: false,
    global: false,
  }
}

/**
 * A blank entry. Defaults match the "World" preset: it lands before the
 * character definition, which is the safe place for setting material.
 */
export function createEmptyEntry(): LorebookEntry {
  return {
    id: ulid(),
    keys: [],
    secondaryKeys: [],
    selectiveLogic: LorebookLogic.AND_ANY,
    content: '',
    comment: '',
    enabled: true,
    // No keys yet, so it would fire on every turn; `constant` mirrors that.
    constant: true,
    position: LorebookPosition.beforeChar,
    depth: 4,
    role: null,
    order: 100,
    probability: 100,
    caseSensitive: false,
    extra: {},
  }
}

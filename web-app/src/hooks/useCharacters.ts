import { getServiceHub } from '@/hooks/useServiceHub'
import { Assistant as CoreAssistant } from '@janhq/core'
import { create } from 'zustand'
import { localStorageKey } from '@/constants/localStorage'
import {
  migrateAssistantToCharacter,
} from '@/lib/character-card'

interface CharacterState {
  characters: Character[]
  currentCharacter: Character | undefined
  loading: boolean
  defaultCharacterId: string
  addCharacter: (character: Character) => void
  updateCharacter: (character: Character) => void
  deleteCharacter: (id: string) => void
  setCurrentCharacter: (
    character: Character | undefined,
    saveToStorage?: boolean
  ) => void
  setDefaultCharacter: (id: string) => void
  setCharacters: (characters: (Assistant | Character)[] | null) => void
}

const setLastUsedCharacterId = (characterId: string) => {
  try {
    localStorage.setItem(localStorageKey.lastUsedCharacter, characterId)
  } catch (error) {
    console.debug('Failed to set last used character in localStorage:', error)
  }
}

/**
 * The built-in character. Keeps the legacy `jan` id so existing installs
 * resolve to it; display-wise it is the neutral "Assistant".
 */
export const defaultCharacter: Character = {
  id: 'jan',
  name: 'Assistant',
  created_at: 1747029866.542,
  parameters: {},
  avatar: '👋',
  description: `You are a helpful, capable AI assistant. Think through problems step by step and answer in the language the user writes in.

Current date: {{current_date}}`,
  instructions: '',
}

const getLastUsedCharacterId = (): string => {
  let lastUsedId
  try {
    // Fall back to the pre-characters key so upgrades keep their selection.
    lastUsedId =
      localStorage.getItem(localStorageKey.lastUsedCharacter) ??
      localStorage.getItem(localStorageKey.lastUsedAssistant)
  } catch (error) {
    console.debug('Failed to get last used character from localStorage:', error)
  }
  return lastUsedId ?? defaultCharacter.id
}

const getDefaultCharacterId = (): string | null => {
  let defaultCharacterId: string | null = null
  try {
    defaultCharacterId =
      localStorage.getItem(localStorageKey.defaultCharacterId) ??
      localStorage.getItem(localStorageKey.defaultAssistantId)
  } catch (error) {
    console.debug('Failed to get default character from localStorage:', error)
  }
  return defaultCharacterId
}

const setDefaultCharacterId = (characterId: string) => {
  try {
    if (!characterId) {
      localStorage.removeItem(localStorageKey.defaultCharacterId)
    } else {
      localStorage.setItem(localStorageKey.defaultCharacterId, characterId)
    }
  } catch (error) {
    console.debug('Failed to set default character in localStorage:', error)
  }
}

export const useCharacters = create<CharacterState>((set, get) => ({
  characters: [defaultCharacter],
  // Undefined while loading: consumers fall back to the built-in only after
  // hydrate resolves the user's actual default, so startup never flashes
  // "Assistant".
  currentCharacter: undefined,
  defaultCharacterId: '',
  loading: true,
  addCharacter: (character) => {
    set({ characters: [...get().characters, character] })
    // The assistant service is a generic JSON upsert; characters share it.
    getServiceHub()
      .assistants()
      .createAssistant(character as unknown as CoreAssistant)
      .catch((error) => {
        console.error('Failed to create character:', error)
      })
  },
  updateCharacter: (character) => {
    const state = get()
    set({
      characters: state.characters.map((c) =>
        c.id === character.id ? character : c
      ),
      // Update currentCharacter if it's the same character being updated
      currentCharacter:
        state.currentCharacter?.id === character.id
          ? character
          : state.currentCharacter,
    })
    getServiceHub()
      .assistants()
      .createAssistant(character as unknown as CoreAssistant)
      .catch((error) => {
        console.error('Failed to update character:', error)
      })
  },
  deleteCharacter: (id) => {
    const state = get()
    getServiceHub()
      .assistants()
      .deleteAssistant(
        state.characters.find((e) => e.id === id) as unknown as CoreAssistant
      )
      .catch((error) => {
        console.error('Failed to delete character:', error)
      })

    const wasCurrentCharacter = state.currentCharacter?.id === id
    const wasDefaultCharacter = state.defaultCharacterId === id

    set({ characters: state.characters.filter((c) => c.id !== id) })

    // If the deleted character was current, fall back to the built-in
    if (wasCurrentCharacter) {
      const fallback =
        get().characters.find((c) => c.id === defaultCharacter.id) ??
        get().characters[0] ??
        defaultCharacter
      set({ currentCharacter: fallback })
      setLastUsedCharacterId(fallback.id)
    }

    // If the deleted character was the default, reset to the built-in
    if (wasDefaultCharacter) {
      set({ defaultCharacterId: '' })
      setDefaultCharacterId('')
    }
  },
  setCurrentCharacter: (character, saveToStorage = true) => {
    const currentCharacter = get().currentCharacter
    if (currentCharacter !== character) {
      set({ currentCharacter: character })
      if (saveToStorage) {
        setLastUsedCharacterId(character?.id || '')
      }
    }
  },
  setDefaultCharacter: (id) => {
    const newCharacter = get().characters?.find((c) => c.id === id)
    if (newCharacter) {
      set({ defaultCharacterId: id, currentCharacter: newCharacter })
      setLastUsedCharacterId(id)
    } else {
      set({ defaultCharacterId: id })
    }
    setDefaultCharacterId(id)
  },
  setCharacters: (incoming) => {
    if (incoming) {
      // Normalize legacy assistant records into Characters.
      const characters = incoming.map((c) =>
        migrateAssistantToCharacter(c as Assistant)
      )
      characters.forEach(
        (c) => (c.id = c.id?.toString()) // new String("id") !== "id"
      )
      // Ensure the built-in always exists.
      if (!characters.some((c) => c.id === defaultCharacter.id)) {
        characters.unshift(defaultCharacter)
      }
      const lastUsedId = getLastUsedCharacterId()
      const lastUsed = characters.find((c) => c.id === lastUsedId)
      const defaultId = getDefaultCharacterId() || ''
      const defaultChar = characters.find((c) => c.id === defaultId)
      set({
        characters,
        currentCharacter: defaultChar || lastUsed,
        defaultCharacterId: defaultId,
        loading: false,
      })
    } else {
      set({ loading: false })
    }
  },
}))

/** Convenience selector for the effective character of a thread context. */
export const resolveThreadCharacter = (
  thread: Thread | undefined,
  characters: Character[]
): Character | undefined => {
  const embedded = thread?.assistants?.[0]
  if (!embedded || embedded.id === 'model-only') return undefined
  return (
    characters.find((c) => c.id === embedded.id) ??
    (embedded as unknown as Character)
  )
}

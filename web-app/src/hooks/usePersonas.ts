import { create } from 'zustand'
import { ulid } from 'ulidx'
import { getServiceHub } from '@/hooks/useServiceHub'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import type { Persona } from '@/lib/persona'
import { resolvePersona } from '@/lib/persona-selection'

interface PersonaState {
  personas: Persona[]
  loading: boolean
  setPersonas: (personas: Persona[]) => void
  /** Persists and returns the stored persona, so callers can select it. */
  addPersona: (persona: Persona) => Persona
  updatePersona: (persona: Persona) => void
  deletePersona: (id: string) => void
}

const persist = (persona: Persona) => {
  getServiceHub()
    .personas()
    .savePersona(persona)
    .catch((error) => console.error('Failed to save persona:', error))
}

export const usePersonas = create<PersonaState>((set, get) => ({
  personas: [],
  loading: true,

  setPersonas: (personas) => set({ personas, loading: false }),

  addPersona: (persona) => {
    const stored: Persona = { ...persona, id: persona.id || ulid() }
    set({ personas: [...get().personas, stored] })
    persist(stored)
    return stored
  },

  updatePersona: (persona) => {
    set({
      personas: get().personas.map((p) => (p.id === persona.id ? persona : p)),
    })
    persist(persona)
  },

  deletePersona: (id) => {
    set({ personas: get().personas.filter((p) => p.id !== id) })
    // A chat pinned to this persona falls back to the active one; clearing the
    // global pointer keeps `resolvePersona` from hunting a file that's gone.
    if (useGeneralSetting.getState().activePersonaId === id) {
      useGeneralSetting.getState().setActivePersonaId(undefined)
    }
    getServiceHub()
      .personas()
      .deletePersona(id)
      .catch((error) => console.error('Failed to delete persona:', error))
  },
}))

/**
 * Turns the old single `userName` setting into a real persona, once. Runs
 * after the library loads: an empty library plus a name the user typed under
 * the previous build is the only case that needs it, so a user who deletes
 * every persona on purpose doesn't get the old name resurrected.
 */
export function migrateUserNameToPersona(): void {
  const { personas } = usePersonas.getState()
  if (personas.length > 0) return

  const general = useGeneralSetting.getState()
  const name = general.userName?.trim()
  if (!name) return

  const stored = usePersonas.getState().addPersona({
    id: ulid(),
    name,
    description: '',
    avatar: '',
  })
  general.setActivePersonaId(stored.id)
}

/**
 * The persona a chat is played with: its own pin, else the active one. The
 * single read point for both the prompt and the UI, so the pill can never
 * disagree with what the model was told.
 */
export function resolveThreadPersona(
  thread: Thread | undefined
): Persona | undefined {
  return resolvePersona({
    library: usePersonas.getState().personas ?? [],
    activeId: useGeneralSetting.getState().activePersonaId,
    threadState: thread?.metadata?.persona,
  })
}

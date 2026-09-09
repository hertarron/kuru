/**
 * Personas — who the *user* is playing.
 *
 * The other half of a roleplay prompt. A character describes the model's
 * side; a persona describes the human's, and its name is what `{{user}}`
 * resolves to. Personas live in their own library on disk, exactly like
 * lorebooks: threads reference one by id, they never embed a copy.
 */

import { ulid } from 'ulidx'

export interface Persona {
  id: string
  /** Substituted for `{{user}}`, and the label on the pill. */
  name: string
  /** Free text about the user's character, injected into the system prompt. */
  description: string
  /** Emoji or data URL, rendered by AvatarEmoji like a character's. */
  avatar?: string
}

export function createEmptyPersona(name = 'New persona'): Persona {
  return { id: ulid(), name, description: '', avatar: '' }
}

/**
 * Coerces a stored or hand-edited file into shape. Anything unreadable is
 * rejected rather than half-loaded, so a corrupt file can't silently become
 * a nameless persona the user can't find or delete.
 */
export function normalizePersona(raw: unknown, id: string): Persona | null {
  if (!raw || typeof raw !== 'object') return null
  const o = raw as Record<string, unknown>
  const name = typeof o.name === 'string' ? o.name.trim() : ''
  if (!name) return null
  return {
    id,
    name,
    description: typeof o.description === 'string' ? o.description : '',
    avatar: typeof o.avatar === 'string' ? o.avatar : '',
  }
}

/**
 * The persona block as it reaches the model. Second person throughout: the
 * model is being told who it is talking to, not handed a second character
 * sheet to voice.
 */
export function buildPersonaPrompt(persona: Persona | undefined): string {
  const description = persona?.description?.trim()
  if (!persona || !description) return ''
  return `You are talking to ${persona.name}.\n${description}`
}

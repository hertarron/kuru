/**
 * Which persona is speaking for the user in a given chat.
 *
 * One persona is active by default; any chat can override it, and that
 * override sticks so re-reading an old log shows the persona it was played
 * with. `null` on the thread means "no persona here" — deliberately distinct
 * from absent, which means "inherit the active one".
 */

import type { Persona } from '@/lib/persona'

export interface ThreadPersonaState {
  /** Persona id for this chat, `null` for none, absent to inherit. */
  personaId?: string | null
}

export function resolvePersona({
  library,
  activeId,
  threadState,
}: {
  library: Persona[]
  activeId?: string
  threadState?: ThreadPersonaState
}): Persona | undefined {
  const pinned = threadState?.personaId
  if (pinned === null) return undefined
  const id = pinned ?? activeId
  if (!id) return undefined
  return library.find((p) => p.id === id)
}

/** Whether this chat is pinned rather than following the active persona. */
export const isPersonaPinned = (threadState?: ThreadPersonaState): boolean =>
  threadState?.personaId !== undefined

/**
 * Which lorebooks are live for a given chat.
 *
 * Three things can put a book in play: the character it is attached to, the
 * always-active flag on the book itself, and a per-chat attachment. A chat can
 * also switch a book off without detaching it anywhere else, and turn all world
 * info off at once. The scanner and the chat pill both read this, so the pill
 * shows exactly what the prompt will get.
 */

import type { Lorebook } from '@/lib/lorebook'

export interface ThreadLorebookState {
  /** Books attached to this chat only. */
  extraIds?: string[]
  /** Books switched off for this chat, whatever put them in play. */
  disabledIds?: string[]
  /** Master switch: no world info at all in this chat. */
  off?: boolean
}

export type LorebookOrigin = 'character' | 'global' | 'chat'

export interface ResolvedLorebook {
  book: Lorebook
  origin: LorebookOrigin
  enabled: boolean
}

export function resolveLorebooks({
  library,
  characterLorebookIds,
  threadState,
}: {
  library: Lorebook[]
  characterLorebookIds?: string[]
  threadState?: ThreadLorebookState
}): ResolvedLorebook[] {
  const disabled = new Set(threadState?.disabledIds ?? [])
  const off = !!threadState?.off

  // A book reachable two ways is listed once, under the first reason that put
  // it in play — that ordering is also what the pill groups by.
  const seen = new Set<string>()
  const resolved: ResolvedLorebook[] = []

  const take = (ids: string[] | undefined, origin: LorebookOrigin) => {
    for (const id of ids ?? []) {
      if (seen.has(id)) continue
      const book = library.find((b) => b.id === id)
      if (!book) continue
      seen.add(id)
      resolved.push({ book, origin, enabled: !off && !disabled.has(id) })
    }
  }

  take(characterLorebookIds, 'character')
  take(
    library.filter((b) => b.global).map((b) => b.id),
    'global'
  )
  take(threadState?.extraIds, 'chat')

  return resolved
}

/** Just the books whose entries the scanner should consider. */
export const activeLorebooks = (resolved: ResolvedLorebook[]): Lorebook[] =>
  resolved.filter((r) => r.enabled).map((r) => r.book)

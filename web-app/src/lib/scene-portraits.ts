/**
 * Which portrait belongs above each message.
 *
 * A user turn stamps whoever was in the seat when it was sent, and replies
 * inherit that stamp, so switching mid-chat re-skins only the turns from that
 * point on. Unstamped messages (greetings, older history) use the live pick.
 */

/** A character or persona, reduced to what a portrait needs. */
type Portrayed = { id: string; avatar?: string }

type SceneMessage = { id: string; role: string; metadata?: unknown }

/**
 * Only real images become portraits. An emoji avatar has nothing to show at
 * portrait size, and rendering the raw string would print the emoji as text.
 */
export const isPortraitImage = (avatar: unknown): avatar is string =>
  typeof avatar === 'string' &&
  (avatar.startsWith('data:image/') || avatar.startsWith('/images/'))

export type PortraitMapOptions = {
  /** Metadata key the stamp is written under: `characterId`, `personaId`. */
  stampKey: string
  /** Everyone who could be stamped, to resolve a stamp to an avatar. */
  library: Portrayed[]
  /** Avatar for messages before the first stamp: the live pick. */
  fallback: string | undefined
  /**
   * Roles that draw this portrait. A row that cannot draw one gets no entry,
   * so layout code can never reserve space for a portrait it will not render.
   */
  rendersOn: (role: string) => boolean
}

export function portraitsByMessageId(
  messages: SceneMessage[],
  { stampKey, library, fallback, rendersOn }: PortraitMapOptions
): Record<string, string | undefined> {
  const avatars = new Map<string, string | undefined>()
  for (const entry of library) {
    avatars.set(
      entry.id,
      isPortraitImage(entry.avatar) ? entry.avatar : undefined
    )
  }

  const map: Record<string, string | undefined> = {}
  let current = fallback
  for (const message of messages) {
    const stamped = (message.metadata as Record<string, unknown> | undefined)?.[
      stampKey
    ]
    // A stamp naming someone who has since been deleted still ends the
    // previous era -- those turns were not played by the live character.
    if (typeof stamped === 'string') current = avatars.get(stamped)
    if (rendersOn(message.role)) map[message.id] = current
  }
  return map
}

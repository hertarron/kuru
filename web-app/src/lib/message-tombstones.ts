/**
 * Session-scoped tombstones for deleted message ids.
 *
 * Message deletion persists asynchronously; a concurrent thread load can read
 * the backing file before the deletion lands and merge the row back into the
 * store, resurrecting a "ghost" of the deleted message. Every id deleted
 * through the messages store is recorded here for the rest of the session and
 * filtered out of fetched message lists. Ids are never reused, so the set is
 * unbounded only by how many distinct messages are deleted per app run.
 */
const tombstones = new Set<string>()

export const markMessageDeleted = (id: string): void => {
  tombstones.add(id)
}

export const isMessageDeleted = (id: string): boolean => tombstones.has(id)

export const filterDeletedMessages = <T extends { id: string }>(
  messages: T[]
): T[] => messages.filter((m) => !isMessageDeleted(m.id))

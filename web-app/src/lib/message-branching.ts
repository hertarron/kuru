import {
  ThreadMessage,
  ContentType,
  MessageStatus,
  ChatCompletionRole,
} from '@janhq/core'

type ThreadContent = NonNullable<ThreadMessage['content']>[number]

/**
 * Message versioning model: a parent-pointer tree stored in `metadata`.
 *
 * - `metadata.parentId` links a message to its predecessor. Messages sharing a
 *   `parentId` (or both rootless) are sibling versions.
 * - `metadata.activeChildId` selects which child branch is shown; absent ⇒ the
 *   newest sibling (by `created_at`) wins.
 *
 * The visible/sent conversation is the path from the active root down the active
 * child at each node. Legacy threads carry no branching metadata and are treated
 * as a single linear path until the first fork backfills parent links.
 */

const meta = (m: ThreadMessage) =>
  (m.metadata ?? {}) as Record<string, unknown>

// Raw link: `undefined` ⇒ legacy/unlinked, `null` ⇒ explicit root, string ⇒ parent.
const rawParent = (m: ThreadMessage): string | null | undefined => {
  const p = meta(m).parentId
  if (p === null) return null
  return typeof p === 'string' ? p : undefined
}

export const getParentId = (m: ThreadMessage): string | null => {
  const p = rawParent(m)
  return typeof p === 'string' ? p : null
}

export const getActiveChildId = (m: ThreadMessage): string | undefined => {
  const c = meta(m).activeChildId
  return typeof c === 'string' ? c : undefined
}

/** True once any message carries branching metadata. */
export const hasBranching = (messages: ThreadMessage[]): boolean =>
  messages.some(
    (m) => meta(m).parentId !== undefined || meta(m).activeChildId !== undefined
  )

const byCreatedAt = (a: ThreadMessage, b: ThreadMessage) =>
  (a.created_at ?? 0) - (b.created_at ?? 0)

/** Sibling versions of `m` (same parent, or fellow roots), oldest → newest. */
export const getSiblings = (
  messages: ThreadMessage[],
  m: ThreadMessage
): ThreadMessage[] => {
  const pid = rawParent(m)
  // Legacy/unlinked messages have no version siblings.
  if (pid === undefined) return [m]
  return messages.filter((x) => rawParent(x) === pid).sort(byCreatedAt)
}

/** 1-based position of `m` among its versions and the total count. */
export const getVersionInfo = (
  messages: ThreadMessage[],
  m: ThreadMessage
): { index: number; count: number } => {
  const siblings = getSiblings(messages, m)
  const idx = siblings.findIndex((x) => x.id === m.id)
  return { index: idx === -1 ? 1 : idx + 1, count: siblings.length }
}

const childrenOf = (
  messages: ThreadMessage[],
  parentId: string | null
): ThreadMessage[] =>
  messages.filter((x) => rawParent(x) === parentId).sort(byCreatedAt)

/** The active child of `parent`: its `activeChildId` if still valid, else newest. */
export const pickActiveChild = (
  messages: ThreadMessage[],
  parent: ThreadMessage
): ThreadMessage | undefined => {
  const children = childrenOf(messages, parent.id)
  if (children.length === 0) return undefined
  const activeId = getActiveChildId(parent)
  const chosen = activeId && children.find((c) => c.id === activeId)
  return chosen || children[children.length - 1]
}

/**
 * The visible linear conversation. Legacy (un-branched) threads are returned
 * unchanged; otherwise walk from the active root following the active child.
 */
export const computeActivePath = (
  messages: ThreadMessage[],
  activeRootId?: string
): ThreadMessage[] => {
  if (!hasBranching(messages)) return messages

  const roots = childrenOf(messages, null)
  if (roots.length === 0) return messages

  const root =
    (activeRootId && roots.find((r) => r.id === activeRootId)) ||
    roots[roots.length - 1]

  const path: ThreadMessage[] = []
  let cur: ThreadMessage | undefined = root
  const seen = new Set<string>()
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id)
    path.push(cur)
    cur = pickActiveChild(messages, cur)
  }
  return path
}

/** Return a copy of `m` with `parentId` set in metadata. */
export const withParentId = (
  m: ThreadMessage,
  parentId: string | null
): ThreadMessage => ({
  ...m,
  metadata: { ...(m.metadata ?? {}), parentId },
})

/** Return a copy of `m` with `activeChildId` set in metadata. */
export const withActiveChild = (
  m: ThreadMessage,
  childId: string
): ThreadMessage => ({
  ...m,
  metadata: { ...(m.metadata ?? {}), activeChildId: childId },
})

/**
 * Assign `parentId` along a linear path so the tree is well-formed before the
 * first fork. Returns only the messages that need a write (empty if already
 * branched). `path` must be in conversation order.
 */
export const backfillParentIds = (path: ThreadMessage[]): ThreadMessage[] => {
  if (path.some((m) => meta(m).parentId !== undefined)) return []
  return path.map((m, i) => withParentId(m, i === 0 ? null : path[i - 1].id))
}

/**
 * Repair threads corrupted on disk by the assistant-orphan bug (#8357 and its
 * earlier variants): in a branched thread an assistant reply was persisted with
 * a detached parent link -- either `parentId: null` or no `parentId` at all.
 * `computeActivePath` then treats it as a phantom root / non-child and drops it,
 * so the conversation renders as user messages in a row with the replies gone.
 *
 * The break compounds: because the reply is detached, the *following* user turn
 * was linked to the previous user instead of to that reply, so the whole tail
 * hangs off the wrong nodes. Repair each detached assistant by:
 *   1. re-parenting it to the user turn it answers (nearest preceding user by
 *      `created_at`, mirroring `resolveAssistantParent`), and
 *   2. re-linking the user turn that follows it (currently a child of that same
 *      user) onto the assistant, restoring the alternating chain so every later
 *      turn rejoins the active path.
 * Genuine version forks are preserved: only detached assistants and the single
 * user turn each one displaced are touched.
 *
 * Returns only the messages that need a write (empty when nothing is broken).
 * Idempotent: once linked, an assistant has a string parent and is skipped.
 * No-op on legacy (un-branched) threads -- without branching, `computeActivePath`
 * returns messages unchanged, so a missing parent is harmless there.
 */
export const repairDetachedAssistants = (
  messages: ThreadMessage[]
): ThreadMessage[] => {
  if (!hasBranching(messages)) return []
  const ordered = [...messages].sort(byCreatedAt)
  const writes = new Map<string, ThreadMessage>()
  const parentOf = (m: ThreadMessage) => rawParent(writes.get(m.id) ?? m)

  for (let i = 0; i < ordered.length; i++) {
    const a = ordered[i]
    if (a.role !== ChatCompletionRole.Assistant) continue
    const p = parentOf(a)
    if (p !== null && p !== undefined) continue

    let user: ThreadMessage | undefined
    for (let j = i - 1; j >= 0; j--) {
      if (ordered[j].role === ChatCompletionRole.User) {
        user = ordered[j]
        break
      }
    }
    if (!user) continue
    writes.set(a.id, withParentId(writes.get(a.id) ?? a, user.id))

    for (let j = i + 1; j < ordered.length; j++) {
      const c = ordered[j]
      if (c.role === ChatCompletionRole.User && parentOf(c) === user.id) {
        writes.set(c.id, withParentId(writes.get(c.id) ?? c, a.id))
        break
      }
    }
  }

  return [...writes.values()]
}

export type SplicePlan = {
  /** Direct children of the deleted node, rewritten onto the grandparent. */
  reparented: ThreadMessage[]
  /**
   * Child promoted into the deleted node's slot on the active path, if any.
   * Prefers the child the node itself had marked active, else the newest --
   * deterministic, instead of relying on the newest-sibling fallback.
   */
  promotedChildId: string | null
  /** True when the deleted node was a root (its promoted child becomes root). */
  wasActiveRootEligible: boolean
}

/**
 * Plan splicing `target` out of the tree: its direct children are re-parented
 * to the target's parent (or become roots), so the visible conversation stays
 * continuous -- "remove this turn" -- rather than amputating everything below.
 * Pure: returns only the messages that need a write; the caller deletes the
 * target itself and re-points the active branch/root.
 *
 * No-op plan for legacy (un-branched) threads: there are no links to fix up,
 * deleting from the linear list is already correct.
 */
export const planSplice = (
  messages: ThreadMessage[],
  target: ThreadMessage
): SplicePlan => {
  if (!hasBranching(messages)) {
    return { reparented: [], promotedChildId: null, wasActiveRootEligible: false }
  }
  const parentId = rawParent(target)
  // Un-linked legacy row inside an otherwise branched thread: no children can
  // reference it through the tree, so there is nothing to splice either.
  if (parentId === undefined) {
    return { reparented: [], promotedChildId: null, wasActiveRootEligible: false }
  }

  const children = childrenOf(messages, target.id)
  const grandparentId = parentId === null ? null : parentId
  const activeChildId = getActiveChildId(target)
  const promoted =
    (activeChildId && children.find((c) => c.id === activeChildId)) ||
    children[children.length - 1]

  return {
    reparented: children.map((c) => withParentId(c, grandparentId)),
    promotedChildId: promoted ? promoted.id : null,
    wasActiveRootEligible: parentId === null,
  }
}


export type UserDeletePlan = {
  /**
   * Direct assistant children of the target -- every version of the paired
   * reply. They are deleted along with the turn instead of being re-parented,
   * so an answer never survives pointing at a question that no longer exists.
   */
  doomedReplyIds: string[]
  /**
   * Children of those replies, promoted onto the grandparent so any later
   * turns continue through the gap left by the removed exchange.
   */
  reparented: ThreadMessage[]
  /** Grandchild taking the deleted turn's slot on the active path, if any. */
  promotedChildId: string | null
  /** True when the deleted turn was a root (promoted child becomes root). */
  wasRootTurn: boolean
}

/**
 * Plan deleting a USER turn together with its paired replies. Unlike
 * planSplice, the direct assistant children are removed rather than kept --
 * an answer to a deleted question would otherwise re-attach to the previous
 * turn and read as answering the wrong thing. Later turns (grandchildren)
 * survive, spliced up so the conversation stays continuous.
 *
 * Pure: returns writes and ids; caller persists them and removes the rows.
 * No-op plan on legacy/un-branched threads (the handler pairs by list order
 * there instead).
 */
export const planUserDeleteWrites = (
  messages: ThreadMessage[],
  target: ThreadMessage
): UserDeletePlan => {
  if (!hasBranching(messages)) {
    return {
      doomedReplyIds: [],
      reparented: [],
      promotedChildId: null,
      wasRootTurn: false,
    }
  }

  const parentId = getParentId(target)
  const replies = messages.filter(
    (m) => rawParent(m) === target.id && m.role === ChatCompletionRole.Assistant
  )
  const replyIds = new Set(replies.map((r) => r.id))
  const grandchildren = replies.flatMap((r) =>
    messages.filter((m) => rawParent(m) === r.id)
  )

  // Which grandchild takes over the turn's slot: follow the chain the turn
  // had marked active, else the newest reply's newest child.
  const activeReplyId = getActiveChildId(target)
  const activeReply =
    (activeReplyId && replies.find((r) => r.id === activeReplyId)) ||
    replies[replies.length - 1]
  const promotedGrandchild = activeReply
    ? pickActiveChild(messages, activeReply)
    : undefined

  return {
    doomedReplyIds: [...replyIds],
    reparented: grandchildren.map((c) => withParentId(c, parentId)),
    promotedChildId: promotedGrandchild ? promotedGrandchild.id : null,
    wasRootTurn: parentId === null,
  }
}

export type ContinuationPlan = {
  parentId: string | null
  deletePartialId: string | null
}

/**
 * Resume a stopped assistant turn in place. The continued reply inherits the
 * stale partial's parent, and that partial is dropped, so the reply replaces it
 * rather than forking a sibling version. Falls back to `fallbackParentId` when
 * no partial is targeted or its id is stale.
 */
export const planContinuation = (
  messages: ThreadMessage[],
  replyId: string,
  partialId: string | null | undefined,
  fallbackParentId: string | null
): ContinuationPlan => {
  const partial = partialId
    ? messages.find((m) => m.id === partialId)
    : undefined
  const parentId = (partial ? getParentId(partial) : null) ?? fallbackParentId
  const deletePartialId =
    partial && partial.id !== replyId ? partial.id : null
  return { parentId, deletePartialId }
}

/**
 * Build a new sibling version of `source`: fresh id/timestamp, same parent,
 * no children, optional text override. Used for edit-user / edit-assistant forks.
 *
 * Fork siblings are independent branches: any inherited surgical tag is
 * stripped, otherwise a fork of a surgical replacement would be mistaken for
 * a shared-subtree sibling during version navigation.
 */
export const makeSibling = (
  source: ThreadMessage,
  opts: { id: string; createdAt: number; text?: string }
): ThreadMessage => {
  const content: ThreadContent[] =
    opts.text !== undefined
      ? [{ type: ContentType.Text, text: { value: opts.text, annotations: [] } }]
      : source.content.map((c) => ({ ...c }))
  const sourceMeta = { ...(source.metadata ?? {}) } as Record<string, unknown>
  delete sourceMeta.activeChildId
  delete sourceMeta.error
  return {
    ...source,
    id: opts.id,
    content,
    status: MessageStatus.Ready,
    created_at: opts.createdAt,
    completed_at: opts.createdAt,
    metadata: { ...sourceMeta, parentId: getParentId(source) },
  }
}

/**
 * Destructive in-place edit: same id, same parent, same children -- only the
 * text changes. The tree is never touched, so no ownership can drift.
 * Pure: returns the updated row; caller persists it.
 */
export const inPlaceEditContent = (
  target: ThreadMessage,
  text: string
): ThreadMessage => ({
  ...target,
  status: MessageStatus.Ready,
  completed_at: Date.now(),
  content: [
    { type: ContentType.Text, text: { value: text, annotations: [] } },
  ],
})

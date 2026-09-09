/**
 * Edit-mode scenario matrix.
 *
 * Drives a faithful in-memory simulation of the thread-route handlers
 * (handleEditMessage surgical/fork, handleSwitchVersion, handleDeleteMessage
 * splice/pair-delete, processAndSendMessage anchoring, onFinish reply linking)
 * handleDeleteMessage splice, processAndSendMessage anchoring, onFinish
 * reply linking) and asserts tree invariants plus expected ownership after
 * every operation. The orchestration mirrors web-app/src/routes/threads/
 * $threadId.tsx; the tree math itself comes from message-branching.ts so the
 * two cannot drift silently -- if a handler changes shape, update the mirror.
 */
import { describe, it, expect } from 'vitest'
import { ContentType, MessageStatus, type ThreadMessage } from '@janhq/core'
import {
  computeActivePath,
  backfillParentIds,
  makeSibling,
  withActiveChild,
  getParentId,
  getSiblings,
  hasBranching,
  planSplice,
  inPlaceEditContent,
  planUserDeleteWrites,
} from '../message-branching'

let clock = 1_000
function msg(
  id: string,
  role: 'user' | 'assistant',
  text: string,
  metadata?: Record<string, unknown>
): ThreadMessage {
  const t = ++clock
  return {
    id,
    object: 'thread.message',
    thread_id: 'thr',
    role: role as ThreadMessage['role'],
    content: [
      { type: ContentType.Text, text: { value: text, annotations: [] } },
    ],
    status: MessageStatus.Ready,
    created_at: t,
    completed_at: t,
    metadata,
  }
}

const ids = (ms: ThreadMessage[]) => ms.map((m) => m.id)

// -----------------------------------------------------------------------------
// Harness: mirrors $threadId.tsx handler orchestration over an array store
// -----------------------------------------------------------------------------

function createHarness(seedMessages: ThreadMessage[]) {
  let msgs: ThreadMessage[] = seedMessages.map((m) => ({ ...m }))
  let activeRootId: string | undefined
  let pendingAssistantParentId: string | null = null
  let editCounter = 0

  const find = (id: string) => msgs.find((m) => m.id === id)

  // updateMessage: synchronous store write
  const apply = (m: ThreadMessage) => {
    const i = msgs.findIndex((x) => x.id === m.id)
    if (i >= 0) msgs[i] = m
  }
  const addMsg = (m: ThreadMessage) => {
    clock += 1
    msgs.push({ ...m, created_at: m.created_at || clock })
  }

  function setActiveBranch(node: ThreadMessage) {
    const parentId = getParentId(node)
    if (parentId === null) {
      activeRootId = node.id
      return
    }
    const parent = find(parentId)
    if (parent) apply(withActiveChild(parent, node.id))
  }

  function path(): ThreadMessage[] {
    return computeActivePath(msgs, activeRootId)
  }

  // ensureBranched: first fork in a linear thread backfills parent links.
  function ensureBranchedIfNeeded() {
    if (!hasBranching(msgs)) {
      backfillParentIds(msgs).forEach(apply)
      msgs = msgs.map((m) => ({ ...m }))
    }
  }

  // onFinish: link the new assistant reply to the pending user turn, falling
  // back to resolveAssistantParent (nearest preceding user on the path).
  function finishReply(replyId: string, text = `reply:${replyId}`) {
    let parentId = pendingAssistantParentId
    if (parentId == null) {
      const p = path()
      for (let i = p.length - 1; i >= 0; i--) {
        if (p[i].role === 'user') {
          parentId = p[i].id
          break
        }
      }
    }
    addMsg(msg(replyId, 'assistant', text, { parentId }))
    pendingAssistantParentId = null
  }

  return {
    msgs: () => msgs,
    find,
    path,

    // processAndSendMessage: anchor the user turn to the active-path tail,
    // then the assistant reply arrives through onFinish.
    send(id: string, text: string) {
      ensureBranchedIfNeeded()
      let parentId: string | null = null
      if (hasBranching(msgs)) {
        const p = path()
        parentId = p.length ? p[p.length - 1].id : null
        pendingAssistantParentId = id
      }
      addMsg(
        msg(id, 'user', text, { ...(parentId !== null ? { parentId } : {}) })
      )
      finishReply(`reply-${id}`)
    },

    // handleEditMessage: destructive in-place overwrite for user and
    // assistant messages alike -- never generates. Same node, same id --
    // the tree is never touched.
    edit(messageId: string, newText: string) {
      ensureBranchedIfNeeded()
      const target = find(messageId)
      if (!target) throw new Error(`edit: missing ${messageId}`)
      apply(inPlaceEditContent(target, newText))
      return messageId
    },

    // handleFork: duplicate a reply as a childless sibling version (2/2).
    fork(messageId: string) {
      ensureBranchedIfNeeded()
      const target = find(messageId)
      if (!target) throw new Error(`fork: missing ${messageId}`)
      const copy = makeSibling(target, {
        id: `${messageId}-f${++editCounter}`,
        createdAt: ++clock,
      })
      addMsg(copy)
      setActiveBranch(copy)
      return copy.id
    },

    // handleRegenerate + onFinish: a fresh sibling under the same parent,
    // answering the nearest preceding user turn.
    regen(messageId: string, text = `fresh-${messageId}`) {
      ensureBranchedIfNeeded()
      const p = path()
      const idx = p.findIndex((m) => m.id === messageId)
      if (idx === -1) throw new Error(`regen: missing ${messageId}`)
      let parentId: string | null = null
      for (let i = idx; i >= 0; i--) {
        if (p[i].role === 'user') {
          parentId = p[i].id
          break
        }
      }
      const replyId = `regen-${messageId}`
      addMsg(msg(replyId, 'assistant', text, { parentId }))
      const parent = find(parentId)
      if (parent) apply(withActiveChild(parent, replyId))
    },

    // handleSwitchVersion
    switchVersion(messageId: string, dir: -1 | 1) {
      const target = find(messageId)
      if (!target) throw new Error(`switch: missing ${messageId}`)
      const siblings = getSiblings(msgs, target)
      const idx = siblings.findIndex((m) => m.id === messageId)
      const next = siblings[idx + dir]
      if (!next) return false
      setActiveBranch(next)
      return true
    },

    // handleDeleteMessage
    del(messageId: string) {
      ensureBranchedIfNeeded()
      const target = find(messageId)
      if (!target) throw new Error(`del: missing ${messageId}`)
      if (!hasBranching(msgs)) {
        // Legacy pairing by list order: the row after a user turn, if it is
        // an assistant reply, goes with it.
        const i = msgs.findIndex((m) => m.id === messageId)
        const doomed = [messageId]
        if (msgs[i + 1]?.role === 'assistant') doomed.push(msgs[i + 1].id)
        msgs = msgs.filter((m) => !doomed.includes(m.id))
        return
      }
      if (target.role === 'user') {
        const plan = planUserDeleteWrites(msgs, target)
        plan.reparented.forEach(apply)
        const onActivePath = path().some((m) => m.id === messageId)
        if (onActivePath && plan.promotedChildId) {
          if (plan.wasRootTurn) {
            activeRootId = plan.promotedChildId
          } else {
            const p = find(getParentId(target)!)
            if (p) apply(withActiveChild(p, plan.promotedChildId))
          }
        }
        msgs = msgs.filter(
          (m) =>
            m.id !== messageId && !plan.doomedReplyIds.includes(m.id)
        )
        return
      }
      if (hasBranching(msgs)) {
        const onActivePath = path().some((m) => m.id === messageId)
        const plan = planSplice(msgs, target)
        plan.reparented.forEach(apply)
        if (onActivePath && plan.promotedChildId) {
          const targetParentId = getParentId(target)
          if (targetParentId === null) {
            activeRootId = plan.promotedChildId
          } else {
            const parent = find(targetParentId)
            if (parent)
              apply(withActiveChild(parent, plan.promotedChildId))
          }
        }
      }
      msgs = msgs.filter((m) => m.id !== messageId)
    },
  }
}

// Seed helpers ---------------------------------------------------------------

/** Linear branched conversation u1 -> a1 -> u2 -> a2 -> u3. */
function linearSeed() {
  const raw = [
    msg('u1', 'user', 'q1'),
    msg('a1', 'assistant', 'r1'),
    msg('u2', 'user', 'q2'),
    msg('a2', 'assistant', 'r2'),
    msg('u3', 'user', 'q3'),
  ]
  backfillParentIds(raw).forEach((m) => {
    const i = raw.findIndex((x) => x.id === m.id)
    raw[i] = m
  })
  return raw
}

// Invariants -----------------------------------------------------------------

function checkInvariants(h: ReturnType<typeof createHarness>) {
  const ms = h.msgs()
  const seen = new Set<string>()
  for (const m of ms) {
    expect(seen.has(m.id)).toBe(false)
    seen.add(m.id)
    const pid = getParentId(m)
    if (pid !== null && !ms.some((x) => x.id === pid)) {
      throw new Error(`dangling parent ${pid} on ${m.id}`)
    }
    const src = (m.metadata as Record<string, unknown>)?.surgicalSource
    if (typeof src === 'string') {
      const source = ms.find((x) => x.id === src)
      // Dangling tag (source deleted) is inert: carries need both nodes alive.
      if (!source) continue
      if (getParentId(source) !== pid)
        throw new Error(`surgicalSource ${src} not a sibling of ${m.id}`)
    }
  }
  // Active path must be a contiguous parent-linked chain.
  const p = h.path()
  for (let i = 1; i < p.length; i++) {
    expect(getParentId(p[i])).toBe(p[i - 1].id)
  }
  // Path must include every ancestor of its tail (no skipped links).
  if (!hasBranching(ms) && ms.length) expect(p).toEqual(ms)
}

// Matrix -----------------------------------------------------------------------

describe('edit-mode scenario matrix', () => {
  const run = (fn: (h: ReturnType<typeof createHarness>) => void) => {
    const h = createHarness(linearSeed())
    fn(h)
    checkInvariants(h)
  }

  describe('A. in-place edits', () => {
    it('A1: tail user edit overwrites the same node without generating', () =>
      run((h) => {
        const count = h.msgs().length
        const id = h.edit('u3', 'q3!')
        expect(id).toBe('u3')
        expect(h.msgs().length).toBe(count)
        expect(h.path().map((m) => m.id)).toEqual([
          'u1', 'a1', 'u2', 'a2', 'u3',
        ])
        expect(h.find('u3')!.content[0].text?.value).toBe('q3!')
      }))

    it('A2: mid-chain assistant edit changes text only -- tree untouched', () =>
      run((h) => {
        const count = h.msgs().length
        h.edit('a2', 'r2!')
        expect(h.msgs().length).toBe(count)
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
        expect(h.find('a2')!.content[0].text?.value).toBe('r2!')
      }))

    it('A3: mid-chain user edit changes text only -- the reply below stays', () =>
      run((h) => {
        h.edit('u2', 'q2!')
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
        expect(h.find('u2')!.content[0].text?.value).toBe('q2!')
      }))

    it('A4: repeated in-place edits stay on one node with no drift', () =>
      run((h) => {
        const count = h.msgs().length
        h.edit('a2', 'one')
        h.edit('a2', 'two')
        h.edit('a2', 'three')
        expect(h.msgs().length).toBe(count)
        expect(h.find('a2')!.content[0].text?.value).toBe('three')
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
      }))
  })

  describe('B. forking replies', () => {
    it('B1: fork duplicates a reply as an identical childless version (2/2)', () =>
      run((h) => {
        const copy = h.fork('a1')
        expect(copy).toBe('a1-f1')
        expect(h.path().map((m) => m.id)).toEqual(['u1', copy])
        // back to 1/1: original branch with ALL its subsequent messages
        h.switchVersion(copy, -1)
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
      }))

    it('B2: editing the fork copy leaves the original branch intact', () =>
      run((h) => {
        const copy = h.fork('a1')
        h.edit(copy, 'rewritten copy')
        expect(h.find(copy)!.content[0].text?.value).toBe('rewritten copy')
        expect(h.find('a1')!.content[0].text?.value).toBe('r1')
        h.switchVersion(copy, -1)
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
      }))

    it('B3: regenerating from the fork adds another version (3/3)', () =>
      run((h) => {
        const f1 = h.fork('a1') // 2/2
        h.regen(f1) // 3/3: fresh sibling under the same parent
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'regen-a1-f1'])
        // walking back lands on each earlier version, ending at the original
        // with its whole downstream branch intact
        h.switchVersion('regen-a1-f1', -1)
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1-f1'])
        h.switchVersion('a1-f1', -1)
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
      }))

    it('B4: a follow-up sent while viewing the fork anchors there alone', () =>
      run((h) => {
        h.fork('a1')
        h.send('m1', 'meowdy')
        expect(h.path().map((m) => m.id)).toEqual([
          'u1', 'a1-f1', 'm1', 'reply-m1',
        ])
        h.switchVersion('a1-f1', -1)
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
      }))
  })

  describe('C. deletes', () => {
    it('D12: deleting an edited assistant splices its chain up', () =>
      run((h) => {
        h.edit('a2', 'r2!')
        h.del('a2')
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'u3'])
      }))

    it('D15: follow-up after an in-place mid-edit anchors under the same node', () =>
      run((h) => {
        h.edit('a2', 'r2!')
        h.send('n1', 'next')
        expect(h.path().map((m) => m.id)).toEqual([
          'u1', 'a1', 'u2', 'a2', 'u3', 'n1', 'reply-n1',
        ])
      }))

    it('D17: legacy linear thread takes an in-place first edit cleanly', () =>
      run((h) => {
        const id = h.edit('u2', 'q2!')
        expect(id).toBe('u2')
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
      }))

    it('D18: root-message in-place edit stays rooted without generating', () =>
      run((h) => {
        const count = h.msgs().length
        h.edit('u1', 'hello!')
        expect(h.msgs().length).toBe(count)
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3'])
        expect(h.find('u1')!.content[0].text?.value).toBe('hello!')
      }))

    it('D19: repeated in-place edits never drift the tree', () =>
      run((h) => {
        const snapshot = () => h.path().map((m) => m.id).join('>')
        const snap = snapshot()
        for (let i = 0; i < 5; i++) h.edit('a2', `v${i}`)
        expect(snapshot()).toBe(snap)
        expect(h.msgs().length).toBe(5)
      }))

    it('D20: deleting the root turn pair-deletes its reply; chain stays anchored', () =>
      run((h) => {
        h.del('u1') // root user message -- pair-delete removes a1 too
        expect(h.path().map((m) => m.id)).toEqual(['u2', 'a2', 'u3'])
        h.send('n1', 'next')
        expect(h.path().map((m) => m.id)).toEqual([
          'u2', 'a2', 'u3', 'n1', 'reply-n1',
        ])
        expect(getParentId(h.find('n1')!)).toBe('u3')
      }))

    it('D21: deleting a user turn takes its paired reply; conversation stays continuous', () =>
      run((h) => {
        h.del('u2')
        expect(h.find('a2')).toBeUndefined()
        expect(h.path().map((m) => m.id)).toEqual(['u1', 'a1', 'u3'])
        h.send('n1', 'next')
        expect(h.path().map((m) => m.id)).toEqual([
          'u1', 'a1', 'u3', 'n1', 'reply-n1',
        ])
      }))

    it('D22: pair-delete removes every forked version of the reply too', () =>
      run((h) => {
        const f1 = h.fork('a1')
        const f2 = h.fork(f1)
        expect(h.msgs().length).toBe(7) // u1,a1,f1,f2,u2,a2,u3
        h.del('u1')
        // The entire exchange -- every reply version and its continuation --
        // goes with the deleted opening turn.
        expect(h.find('a1')).toBeUndefined()
        expect(h.find(f1)).toBeUndefined()
        expect(h.find(f2)).toBeUndefined()
        expect(h.path().map((m) => m.id)).toEqual(['u2', 'a2', 'u3'])
      }))
  })
})
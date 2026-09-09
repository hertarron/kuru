import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { act } from '@testing-library/react'
import type { ThreadMessage } from '@janhq/core'
import {
  runMemoryFold,
  stripReasoning,
  requestMemoryStop,
  type MemorySummarizer,
} from '../memory-runner'
import { useMessages } from '@/hooks/useMessages'
import { useThreads } from '@/hooks/useThreads'
import { useCharacters } from '@/hooks/useCharacters'
import { useMemorySettings } from '@/hooks/useMemorySettings'
import { useThreadMemory } from '@/hooks/useThreadMemory'
import {
  chaptersForPath,
  coverageCursor,
  emptyThreadMemory,
  rootIdOf,
  rootScope,
} from '../thread-memory'

const THREAD = 'mem-thread'

const SIZES = { keepRaw: 4, keepChapters: 4, batch: 4, responseLength: 100 }

function message(
  id: string,
  role: 'user' | 'assistant',
  text = `line ${id}`,
  metadata: Record<string, unknown> = {}
): ThreadMessage {
  return {
    id,
    object: 'thread.message',
    thread_id: THREAD,
    role,
    content: [{ type: 'text', text: { value: text, annotations: [] } }],
    status: 'ready',
    created_at: 1,
    completed_at: 1,
    metadata,
  } as ThreadMessage
}

/** N alternating turns, oldest first. */
function conversation(n: number, prefix = 'm'): ThreadMessage[] {
  return Array.from({ length: n }, (_, i) =>
    message(`${prefix}${i + 1}`, i % 2 === 0 ? 'user' : 'assistant')
  )
}

function seedThread(n: number, prefix = 'm') {
  act(() => {
    useMessages.setState({ messages: { [THREAD]: conversation(n, prefix) } })
  })
}

function threadMeta() {
  return { memory: { auto: 'on' as const, sizes: { ...SIZES } } }
}

function seedStores() {
  act(() => {
    useThreadMemory.getState().clearAll()
    useMessages.setState({ messages: {} })
    useThreads.setState({
      threads: {
        [THREAD]: {
          id: THREAD,
          object: 'thread',
          title: 't',
          assistants: [{ id: 'char1' }],
          created: 1,
          updated: 1,
          metadata: threadMeta(),
        } as never,
      },
      isLoadingThreads: false,
    })
    useCharacters.setState({
      characters: [{ id: 'char1', kind: 'roleplay', name: 'Mira' } as never],
    })
    useMemorySettings.setState({ autoDefault: false })
  })
}

/** Canned weak model: chapters become note lines, canon calls become facts. */
function stubSummarizer(opts: {
  canon?: (call: number) => string
  onChapter?: (call: number) => void
} = {}): MemorySummarizer & { calls: { chapter: number; canon: number } } {
  const calls = { chapter: 0, canon: 0 }
  const fn = (async (
    prompt: string,
    _o: { maxTokens: number; signal: AbortSignal }
  ) => {
    if (prompt.includes('## Permanent record so far')) {
      calls.canon++
      return opts.canon?.(calls.canon) ?? `+ character | fact ${calls.canon}`
    }
    calls.chapter++
    opts.onChapter?.(calls.chapter)
    return `- noted ${calls.chapter}`
  }) as MemorySummarizer & { calls: { chapter: number; canon: number } }
  fn.calls = calls
  return fn
}

let idCounter = 0

describe('memory runner (headless simulation)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    idCounter = 0
    seedStores()
  })

  it('folds a long back-and-forth into whole batches, raw tail intact', async () => {
    seedThread(40)
    const summarize = stubSummarizer()

    const result = await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })

    // 40 owed-eligible minus the 4 raw tail: 36 owed, 9 chapters.
    expect(result.chaptersWritten).toBe(9)
    // 9 unculled >= keep 4 + take 4: one canon pass over the oldest 4.
    expect(result.canonPasses).toBe(1)
    expect(result.canonChanged).toBeGreaterThan(0)

    const record = useThreadMemory.getState().records[THREAD]
    const scope = rootScope(record, 'm1')
    expect(scope.chapters).toHaveLength(9)
    expect(scope.chapters.filter((c) => c.culled)).toHaveLength(4)
    expect(scope.canon.length).toBeGreaterThan(0)
    expect(scope.passes).toHaveLength(1)
    expect(scope.passes[0].chapterIds).toHaveLength(4)
    expect(scope.passes[0].before).toEqual([])

    // Coverage: m1..m36 folded, m37..m40 still raw.
    const path = useMessages
      .getState()
      .getMessages(THREAD)
      .map((m) => ({
        id: m.id,
        role: String(m.role),
        text: (m.content[0] as { text?: { value?: string } })?.text?.value ?? '',
      }))
    const { cursor } = coverageCursor(path, scope.chapters, new Set())
    expect(cursor).toBe(36)
    expect(path.slice(0, 36).every((m) => m.id.startsWith('m'))).toBe(true)
  })

  it('revises canon across passes without stacking copies', async () => {
    seedThread(40)
    let pass = 0
    const summarize = stubSummarizer({
      canon: () => {
        pass++
        return pass === 1
          ? '+ goal | Reach the archivist.'
          : '~ C1 | Reached the archivist.'
      },
    })
    await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })

    // Grow the chat and fold again: 4 more chapters complete the next window.
    act(() => {
      const prev = useMessages.getState().getMessages(THREAD)
      useMessages.setState({
        messages: {
          [THREAD]: [...prev, ...conversation(16, 'n').map((m, i) => ({ ...m, id: `n${i + 1}` }))],
        },
      })
    })
    const second = await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })
    expect(second.canonPasses).toBe(1)

    const scope = rootScope(useThreadMemory.getState().records[THREAD], 'm1')
    const goals = scope.canon.filter((e) => e.cat === 'goal')
    expect(goals).toHaveLength(1)
    expect(goals[0].text).toBe('Reached the archivist.')
    expect(scope.passes[1].before.map((e) => e.text)).toContain(
      'Reach the archivist.'
    )
  })

  it('keeps prefix chapters on a fork and folds the new tail only', async () => {
    seedThread(12)
    const summarize = stubSummarizer()
    await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })
    // 12 - 4 raw = 8 owed: chapters over m1-m4 and m5-m8.
    expect(
      rootScope(useThreadMemory.getState().records[THREAD], 'm1').chapters
    ).toHaveLength(2)

    // Fork at m8: same prefix, eight new branch messages.
    const prefix = conversation(8).map((m, i) => ({
      ...m,
      metadata: { parentId: i === 0 ? null : `m${i}` },
    }))
    const branch = Array.from({ length: 8 }, (_, i) =>
      message(`b${i + 1}`, i % 2 === 0 ? 'user' : 'assistant', `branch ${i + 1}`, {
        parentId: i === 0 ? 'm8' : `b${i}`,
      })
    )
    act(() => {
      useMessages.setState({ messages: { [THREAD]: [...prefix, ...branch] } })
      useThreads.setState({
        threads: {
          [THREAD]: {
            ...(useThreads.getState().threads[THREAD] as object),
            metadata: { ...threadMeta(), activeRootId: 'm1' },
          } as never,
        },
      })
    })

    const result = await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })
    // b1..b8 minus the 4 raw tail: one chapter over the branch tail.
    expect(result.chaptersWritten).toBe(1)
    const record = useThreadMemory.getState().records[THREAD]
    const scope = rootScope(record, 'm1')
    expect(scope.chapters).toHaveLength(3)
    const pathIds = new Set(
      useMessages.getState().getMessages(THREAD).map((m) => m.id)
    )
    // Prefix chapters still describe the forked path.
    expect(chaptersForPath(scope.chapters.slice(0, 2), pathIds)).toHaveLength(2)
  })

  it('heals after a delete by refolding the survivors', async () => {
    seedThread(12)
    const summarize = stubSummarizer()
    await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })

    act(() => {
      useMessages.setState({
        messages: {
          [THREAD]: useMessages
            .getState()
            .getMessages(THREAD)
            .filter((m) => m.id !== 'm3'),
        },
      })
    })
    const result = await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })
    // The m1-m4 chapter no longer covers; survivors refold.
    expect(result.chaptersWritten).toBeGreaterThan(0)
    const scope = rootScope(useThreadMemory.getState().records[THREAD], 'm1')
    const ids = new Set(
      useMessages.getState().getMessages(THREAD).map((m) => m.id)
    )
    const newest = scope.chapters[scope.chapters.length - 1]
    expect(newest.messageIds).not.toContain('m3')
    expect(newest.messageIds.every((id) => ids.has(id))).toBe(true)
  })

  it('stops between passes when asked', async () => {
    seedThread(20)
    const summarize = stubSummarizer({
      onChapter: (call) => {
        if (call === 1) requestMemoryStop(THREAD)
      },
    })
    const result = await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })
    expect(result.stopped).toBe(true)
    expect(result.chaptersWritten).toBe(1)
  })

  // "Write now" reports what happened; with nothing owed it has to say so
  // rather than fall through silently.
  it('reports a manual run with nothing owed as idle', async () => {
    seedThread(2)
    const summarize = stubSummarizer()
    const result = await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })
    expect(result.didNothing).toBe(true)
    expect(summarize.calls.chapter).toBe(0)
  })

  // A summarizer that answers nothing once (a reasoning model spending its
  // whole budget on thinking) must not cost the chapter.
  it('retries an empty answer instead of losing the chapter', async () => {
    seedThread(8)
    let calls = 0
    const summarize = (async () => {
      calls++
      return calls === 1 ? '' : '- noted'
    }) as MemorySummarizer
    const result = await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })
    expect(calls).toBeGreaterThan(1)
    expect(result.chaptersWritten).toBe(1)
  }, 20000)

  it('auto runs only when the chat is effectively on', async () => {
    seedThread(12)
    // Untouched chat (no auto override) with the global default off: nothing.
    act(() => {
      const threads = useThreads.getState().threads
      useThreads.setState({
        threads: {
          ...threads,
          [THREAD]: {
            ...(threads[THREAD] as object),
            metadata: { memory: { sizes: { ...SIZES } } },
          } as never,
        },
      })
    })
    const summarize = stubSummarizer()
    const idle = await runMemoryFold({
      threadId: THREAD,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })
    expect(idle.didNothing).toBe(true)
    expect(summarize.calls.chapter).toBe(0)

    // Flip the global default: the same chat folds unasked.
    act(() => {
      useMemorySettings.setState({ autoDefault: true })
    })
    const auto = await runMemoryFold({
      threadId: THREAD,
      summarize,
      makeId: () => `ch${++idCounter}`,
    })
    expect(auto.chaptersWritten).toBe(2)
  })

  it('leaves non-roleplay threads alone', async () => {
    seedThread(12)
    act(() => {
      useCharacters.setState({
        characters: [{ id: 'char1', kind: 'assistant' } as never],
      })
    })
    const result = await runMemoryFold({
      threadId: THREAD,
      manual: true,
      summarize: stubSummarizer(),
      makeId: () => `ch${++idCounter}`,
    })
    expect(result.didNothing).toBe(true)
  })
})

describe('thread memory persistence', () => {
  afterEach(() => {
    act(() => {
      useThreadMemory.getState().clearAll()
    })
    localStorage.removeItem('thread-memory')
  })

  // Records must survive an app reload: chapters/canon come back per chat
  // while snapshots (in-memory by design) do not.
  it('round-trips records through storage across a reload', async () => {
    act(() => {
      useThreadMemory.getState().put('t1', {
        ...emptyThreadMemory(),
        pins: ['m1'],
      })
    })
    await act(async () => {
      await new Promise((r) => setTimeout(r, 20))
    })
    expect(localStorage.getItem('thread-memory')).toContain('"pins":["m1"]')

    // Simulate the reload with a fresh store: empty in-memory state,
    // rehydrated from storage (no clobbering write in between, the way
    // startup rehydration runs before anything renders).
    vi.resetModules()
    const { useThreadMemory: fresh } = await import('@/hooks/useThreadMemory')
    await act(async () => {
      await fresh.persist.rehydrate()
    })
    expect(fresh.getState().records['t1']?.pins).toEqual(['m1'])
  })
})

describe('stripReasoning', () => {  it('removes thinking blocks and stray tags', () => {
    expect(
      stripReasoning('<think>hmm</think>\n- noted one')
    ).toBe('- noted one')
    expect(stripReasoning('</think>\n+ character | Mira is here.')).toBe(
      '+ character | Mira is here.'
    )
    expect(stripReasoning('- plain')).toBe('- plain')
  })

  // A reasoning model that runs out of budget mid-thought answers with an
  // open tag and no close; filing that as the chapter would store its
  // deliberation as the story.
  it('treats an unterminated thinking block as no answer', () => {
    expect(stripReasoning('<think>still deciding, maybe the dock')).toBe('')
  })
})

import { describe, it, expect } from 'vitest'
import {
  applyCanonDiff,
  buildCanonPrompt,
  buildChapterPrompt,
  canonBudget,
  canonCandidates,
  canonPassBudget,
  chaptersForPath,
  chapterRange,
  chatAutoMode,
  coverageCursor,
  effectiveAutoOn,
  effectiveSizes,
  foldOutlook,
  foldableOwed,
  importCanonEntries,
  maxSummaryLines,
  messagesPerTurn,
  nextCanonId,
  normalizeSummary,
  ordinalOf,
  owedFrom,
  parseCanonDiff,
  planChapterBatches,
  planMemoryWindow,
  plannedBlocks,
  presetCost,
  restoreSameThread,
  rootIdOf,
  sizePresetFor,
  MEMORY_SIZE_PRESETS,
  type CanonEntry,
  type MemoryChapter,
  type MemoryPathMessage,
} from '../thread-memory'

const msg = (
  id: string,
  role = 'user',
  text = `text ${id}`
): MemoryPathMessage => ({ id, role, text })

const path10 = (): MemoryPathMessage[] =>
  Array.from({ length: 10 }, (_, i) =>
    msg(`m${i + 1}`, i % 2 === 0 ? 'user' : 'assistant')
  )

const chapter = (
  id: string,
  ids: string[],
  rootId = 'm1'
): MemoryChapter => ({
  id,
  rootId,
  messageIds: ids,
  hashes: {},
  text: `- chapter ${id}`,
  tokens: 10,
  created: 1,
})

describe('size presets', () => {
  it('labels exact bundles, custom otherwise', () => {
    expect(sizePresetFor(MEMORY_SIZE_PRESETS.medium)).toBe('medium')
    expect(sizePresetFor({ ...MEMORY_SIZE_PRESETS.medium, batch: 11 })).toBe(
      'custom'
    )
  })

  it('costs one chapter budget per kept chapter', () => {
    expect(presetCost(MEMORY_SIZE_PRESETS.medium)).toBe(25 * 250)
    expect(canonBudget(MEMORY_SIZE_PRESETS.medium)).toBe(25 * 250)
  })
})

describe('auto tri-state', () => {
  it('per-chat on/off wins in either direction', () => {
    expect(effectiveAutoOn({ memory: { auto: 'on' } }, false)).toBe(true)
    expect(effectiveAutoOn({ memory: { auto: 'off' } }, true)).toBe(false)
  })

  it('default falls through to the global', () => {
    expect(effectiveAutoOn({}, true)).toBe(true)
    expect(effectiveAutoOn({}, false)).toBe(false)
    expect(effectiveAutoOn(undefined, true)).toBe(true)
    expect(chatAutoMode({ memory: { auto: 'bogus' } })).toBe('default')
  })

  it('per-chat sizes win outright, else the global', () => {
    const global = MEMORY_SIZE_PRESETS.medium
    const override = MEMORY_SIZE_PRESETS.small
    expect(effectiveSizes({ memory: { sizes: override } }, global)).toBe(
      override
    )
    expect(effectiveSizes({}, global)).toBe(global)
    expect(effectiveSizes(undefined, global)).toBe(global)
  })
})

describe('coverage-derived owed', () => {
  it('whole batches only; remainder stays raw', () => {
    expect(plannedBlocks(10, 10)).toBe(1)
    expect(plannedBlocks(19, 10)).toBe(1)
    expect(plannedBlocks(9, 10)).toBe(0)
  })

  it('starts uncovered at the path start', () => {
    const path = path10()
    const { cursor, coveredCount } = coverageCursor(path, [], new Set())
    expect(cursor).toBe(0)
    expect(coveredCount).toBe(0)
    expect(owedFrom(path, cursor, new Set())).toHaveLength(10)
  })

  it('resumes after covered prefix', () => {
    const path = path10()
    const { cursor, coveredCount } = coverageCursor(
      path,
      [chapter('c1', path.slice(0, 10).map((m) => m.id))],
      new Set()
    )
    expect(cursor).toBe(10)
    expect(coveredCount).toBe(10)
  })

  it('pins and empties are never counted and never break contiguity', () => {
    const path = [
      msg('m1'),
      { ...msg('m2'), text: '   ' },
      msg('m3'),
      msg('m4'),
    ]
    const pins = new Set(['m3'])
    // Only m1 and m4 are countable.
    const { cursor } = coverageCursor(path, [chapter('c1', ['m1'])], pins)
    expect(cursor).toBe(3)
    expect(owedFrom(path, cursor, pins).map((m) => m.id)).toEqual(['m4'])
  })

  it('a chapter is valid only when every id is on the path', () => {
    const path = path10()
    const ids = new Set(path.map((m) => m.id))
    // Forked branch shares the first 5 messages only.
    const forkIds = new Set(path.slice(0, 5).map((m) => m.id))
    const ch = chapter('c1', path.slice(0, 10).map((m) => m.id))
    expect(chaptersForPath([ch], ids)).toHaveLength(1)
    expect(chaptersForPath([ch], forkIds)).toHaveLength(0)
    // Prefix chapter stays valid on the fork.
    const prefix = chapter('c0', path.slice(0, 4).map((m) => m.id))
    expect(chaptersForPath([prefix], forkIds)).toHaveLength(1)
  })

  it('a deleted middle invalidates the chapter; survivors become owed', () => {
    const path = path10()
    const ch = chapter('c1', path.map((m) => m.id))
    const spliced = path.filter((m) => m.id !== 'm5')
    const splicedIds = new Set(spliced.map((m) => m.id))
    expect(chaptersForPath([ch], splicedIds)).toHaveLength(0)
    const { cursor } = coverageCursor(spliced, [ch], new Set())
    expect(cursor).toBe(0)
  })

  it('plans consecutive whole batches of ids, never indices', () => {
    const path = path10()
    const batches = planChapterBatches(path, 0, new Set(), 10)
    expect(batches).toEqual([path.map((m) => m.id)])
    expect(planChapterBatches(path, 0, new Set(), 6)).toHaveLength(1)
  })

  it('keeps the raw window out of the foldable pool', () => {
    const path = path10()
    // 10 countable, keepRaw 4: 6 owed, no whole batch of 10.
    expect(foldableOwed(path, 0, new Set(), 4)).toHaveLength(6)
    expect(planChapterBatches(path, 0, new Set(), 5, 4)).toHaveLength(1)
    expect(planChapterBatches(path, 0, new Set(), 10, 4)).toHaveLength(0)
    // keepRaw past the end owes nothing, never negative.
    expect(foldableOwed(path, 0, new Set(), 99)).toHaveLength(0)
  })

  it('culls the oldest chapters in whole batches past keep', () => {
    const chapters = Array.from({ length: 35 }, (_, i) =>
      chapter(`c${i}`, [`m${i}`])
    )
    // keep 25 + take 10: exactly at the line culls the oldest 10.
    expect(canonCandidates(chapters, 25, 10).map((c) => c.id)).toEqual(
      chapters.slice(0, 10).map((c) => c.id)
    )
    // One short of the line: nothing.
    expect(canonCandidates(chapters.slice(0, 34), 25, 10)).toEqual([])
    // Culled chapters don't count toward the window.
    const culled = chapters.map((c, i) =>
      i < 10 ? { ...c, culled: true } : c
    )
    expect(canonCandidates(culled, 25, 10)).toEqual([])
  })

  it('scales the canon budget with the pass, clamped', () => {
    expect(canonPassBudget(250, 10)).toBe(750)
    expect(canonPassBudget(250, 1)).toBe(188)
    expect(canonPassBudget(250, 100)).toBe(4096)
  })

  it('withholds covered messages but keeps pins, windows chapters', () => {
    const path = path10()
    const chapters = [
      { ...chapter('c1', ['m1', 'm2', 'm3', 'm4']), culled: true },
      chapter('c2', ['m5', 'm6']),
    ]
    const scope = {
      chapters,
      canon: [
        {
          id: 'C1',
          cat: 'goal',
          text: 'Reach the archivist.',
          src: 'fold' as const,
          created: 1,
          updated: 1,
        },
      ],
      passes: [],
    }
    const w = planMemoryWindow({
      path,
      scope,
      sizes: { keepRaw: 4, keepChapters: 1, batch: 2, responseLength: 100 },
      pins: new Set(['m2']),
    })
    // Covered ids leave, except the pin and the root; culled still covers.
    expect([...w.excludedIds].sort()).toEqual(['m3', 'm4', 'm5', 'm6'])
    // Chapters block holds the newest live chapter only, numbered live.
    expect(w.chapters).toContain('### Chapter (messages 4-5)')
    expect(w.chapters).not.toContain('messages 0-3')
    expect(w.canon).toContain('[C1] goal | Reach the archivist.')
  })

  // The transport folds every leading assistant into the system string, so a
  // window that starts on an assistant reply would send it as a second
  // greeting. Both fixes are load-bearing there: the root never leaves, and
  // the cut lands on a user turn.
  it('keeps the root and starts the window on a user turn', () => {
    const path = [
      msg('m0', 'assistant'),
      msg('m1', 'user'),
      msg('m2', 'assistant'),
      msg('m3', 'user'),
      msg('m4', 'assistant'),
      msg('m5', 'user'),
    ]
    const w = planMemoryWindow({
      path,
      // An odd batch cuts mid-turn: m0-m2 covered leaves m3 first, but m0-m3
      // covered would leave the assistant m4 first.
      scope: {
        chapters: [chapter('c1', ['m0', 'm1', 'm2', 'm3'], 'm0')],
        canon: [],
        passes: [],
      },
      sizes: { keepRaw: 0, keepChapters: 5, batch: 4, responseLength: 100 },
      pins: new Set(),
    })
    expect([...w.excludedIds].sort()).toEqual(['m1', 'm2'])
  })

  it('windows nothing on an empty scope', () => {
    const w = planMemoryWindow({
      path: path10(),
      scope: { chapters: [], canon: [], passes: [] },
      sizes: MEMORY_SIZE_PRESETS.medium,
      pins: new Set(),
    })
    expect(w.excludedIds.size).toBe(0)
    expect(w.canon).toBe('')
    expect(w.chapters).toBe('')
  })
})

describe('turn counting', () => {
  it('foldOutlook counts in turns, first turn t with uncovered + step*t >= batch + keepRaw', () => {
    // batch 10, nothing uncovered, 2/turn: turns 5, writes 1 at 10.
    expect(foldOutlook({ uncovered: 0, batch: 10, perTurn: 2 })).toMatchObject({
      turns: 5,
      blocks: 1,
      at: 10,
    })
    // 8 uncovered: next turn trips it.
    expect(foldOutlook({ uncovered: 8, batch: 10, perTurn: 2 })).toMatchObject({
      turns: 1,
      blocks: 1,
      at: 10,
    })
  })

  it('foldOutlook counts the raw window in, so the line moves from message one', () => {
    // Medium: the fold trips at batch + keepRaw = 35 countable messages.
    const at = (uncovered: number) =>
      foldOutlook({ uncovered, batch: 10, keepRaw: 25, perTurn: 2 }).turns
    expect(at(1)).toBe(17)
    expect(at(3)).toBe(16)
    expect(at(25)).toBe(5)
    expect(at(33)).toBe(1)
    // A backlog past the trip point still reads as "next reply".
    expect(at(60)).toBe(1)
    expect(
      foldOutlook({ uncovered: 33, batch: 10, keepRaw: 25, perTurn: 2 }).blocks
    ).toBe(1)
  })

  it('messagesPerTurn measures the last finished turn, never below 2', () => {
    expect(messagesPerTurn(['user', 'assistant'])).toBe(2)
    expect(messagesPerTurn(['user', 'assistant', 'user'])).toBe(2)
    expect(
      messagesPerTurn(['user', 'assistant', 'assistant', 'assistant', 'user'])
    ).toBe(4)
    expect(messagesPerTurn([])).toBe(2)
    expect(messagesPerTurn(['assistant'])).toBe(2)
  })
})

describe('ordinals', () => {
  it('are 1-based and live-resolved', () => {
    const path = path10()
    expect(ordinalOf(path, 'm1')).toBe(1)
    expect(ordinalOf(path, 'm10')).toBe(10)
    expect(ordinalOf(path, 'nope')).toBe(-1)
    expect(chapterRange(chapter('c1', ['m3', 'm4', 'm5']), path)).toEqual({
      from: 3,
      to: 5,
    })
    expect(chapterRange(chapter('c1', ['gone']), path)).toBeNull()
  })

  it('rootIdOf takes the path head', () => {
    expect(rootIdOf(path10())).toBe('m1')
    expect(rootIdOf([])).toBe('')
  })
})

describe('parseCanonDiff', () => {
  it('parses adds and revisions', () => {
    const diff = parseCanonDiff(
      '+ character | Mira distrusts the harbor patrol.\n~ C4 | Toren paid his debt.'
    )
    expect(diff.adds).toEqual([
      { cat: 'character', text: 'Mira distrusts the harbor patrol.' },
    ])
    expect(diff.revisions).toEqual([{ id: 'C4', text: 'Toren paid his debt.' }])
  })

  it('accepts a bare category | fact line for known categories only', () => {
    const diff = parseCanonDiff('goal | Reach the archivist.\nprose | with a pipe')
    expect(diff.adds).toEqual([{ cat: 'goal', text: 'Reach the archivist.' }])
  })

  it('files unknown categories under misc and skips refusals', () => {
    const diff = parseCanonDiff(
      '+ starship | The Dawn sails at dusk.\n(nothing durable)\nLet me know if you need more.'
    )
    expect(diff.adds).toEqual([
      { cat: 'misc', text: 'The Dawn sails at dusk.' },
    ])
  })
})

describe('applyCanonDiff', () => {
  const base: CanonEntry[] = [
    {
      id: 'C1',
      cat: 'goal',
      text: 'Reach the archivist.',
      src: 'fold',
      created: 1,
      updated: 1,
    },
    {
      id: 'C2',
      cat: 'character',
      text: 'Mira is a smuggler.',
      src: 'manual',
      created: 1,
      updated: 1,
    },
  ]

  it('revises folded entries but never manual ones', () => {
    const { canon, revised } = applyCanonDiff(
      base,
      {
        adds: [],
        revisions: [
          { id: 'C1', text: 'Reached the archivist.' },
          { id: 'C2', text: 'Mira is a pirate.' },
        ],
      },
      2
    )
    expect(revised).toBe(1)
    expect(canon.find((e) => e.id === 'C1')?.text).toBe('Reached the archivist.')
    expect(canon.find((e) => e.id === 'C2')?.text).toBe('Mira is a smuggler.')
  })

  it('skips exact-text duplicates case-insensitively', () => {
    const { canon, added } = applyCanonDiff(
      base,
      { adds: [{ cat: 'goal', text: 'reach the ARCHIVIST.' }], revisions: [] },
      2
    )
    expect(added).toBe(0)
    expect(canon).toHaveLength(2)
    expect(nextCanonId(base)).toBe('C3')
  })
})

describe('prompts and summaries', () => {
  it('caps summary lines by block size and response length', () => {
    expect(maxSummaryLines(10, 250)).toBe(8)
    expect(maxSummaryLines(100, 250)).toBe(12)
    expect(maxSummaryLines(2, 250)).toBe(3)
  })

  it('normalizes summaries to one "- " line per event', () => {
    expect(
      normalizeSummary('1. First thing\n* Second thing\nLet me know if more.')
    ).toBe('- First thing\n- Second thing')
  })

  it('drops the prior section when empty', () => {
    const p = buildChapterPrompt('- a\n- b', { maxLines: 8 })
    expect(p).toContain('at most 8 "- " lines')
    expect(p).not.toContain('{{prior}}')
    expect(p).not.toContain('Prior story notes')
  })

  it('fills the canon template', () => {
    const p = buildCanonPrompt('[C1] goal | x', '- chapter one')
    expect(p).toContain('[C1] goal | x')
    expect(p).toContain('- chapter one')
    expect(p).not.toContain('{{canon}}')
  })
})

describe('export and import', () => {
  it('cross-thread import takes canon with fresh ids, skipping dupes', () => {
    const canon: CanonEntry[] = [
      {
        id: 'C1',
        cat: 'goal',
        text: 'Reach the archivist.',
        src: 'fold',
        created: 1,
        updated: 1,
      },
    ]
    const { canon: next, imported } = importCanonEntries(
      canon,
      {
        kind: 'kuru-thread-memory',
        v: 1,
        threadId: 'other',
        exportedAt: 2,
        chapters: [],
        canon: [
          {
            id: 'C9',
            cat: 'goal',
            text: 'reach the archivist.',
            src: 'fold',
            created: 1,
            updated: 1,
          },
          {
            id: 'C10',
            cat: 'secret',
            text: 'The vault is empty.',
            src: 'fold',
            created: 1,
            updated: 1,
          },
        ],
      },
      3
    )
    expect(imported).toBe(1)
    expect(next.map((e) => e.id)).toEqual(['C1', 'C2'])
  })

  it('same-thread restore carries manual entries forward', () => {
    const current = {
      chapters: [],
      canon: [
        {
          id: 'C5',
          cat: 'secret',
          text: 'Hand-written truth.',
          src: 'manual' as const,
          created: 1,
          updated: 1,
        },
      ],
      passes: [],
    }
    const { canon, carried } = restoreSameThread(current, {
      chapters: [],
      canon: [],
    })
    expect(carried).toBe(1)
    expect(canon.map((e) => e.text)).toEqual(['Hand-written truth.'])
  })
})

import { describe, it, expect } from 'vitest'
import {
  applyDepthInjections,
  chatScanText,
  defaultLorebookRuntimeSettings,
  entryMatches,
  groupActivations,
  scanLorebooks,
} from '../lorebook-runtime'
import {
  LorebookLogic,
  LorebookPosition,
  type Lorebook,
  type LorebookEntry,
} from '../lorebook'

const entry = (over: Partial<LorebookEntry> = {}): LorebookEntry => ({
  id: over.id ?? 'e1',
  keys: [],
  secondaryKeys: [],
  selectiveLogic: LorebookLogic.AND_ANY,
  content: 'content',
  comment: '',
  enabled: true,
  constant: true,
  position: LorebookPosition.beforeChar,
  depth: 4,
  role: null,
  order: 100,
  probability: 100,
  caseSensitive: false,
  extra: {},
  ...over,
})

const book = (
  entries: LorebookEntry[],
  over: Partial<Lorebook> = {}
): Lorebook => ({
  id: 'b1',
  name: 'Book',
  description: '',
  entries,
  scanDepth: null,
  tokenBudget: null,
  recursiveScanning: false,
  global: false,
  ...over,
})

const settings = defaultLorebookRuntimeSettings

describe('entryMatches', () => {
  it('fires with no keys at all', () => {
    expect(entryMatches(entry(), 'anything')).toBe(true)
  })

  it('matches plain keys on word boundaries, ignoring case', () => {
    const e = entry({ keys: ['orc'] })
    expect(entryMatches(e, 'an Orc appears')).toBe(true)
    expect(entryMatches(e, 'past the orchard')).toBe(false)
  })

  it('keeps an imported entry’s whole-word override', () => {
    const loose = entry({ keys: ['orc'], extra: { matchWholeWords: false } })
    expect(entryMatches(loose, 'past the orchard')).toBe(true)
  })

  it('uses a /pattern/flags key exactly as written', () => {
    expect(entryMatches(entry({ keys: ['/gold(en)? crown/'] }), 'the golden crown')).toBe(true)
    expect(entryMatches(entry({ keys: ['/gold(en)? crown/'] }), 'the crown')).toBe(false)
    // No implicit `i`: a bare pattern is case-sensitive, an `/…/i` one is not.
    expect(entryMatches(entry({ keys: ['/rune/'] }), 'a Rune')).toBe(false)
    expect(entryMatches(entry({ keys: ['/rune/i'] }), 'a Rune')).toBe(true)
  })

  it('keeps an imported entry’s case sensitivity for plain keys', () => {
    expect(entryMatches(entry({ keys: ['Rune'] }), 'a rune')).toBe(true)
    expect(
      entryMatches(entry({ keys: ['Rune'], caseSensitive: true }), 'a rune')
    ).toBe(false)
  })

  it('applies the selective logic to secondary keys', () => {
    const base = { keys: ['castle'], secondaryKeys: ['night', 'storm'] }
    const text = 'the castle at night'
    expect(
      entryMatches(
        entry({ ...base, selectiveLogic: LorebookLogic.AND_ANY }),
        text
      )
    ).toBe(true)
    expect(
      entryMatches(
        entry({ ...base, selectiveLogic: LorebookLogic.AND_ALL }),
        text
      )
    ).toBe(false)
    expect(
      entryMatches(
        entry({ ...base, selectiveLogic: LorebookLogic.NOT_ANY }),
        text
      )
    ).toBe(false)
    expect(
      entryMatches(
        entry({ ...base, selectiveLogic: LorebookLogic.NOT_ALL }),
        text
      )
    ).toBe(true)
  })
})

describe('scanLorebooks', () => {
  it('skips disabled and contentless entries', () => {
    const { activated } = scanLorebooks({
      books: [
        book([
          entry({ id: 'off', enabled: false }),
          entry({ id: 'blank', content: '   ' }),
          entry({ id: 'live' }),
        ]),
      ],
      chatText: 'hello',
      settings,
    })
    expect(activated.map((a) => a.entry.id)).toEqual(['live'])
  })

  it('resolves {{char}} and {{user}} in entry content', () => {
    const { activated } = scanLorebooks({
      books: [book([entry({ content: '{{char}} knows {{user}}.' })])],
      chatText: 'hello',
      settings,
      vars: { char: 'Aria', user: 'Sam' },
    })
    expect(activated[0].text).toBe('Aria knows Sam.')
  })

  it('drops the least prominent entries to fit the budget, never all of them', () => {
    const long = 'word '.repeat(200)
    const { activated, droppedForBudget } = scanLorebooks({
      books: [
        book([
          entry({ id: 'low', order: 1, content: long }),
          entry({ id: 'high', order: 900, content: long }),
        ]),
      ],
      chatText: 'hello',
      settings,
      budgetTokens: 50,
    })
    expect(activated.map((a) => a.entry.id)).toEqual(['high'])
    expect(droppedForBudget.map((a) => a.entry.id)).toEqual(['low'])
  })

  it('activates recursively only for a book that allows it, honouring the ST flags', () => {
    const entries = () => [
      entry({ id: 'seed', keys: ['tavern'], content: 'The Rusty Anchor.' }),
      entry({ id: 'chain', keys: ['anchor'], content: 'A sailor bar.' }),
    ]
    const chatText = 'we enter the tavern'

    expect(
      scanLorebooks({ books: [book(entries())], chatText, settings }).activated.map(
        (a) => a.entry.id
      )
    ).toEqual(['seed'])

    const recursive = scanLorebooks({
      books: [book(entries(), { recursiveScanning: true })],
      chatText,
      settings,
    })
    expect(recursive.activated.map((a) => a.entry.id).sort()).toEqual([
      'chain',
      'seed',
    ])

    const excluded = scanLorebooks({
      books: [
        book(
          [
            entry({
              id: 'seed',
              keys: ['tavern'],
              content: 'The Rusty Anchor.',
              extra: { excludeRecursion: true },
            }),
            entry({ id: 'chain', keys: ['anchor'], content: 'A sailor bar.' }),
          ],
          { recursiveScanning: true }
        ),
      ],
      chatText,
      settings,
    })
    expect(excluded.activated.map((a) => a.entry.id)).toEqual(['seed'])

    // A book that opted out is not dragged along by one that opted in.
    const mixed = scanLorebooks({
      books: [
        book([entry({ id: 'seed', keys: ['tavern'], content: 'The Rusty Anchor.' })], {
          recursiveScanning: true,
        }),
        book([entry({ id: 'quiet', keys: ['anchor'], content: 'A sailor bar.' })]),
      ],
      chatText,
      settings,
    })
    expect(mixed.activated.map((a) => a.entry.id)).toEqual(['seed'])
  })
})

describe('groupActivations', () => {
  it('routes unsupported positions to a slot that exists', () => {
    const { activated } = scanLorebooks({
      books: [
        book([
          entry({ id: 'an', position: LorebookPosition.ANTop }),
          entry({ id: 'outlet', position: LorebookPosition.outlet }),
        ]),
      ],
      chatText: 'hello',
      settings,
    })
    const grouped = groupActivations(activated)
    // Author's note becomes at-depth, outlet becomes before-character.
    expect([...grouped.atDepth.values()].flat().map((a) => a.entry.id)).toEqual(
      ['an']
    )
    expect(grouped.beforeChar.map((a) => a.entry.id)).toEqual(['outlet'])
  })
})

describe('applyDepthInjections', () => {
  const messages = [
    { role: 'user', content: 'u1' },
    { role: 'assistant', content: 'a1' },
    { role: 'user', content: 'u2' },
  ]
  const at = (depth: number) =>
    groupActivations(
      scanLorebooks({
        books: [
          book([
            entry({ position: LorebookPosition.atDepth, depth, content: 'WI' }),
          ]),
        ],
        chatText: 'hello',
        settings,
      }).activated
    ).atDepth

  it('keeps the role sequence intact by merging instead of inserting a turn', () => {
    const out = applyDepthInjections(messages, at(1))
    expect(out.messages.map((m) => m.role)).toEqual([
      'user',
      'assistant',
      'user',
    ])
    expect(out.messages[2].content).toBe('WI\n\nu2')
  })

  it('appends to the last message at depth 0', () => {
    expect(applyDepthInjections(messages, at(0)).messages[2].content).toBe(
      'u2\n\nWI'
    )
  })

  it('collapses a depth past the window onto the oldest message', () => {
    expect(applyDepthInjections(messages, at(99)).messages[0].content).toBe(
      'WI\n\nu1'
    )
  })

  it('reports entries as unplaced when there is no message to merge into', () => {
    const out = applyDepthInjections([], at(1))
    expect(out.unplaced).toHaveLength(1)
  })
})

describe('chatScanText', () => {
  it('reads only the last N messages, and only their text parts', () => {
    const messages = [
      { parts: [{ type: 'text', text: 'oldest' }] },
      { parts: [{ type: 'reasoning', text: 'scratch' }] },
      { parts: [{ type: 'text', text: 'newest' }] },
    ]
    expect(chatScanText(messages, 2)).toBe('newest')
    expect(chatScanText(messages, 5)).toBe('oldest\nnewest')
  })
})

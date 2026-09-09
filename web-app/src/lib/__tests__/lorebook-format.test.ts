import { describe, it, expect } from 'vitest'
import { normalizeLorebook, lorebookToCardBook } from '../lorebook'

describe('normalizeLorebook re-reading our own saves', () => {
  it('keeps book-level settings saved under our own names', () => {
    const book = normalizeLorebook(
      {
        name: 'Book',
        recursiveScanning: true,
        scanDepth: 6,
        entries: [{ id: '1', keys: ['a'], content: 'c' }],
      },
      { id: 'b' }
    )!
    expect(book.recursiveScanning).toBe(true)
    expect(book.scanDepth).toBe(6)
  })


  it('keeps secondary keys and does not nest the extra bag', () => {
    const saved = {
      name: 'Book',
      entries: [
        {
          id: '1',
          keys: ['castle'],
          secondaryKeys: ['night'],
          content: 'c',
          depth: 2,
          role: 1,
          // What earlier saves left behind: the real ST flags buried under
          // layers of the entry's own fields.
          extra: {
            secondaryKeys: [],
            depth: 4,
            extra: { excludeRecursion: true, priority: 10 },
          },
        },
      ],
    }

    const once = normalizeLorebook(saved, { id: 'b', name: 'Book' })!
    const entry = once.entries[0]
    expect(entry.secondaryKeys).toEqual(['night'])
    expect(entry.depth).toBe(2)
    expect(entry.extra).toEqual({ excludeRecursion: true, priority: 10 })

    // Round-tripping the normalized book must not grow it again.
    const twice = normalizeLorebook(JSON.parse(JSON.stringify(once)), {
      id: 'b',
      name: 'Book',
    })!
    expect(twice.entries[0]).toEqual(entry)
    expect(lorebookToCardBook(twice).entries).toEqual(
      lorebookToCardBook(once).entries
    )
  })
})

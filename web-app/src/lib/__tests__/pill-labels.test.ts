import { describe, it, expect } from 'vitest'
import { labelForms, fitPillLabels, type PillMetrics } from '../pill-labels'

describe('labelForms', () => {
  it('breaks a name with spaces on its spaces, keeping inner hyphens', () => {
    expect(labelForms('Arkan - Fantasy World')).toEqual([
      '',
      'Arkan',
      'Arkan - Fantasy',
      'Arkan - Fantasy World',
    ])
    expect(labelForms('Jean-Luc Picard')).toEqual([
      '',
      'Jean-Luc',
      'Jean-Luc Picard',
    ])
  })

  it('breaks a model id on its own separators', () => {
    expect(labelForms('Gemma-3-1b-It')).toEqual([
      '',
      'Gemma',
      'Gemma-3',
      'Gemma-3-1b',
      'Gemma-3-1b-It',
    ])
  })

  it('gives a single word one form, and nothing an empty one', () => {
    expect(labelForms('Wren')).toEqual(['', 'Wren'])
    expect(labelForms('')).toEqual([''])
    expect(labelForms(undefined)).toEqual([''])
  })
})

/** One pixel per character, no fixed width, so widths read at a glance. */
const metrics = (forms: string[], fixed = 0): PillMetrics => ({
  forms,
  formWidths: forms.map((f) => f.length),
  fixed,
  labelGap: 0,
})

describe('fitPillLabels', () => {
  const pills = () =>
    new Map([
      ['model' as const, metrics(labelForms('Gemma-3-1b-It-Glm'))],
      ['character' as const, metrics(labelForms('Arkan - Fantasy World'))],
      ['persona' as const, metrics(labelForms('Wren'))],
      ['lorebook' as const, metrics(['', '1', '1 lorebook'])],
    ])

  it('leaves every label whole at each width it picks', () => {
    for (let available = 0; available < 80; available++) {
      for (const [key, label] of fitPillLabels(pills(), available, 0)) {
        expect(pills().get(key)!.forms).toContain(label)
      }
    }
  })

  it('grows the labels back in priority order as room appears', () => {
    const at = (available: number) =>
      Object.fromEntries(fitPillLabels(pills(), available, 0))

    // The model's first word and the lorebook count come before any name.
    expect(at(6)).toMatchObject({
      model: 'Gemma',
      character: '',
      persona: '',
      lorebook: '1',
    })
    // Then the two names, with no room left for a second model segment.
    expect(at(16)).toMatchObject({
      model: 'Gemma',
      character: 'Arkan',
      persona: 'Wren',
      lorebook: '1',
    })
    // The model fills out before the lorebook spells its label.
    expect(at(30).model).toBe('Gemma-3-1b')
    // Everything, given room for everything.
    expect(at(500)).toMatchObject({
      model: 'Gemma-3-1b-It-Glm',
      character: 'Arkan - Fantasy World',
      persona: 'Wren',
      lorebook: '1 lorebook',
    })
  })

  it('lets a later pill grow past one that does not fit', () => {
    const long = new Map([
      ['character' as const, metrics(['', 'Arkan', 'A'.repeat(50)])],
      ['persona' as const, metrics(['', 'Wren'])],
    ])
    const chosen = fitPillLabels(long, 12, 0)
    expect(chosen.get('character')).toBe('Arkan')
    expect(chosen.get('persona')).toBe('Wren')
  })

  it('charges the fixed width of pills that have no label', () => {
    const tight = new Map([
      ['model' as const, metrics(['', 'Gemma'], 10)],
      ['character' as const, metrics(['', 'Arkan'], 10)],
    ])
    expect(fitPillLabels(tight, 20, 0).get('model')).toBe('')
    expect(fitPillLabels(tight, 25, 0).get('model')).toBe('Gemma')
  })
})

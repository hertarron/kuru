import { describe, it, expect } from 'vitest'
import {
  stripOoc,
  trimIncompleteSentence,
  cleanPriorTurns,
  defaultRpTextSettings,
} from '../rp-text'

describe('stripOoc', () => {
  it('removes the three OOC notations in circulation', () => {
    expect(stripOoc('She nods.\n\n((nice pacing))')).toBe('She nods.')
    expect(stripOoc('She nods.\n\n[OOC: slow down]')).toBe('She nods.')
    expect(stripOoc('She nods.\n\n<ooc>slow down</ooc>')).toBe('She nods.')
  })

  it('leaves single parentheses alone', () => {
    // Ordinary prose uses these constantly; only the doubled form is an aside.
    const prose = 'She nods (slowly) and turns away.'
    expect(stripOoc(prose)).toBe(prose)
  })

  it('does not let an unclosed (( eat the rest of the message', () => {
    const text = 'She nods. ((oops\n\nShe reaches for the door.'
    expect(stripOoc(text)).toContain('She reaches for the door.')
  })
})

describe('trimIncompleteSentence', () => {
  it('drops a fragment left by a truncated reply', () => {
    expect(trimIncompleteSentence('She nods. She reached for the doo')).toBe(
      'She nods.'
    )
  })

  it('keeps a short unpunctuated line, which is normal RP prose', () => {
    // An action beat with no full stop is a deliberate way to write and must
    // survive; only a fragment trailing a real sentence is truncation.
    const beat = '*She reaches for the door*'
    expect(trimIncompleteSentence(beat)).toBe(beat)
  })

  it('keeps a closing quote that follows the final punctuation', () => {
    expect(trimIncompleteSentence('"Get back," she said. Then she tur')).toBe(
      '"Get back," she said.'
    )
  })
})

describe('cleanPriorTurns', () => {
  const settings = { ...defaultRpTextSettings, hideOocFromPrompt: true }

  it('never strips the turn being sent', () => {
    // The whole point of OOC is asking the model something directly; stripping
    // the current turn would make the feature silently swallow the question.
    const messages = [
      { role: 'user', content: 'Hi. ((be brief))' },
      { role: 'assistant', content: 'She nods. ((sure))' },
      { role: 'user', content: '((why did she leave?))' },
    ]
    const out = cleanPriorTurns(messages, settings)
    expect(out[0].content).toBe('Hi.')
    expect(out[1].content).toBe('She nods.')
    expect(out[2].content).toBe('((why did she leave?))')
  })

  it('leaves everything after the last user message alone', () => {
    // A continue-prefill belongs to the turn being generated.
    const messages = [
      { role: 'user', content: 'Go on. ((short))' },
      { role: 'assistant', content: 'She rea ((cut off))' },
    ]
    const out = cleanPriorTurns(messages, settings)
    expect(out[1].content).toBe('She rea ((cut off))')
  })

  it('cleans text parts inside array content', () => {
    const messages = [
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'She nods. ((aside))' }],
      },
      { role: 'user', content: 'And then?' },
    ]
    const out = cleanPriorTurns(messages, settings)
    expect((out[0].content as Array<{ text: string }>)[0].text).toBe(
      'She nods.'
    )
  })

  it('is a no-op when both cleanups are off', () => {
    const messages = [
      { role: 'assistant', content: 'She nods. ((aside))' },
      { role: 'user', content: 'And then?' },
    ]
    const out = cleanPriorTurns(messages, {
      hideOocFromPrompt: false,
      trimIncompleteSentence: false,
    })
    expect(out).toBe(messages)
  })
})

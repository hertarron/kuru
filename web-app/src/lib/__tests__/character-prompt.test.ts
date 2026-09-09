import { describe, it, expect } from 'vitest'
import { buildCharacterSystemPrompt } from '../character-prompt'

describe('buildCharacterSystemPrompt', () => {
  it('returns undefined when no character is given', () => {
    expect(buildCharacterSystemPrompt(undefined)).toBeUndefined()
  })

  it('returns undefined when all prompt fields are empty', () => {
    const character = {
      id: 'c1',
      name: 'Empty',
      created_at: 1,
      instructions: '',
      parameters: {},
    }
    const result = buildCharacterSystemPrompt(character as never)
    expect(result).toBeUndefined()
  })

  it('returns undefined for whitespace-only fields', () => {
    const character = {
      id: 'c2',
      name: 'Blank',
      created_at: 1,
      description: '   ',
      personality: '\n\t',
      instructions: '',
      parameters: {},
    }
    expect(buildCharacterSystemPrompt(character as never)).toBeUndefined()
  })

  it('returns the trimmed description when only a description exists', () => {
    const character = {
      id: 'c3',
      name: 'Desc',
      created_at: 1,
      description: '  A quiet archivist.  ',
      instructions: '',
      parameters: {},
    }
    expect(buildCharacterSystemPrompt(character as never)).toBe(
      'A quiet archivist.'
    )
  })

  it('composes sections in order: description, personality, scenario', () => {
    const character = {
      id: 'c4',
      name: 'Full',
      created_at: 1,
      description: 'A knight.',
      personality: 'Brave and stubborn.',
      scenario: 'The kingdom is at war.',
      instructions: '',
      parameters: {},
    }
    const out = buildCharacterSystemPrompt(character as never)
    expect(out).toBe(
      'A knight.\n\nPersonality: Brave and stubborn.\n\nScenario: The kingdom is at war.'
    )
  })

  it('appends legacy instructions after the scenario', () => {
    const character = {
      id: 'c5',
      name: 'Legacy',
      created_at: 1,
      description: 'D.',
      scenario: 'S.',
      instructions: 'Always answer in rhyme.',
      parameters: {},
    }
    const out = buildCharacterSystemPrompt(character as never)
    expect(out).toBe('D.\n\nScenario: S.\n\nAlways answer in rhyme.')
  })

  it('appends mes_example last as an Example dialogue block', () => {
    const character = {
      id: 'c6',
      name: 'Chatty',
      created_at: 1,
      description: 'D.',
      personality: 'P.',
      scenario: 'S.',
      instructions: 'I.',
      mes_example: '  <START>\n{{user}}: hi\n{{char}}: hello  ',
      parameters: {},
    }
    const out = buildCharacterSystemPrompt(character as never)
    expect(out).toBe(
      [
        'D.',
        'Personality: P.',
        'Scenario: S.',
        'I.',
        'Example dialogue:\n<START>\n{{user}}: hi\nChatty: hello',
      ].join('\n\n')
    )
  })

  it('leads with system_prompt but still composes the character definition', () => {
    const character = {
      id: 'c7',
      name: 'Override',
      created_at: 1,
      description: 'Card description.',
      personality: 'Card personality.',
      scenario: 'Card scenario.',
      instructions: '',
      mes_example: '<START>\n{{user}}: hi',
      system_prompt: '  Style note.  ',
      parameters: {},
    }
    const out = buildCharacterSystemPrompt(character as never)
    expect(out).toContain('Style note.')
    // Card definition sections follow the style note.
    expect(out.indexOf('Style note.')).toBeLessThan(
      out.indexOf('Card description.')
    )
    expect(out).toContain('Personality: Card personality.')
    expect(out).toContain('Scenario: Card scenario.')
    expect(out).toContain('Example dialogue:')
  })

  it('renders {{current_date}} to include the current year', () => {
    const character = {
      id: 'c8',
      name: 'Dated',
      created_at: 1,
      description: 'Today is {{current_date}}.',
      instructions: '',
      parameters: {},
    }
    const out = buildCharacterSystemPrompt(character as never)
    expect(out).toContain(String(new Date().getFullYear()))
    expect(out).not.toContain('{{current_date}}')
  })

  it('replaces every {{char}} occurrence with the character name', () => {
    const character = {
      id: 'c9',
      name: 'Seraphine',
      created_at: 1,
      description:
        '{{char}} is a seer. Speak as {{char}}, act as {{CHAR}} would.',
      instructions: '',
      parameters: {},
    }
    const out = buildCharacterSystemPrompt(character as never)
    expect(out).toBe(
      'Seraphine is a seer. Speak as Seraphine, act as Seraphine would.'
    )
  })

  it('replaces {{user}} with the provided userName', () => {
    const character = {
      id: 'c10',
      name: 'Guide',
      created_at: 1,
      description: 'Greet {{user}} warmly.',
      instructions: '',
      parameters: {},
    }
    const out = buildCharacterSystemPrompt(character as never, {
      userName: 'Alex',
    })
    expect(out).toBe('Greet Alex warmly.')
  })

  it('leaves macros literal when no opts are provided', () => {
    const character = {
      id: 'c11',
      name: 'Literalist',
      created_at: 1,
      description: '{{user}} meets {{char}}.',
      instructions: '',
      parameters: {},
    }
    const out = buildCharacterSystemPrompt(character as never)
    expect(out).toBe('{{user}} meets Literalist.')
  })
})

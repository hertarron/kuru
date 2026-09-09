import { describe, it, expect } from 'vitest'
import {
  isPngBuffer,
  normalizeCardJson,
  parseCharacterCard,
  cardToCharacter,
  migrateAssistantToCharacter,
  inferCharacterKind,
} from '../character-card'

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

function crc32(bytes: Uint8Array): number {
  let c: number
  const table: number[] = []
  for (let n = 0; n < 256; n++) {
    c = n
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    }
    table[n] = c
  }
  let crc = 0xffffffff
  for (const b of bytes) crc = table[(crc ^ b) & 0xff] ^ (crc >>> 8)
  return (crc ^ 0xffffffff) >>> 0
}

function makeChunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length)
  const dv = new DataView(out.buffer)
  dv.setUint32(0, data.length)
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i)
  out.set(data, 8)
  const crcInput = new Uint8Array(4 + data.length)
  for (let i = 0; i < 4; i++) crcInput[i] = type.charCodeAt(i)
  crcInput.set(data, 4)
  dv.setUint32(8 + data.length, crc32(crcInput))
  return out
}

function textChunkData(keyword: string, payload: string): Uint8Array {
  const keywordBytes = new TextEncoder().encode(keyword)
  const payloadBytes = new TextEncoder().encode(payload)
  const out = new Uint8Array(keywordBytes.length + 1 + payloadBytes.length)
  out.set(keywordBytes, 0)
  out[keywordBytes.length] = 0
  out.set(payloadBytes, keywordBytes.length + 1)
  return out
}

function buildPng(chunks: Uint8Array[]): ArrayBuffer {
  const parts: Uint8Array[] = [new Uint8Array(PNG_SIGNATURE), ...chunks]
  const total = parts.reduce((sum, p) => sum + p.length, 0)
  const out = new Uint8Array(total)
  let offset = 0
  for (const p of parts) {
    out.set(p, offset)
    offset += p.length
  }
  return out.buffer
}

function base64(json: string): string {
  return Buffer.from(json, 'utf-8').toString('base64')
}

const v2Card = {
  spec: 'chara_card_v2',
  spec_version: '2.0',
  data: {
    name: 'Aria',
    description: 'A wandering mage',
    personality: 'curious',
    scenario: 'a tavern at dusk',
    first_mes: 'Hello there!',
    alternate_greetings: ['Hi!', 42, null] as unknown[],
    mes_example: '<START>',
    system_prompt: 'You are Aria.',
    creator: 'someone',
    tags: ['mage', 7] as unknown[],
  },
}

describe('isPngBuffer', () => {
  it('returns true for a buffer starting with the PNG signature', () => {
    expect(isPngBuffer(buildPng([]))).toBe(true)
  })

  it('returns false for a wrong signature', () => {
    const buf = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x00, 0x00, 0x00, 0x00])
    expect(isPngBuffer(buf.buffer)).toBe(false)
  })

  it('returns false for a short buffer', () => {
    const buf = new Uint8Array(PNG_SIGNATURE.slice(0, 4))
    expect(isPngBuffer(buf.buffer)).toBe(false)
  })

  it('returns false for JSON text bytes', () => {
    const buf = new TextEncoder().encode('{"name":"x"}')
    expect(isPngBuffer(buf.buffer as ArrayBuffer)).toBe(false)
  })
})

describe('normalizeCardJson', () => {
  it('unwraps a V2 envelope', () => {
    const card = normalizeCardJson(v2Card)
    expect(card).not.toBeNull()
    expect(card?.name).toBe('Aria')
    expect(card?.description).toBe('A wandering mage')
  })

  it('unwraps a V3 envelope', () => {
    const card = normalizeCardJson({
      spec: 'chara_card_v3',
      spec_version: '3.0',
      data: { name: 'Vex' },
    })
    expect(card?.name).toBe('Vex')
  })

  it('accepts a flat V1 object', () => {
    const card = normalizeCardJson({ name: 'Old Bot', description: 'v1' })
    expect(card?.name).toBe('Old Bot')
    expect(card?.description).toBe('v1')
  })

  it('returns null for null input', () => {
    expect(normalizeCardJson(null)).toBeNull()
  })

  it('returns null for non-object input', () => {
    expect(normalizeCardJson('hello')).toBeNull()
    expect(normalizeCardJson(42)).toBeNull()
  })

  it('returns null when no name and no prompt-bearing fields', () => {
    expect(normalizeCardJson({ creator: 'x', tags: ['a'] })).toBeNull()
  })

  it('keeps a card with only a name', () => {
    expect(normalizeCardJson({ name: 'Just Name' })?.name).toBe('Just Name')
  })

  it('keeps a card with only a description and empty name', () => {
    const card = normalizeCardJson({ description: 'desc only' })
    expect(card?.description).toBe('desc only')
    expect(card?.name).toBe('')
  })

  it('filters alternate_greetings down to strings', () => {
    const card = normalizeCardJson(v2Card)
    expect(card?.alternate_greetings).toEqual(['Hi!'])
  })

  it('filters tags down to strings', () => {
    const card = normalizeCardJson(v2Card)
    expect(card?.tags).toEqual(['mage'])
  })
})

describe('parseCharacterCard (PNG)', () => {
  it('parses a V2 card from a chara tEXt chunk', async () => {
    const png = buildPng([
      makeChunk(
        'tEXt',
        textChunkData('chara', base64(JSON.stringify(v2Card)))
      ),
      makeChunk('IEND', new Uint8Array(0)),
    ])
    const card = await parseCharacterCard(png, 'aria.png')
    expect(card.name).toBe('Aria')
    expect(card.system_prompt).toBe('You are Aria.')
  })

  it('prefers ccv3 over chara when both present', async () => {
    const v3Card = {
      spec: 'chara_card_v3',
      spec_version: '3.0',
      data: { name: 'Vex Three', description: 'v3 version' },
    }
    const png = buildPng([
      makeChunk(
        'tEXt',
        textChunkData('chara', base64(JSON.stringify(v2Card)))
      ),
      makeChunk(
        'tEXt',
        textChunkData('ccv3', base64(JSON.stringify(v3Card)))
      ),
      makeChunk('IEND', new Uint8Array(0)),
    ])
    const card = await parseCharacterCard(png, 'duo.png')
    expect(card.name).toBe('Vex Three')
  })

  it('falls back to chara when only chara exists even after other chunks', async () => {
    const png = buildPng([
      makeChunk(
        'tEXt',
        textChunkData('Comment', 'not a card')
      ),
      makeChunk(
        'tEXt',
        textChunkData('chara', base64(JSON.stringify({ name: 'Solo' })))
      ),
      makeChunk('IEND', new Uint8Array(0)),
    ])
    const card = await parseCharacterCard(png, 'solo.png')
    expect(card.name).toBe('Solo')
  })

  it('throws with the filename when the PNG has no character chunks', async () => {
    const png = buildPng([
      makeChunk('IEND', new Uint8Array(0)),
    ])
    await expect(parseCharacterCard(png, 'plain-image.png')).rejects.toThrow(
      /plain-image\.png/
    )
  })

  it('throws when the embedded base64 is not valid JSON', async () => {
    const badBase64 = Buffer.from('not json at all', 'utf-8').toString(
      'base64'
    )
    const png = buildPng([
      makeChunk('tEXt', textChunkData('chara', badBase64)),
      makeChunk('IEND', new Uint8Array(0)),
    ])
    await expect(parseCharacterCard(png, 'bad.png')).rejects.toThrow()
  })

  it('throws when the embedded JSON is not a recognizable card', async () => {
    const png = buildPng([
      makeChunk(
        'tEXt',
        textChunkData('chara', base64(JSON.stringify({ foo: 'bar' })))
      ),
      makeChunk('IEND', new Uint8Array(0)),
    ])
    await expect(
      parseCharacterCard(png, 'junk.png')
    ).rejects.toThrow(/does not contain a recognizable character card/)
  })
})

describe('parseCharacterCard (raw JSON)', () => {
  it('parses a flat V1 card from plain JSON bytes', async () => {
    const json = JSON.stringify({ name: 'Plain', first_mes: 'yo' })
    const buf = new TextEncoder().encode(json) as unknown as ArrayBuffer
    const card = await parseCharacterCard(buf, 'card.json')
    expect(card.name).toBe('Plain')
    expect(card.first_mes).toBe('yo')
  })

  it('throws on invalid JSON text', async () => {
    const buf = new TextEncoder().encode('{nope') as unknown as ArrayBuffer
    await expect(parseCharacterCard(buf, 'broken.json')).rejects.toThrow()
  })
})

describe('cardToCharacter', () => {
  it('maps a full parsed card to a Character', () => {
    const before = Date.now() / 1000 - 5
    const character = cardToCharacter(normalizeCardJson(v2Card)!)
    const after = Date.now() / 1000 + 5

    expect(typeof character.id).toBe('string')
    expect(character.id.length).toBeGreaterThan(0)
    expect(character.name).toBe('Aria')
    expect(character.created_at).toBeGreaterThanOrEqual(before)
    expect(character.created_at).toBeLessThanOrEqual(after)
    expect(character.description).toBe('A wandering mage')
    expect(character.instructions).toBe('You are Aria.')
    expect(character.system_prompt).toBe('You are Aria.')
    expect(character.personality).toBe('curious')
    expect(character.scenario).toBe('a tavern at dusk')
    expect(character.first_mes).toBe('Hello there!')
    expect(character.alternate_greetings).toEqual(['Hi!'])
    expect(character.mes_example).toBe('<START>')
    expect(character.creator).toBe('someone')
    expect(character.parameters).toEqual({})
  })

  it('falls back to Unnamed character when the card has no name', () => {
    const character = cardToCharacter(normalizeCardJson({ description: 'd' })!)
    expect(character.name).toBe('Unnamed character')
  })

  it('omits alternate_greetings and tags when empty', () => {
    const character = cardToCharacter(
      normalizeCardJson({ name: 'Minimal', description: 'd' })!
    )
    expect(character.alternate_greetings).toBeUndefined()
    expect(character.tags).toBeUndefined()
  })
})

describe('migrateAssistantToCharacter', () => {
  it('renames the built-in jan assistant to Assistant', () => {
    const assistant = {
      id: 'jan',
      name: 'Jan',
      created_at: 123,
      instructions: 'Be helpful.',
      parameters: {},
    }
    const character = migrateAssistantToCharacter(assistant)
    expect(character.name).toBe('Assistant')
    expect(character.parameters).toEqual({})
  })

  it('folds legacy instructions into description and empties instructions', () => {
    const assistant = {
      id: 'jan',
      name: 'Jan',
      created_at: 123,
      description: 'A desktop assistant.',
      instructions: 'Be helpful.',
      parameters: {},
    }
    const character = migrateAssistantToCharacter(assistant)
    expect(character.description).toBe('A desktop assistant.\n\nBe helpful.')
    expect(character.instructions).toBe('')
  })

  it('strips dead assistant-only fields but keeps card fields on re-runs', () => {
    const assistant = {
      id: 'custom-id',
      name: 'My Helper',
      created_at: 456,
      instructions: '',
      object: 'assistant',
      tools: [{ type: 'retrieval', enabled: false }],
      file_ids: ['f1'],
      personality: 'Cheerful',
      scenario: 'A tavern',
      parameters: {},
    } as unknown as Parameters<typeof migrateAssistantToCharacter>[0]
    const character = migrateAssistantToCharacter(assistant)
    expect(character.id).toBe('custom-id')
    expect(character.name).toBe('My Helper')
    expect((character as Record<string, unknown>).object).toBeUndefined()
    expect((character as Record<string, unknown>).tools).toBeUndefined()
    expect((character as Record<string, unknown>).file_ids).toBeUndefined()
    expect(character.personality).toBe('Cheerful')
    expect(character.scenario).toBe('A tavern')
  })

  it('preserves the name of non-built-in assistants', () => {
    const assistant = {
      id: 'custom-id',
      name: 'My Helper',
      created_at: 456,
      instructions: 'Do things.',
      parameters: {},
    }
    const character = migrateAssistantToCharacter(assistant)
    expect(character.id).toBe('custom-id')
    expect(character.name).toBe('My Helper')
    expect(character.first_mes).toBeUndefined()
    expect(character.tags).toBeUndefined()
  })
})

describe('inferCharacterKind', () => {
  it('reads a stored kind rather than re-deriving it', () => {
    // A card-shaped character the user deliberately switched to assistant
    // must stay one, or the pills come back on every load.
    expect(
      inferCharacterKind({ kind: 'assistant', first_mes: 'Hello!' })
    ).toBe('assistant')
  })

  it('treats any card field as evidence of a roleplay character', () => {
    expect(inferCharacterKind({ scenario: 'A tavern at dusk.' })).toBe(
      'roleplay'
    )
    expect(inferCharacterKind({ alternate_greetings: ['Hi'] })).toBe(
      'roleplay'
    )
  })

  it('treats a description-only character as an assistant', () => {
    // This is the shape a hand-written coding helper has, and the shape the
    // built-in Jan assistant has -- neither should get world info.
    expect(
      inferCharacterKind({ id: 'jan', description: 'A long system prompt.' })
    ).toBe('assistant')
    expect(inferCharacterKind({ description: 'Reviews Rust code.' })).toBe(
      'assistant'
    )
  })
})

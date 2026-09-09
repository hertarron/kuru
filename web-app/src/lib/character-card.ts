import { generateId } from 'ai'
import { lorebookFromCard, type Lorebook } from './lorebook'

/**
 * Community character card import. Parses the two distribution formats:
 *  - JSON files (V1 flat fields, or the V2/V3 `{ spec, spec_version, data }`
 *    envelope)
 *  - PNG images with the card JSON embedded (base64-encoded) in a `tEXt`
 *    metadata chunk keyed `chara` (V2) or `ccv3` (V3)
 *
 * Everything normalizes into our Character type. Unknown fields from the
 * source card are dropped only here at the boundary; the store never
 * re-exports, so no ecosystem data can be destroyed by editing.
 */

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

/** Max bytes we will inflate for a zTXt chunk (zip-bomb guard). */
const MAX_ZTXT_SIZE = 2_000_000

export type ParsedCard = {
  name?: string
  description?: string
  personality?: string
  scenario?: string
  first_mes?: string
  alternate_greetings?: string[]
  mes_example?: string
  system_prompt?: string
  post_history_instructions?: string
  creator_notes?: string
  creator?: string
  character_version?: string
  tags?: string[]
  /** The card's own world info, when it ships one. */
  character_book?: Lorebook
}

export function isPngBuffer(buffer: ArrayBuffer): boolean {
  const bytes = new Uint8Array(buffer.slice(0, 8))
  return PNG_SIGNATURE.every((b, i) => bytes[i] === b)
}

function decodeBase64(value: string): string {
  // Tolerate base64 with embedded whitespace/newlines.
  const cleaned = value.replace(/\s+/g, '')
  const binary = atob(cleaned)
  const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0))
  return new TextDecoder().decode(bytes)
}

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as unknown as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream('deflate'))
  const buf = await new Response(stream).arrayBuffer()
  if (buf.byteLength > MAX_ZTXT_SIZE) {
    throw new Error('Compressed card metadata exceeds size limit')
  }
  return new Uint8Array(buf)
}

type TextChunk = { keyword: string; text: string }

/** Walk PNG chunks and collect tEXt/zTXt entries. */
async function extractTextChunks(
  buffer: ArrayBuffer
): Promise<TextChunk[]> {
  const view = new DataView(buffer)
  const bytes = new Uint8Array(buffer)
  const chunks: TextChunk[] = []
  let offset = 8 // skip PNG signature

  while (offset + 12 <= buffer.byteLength) {
    const length = view.getUint32(offset)
    const type = String.fromCharCode(
      bytes[offset + 4],
      bytes[offset + 5],
      bytes[offset + 6],
      bytes[offset + 7]
    )
    const dataStart = offset + 8
    if (dataStart + length > buffer.byteLength) break

    if (type === 'tEXt') {
      const data = bytes.subarray(dataStart, dataStart + length)
      const separator = data.indexOf(0)
      if (separator > 0) {
        chunks.push({
          keyword: new TextDecoder().decode(data.subarray(0, separator)),
          text: new TextDecoder().decode(data.subarray(separator + 1)),
        })
      }
    } else if (type === 'zTXt') {
      const data = bytes.subarray(dataStart, dataStart + length)
      const separator = data.indexOf(0)
      if (separator > 0 && separator + 1 < data.length) {
        // keyword \0 compression-method(1 byte) deflate-data
        try {
          const inflated = await inflate(data.subarray(separator + 2))
          chunks.push({
            keyword: new TextDecoder().decode(data.subarray(0, separator)),
            text: new TextDecoder().decode(inflated),
          })
        } catch {
          // Skip undecodable compressed chunks rather than failing import.
        }
      }
    }

    offset = dataStart + length + 4 // skip CRC
    if (type === 'IEND') break
  }
  return chunks
}

/** Accept V1 flat objects and V2/V3 envelopes; returns null when unusable. */
export function normalizeCardJson(raw: unknown): ParsedCard | null {
  if (!raw || typeof raw !== 'object') return null
  const obj = raw as Record<string, unknown>
  const data =
    (obj.spec === 'chara_card_v2' || obj.spec === 'chara_card_v3') &&
    obj.data &&
    typeof obj.data === 'object'
      ? (obj.data as Record<string, unknown>)
      : obj

  const str = (key: string): string => {
    const v = data[key]
    return typeof v === 'string' ? v : ''
  }

  const name = str('name').trim()
  // A card without a name or any prompt-bearing field is not a character.
  const hasContent = [
    'description',
    'personality',
    'scenario',
    'first_mes',
    'mes_example',
  ].some((k) => str(k).length > 0)
  if (!name && !hasContent) return null

  const greetings = Array.isArray(data.alternate_greetings)
    ? (data.alternate_greetings as unknown[]).filter(
        (g): g is string => typeof g === 'string'
      )
    : []
  const tags = Array.isArray(data.tags)
    ? (data.tags as unknown[]).filter(
        (t): t is string => typeof t === 'string'
      )
    : []

  const book = lorebookFromCard(raw, {
    id: generateId(),
    name: name ? `${name}'s world info` : 'World info',
  })

  return {
    character_book: book ?? undefined,
    name,
    description: str('description'),
    personality: str('personality'),
    scenario: str('scenario'),
    first_mes: str('first_mes'),
    alternate_greetings: greetings,
    mes_example: str('mes_example'),
    system_prompt: str('system_prompt'),
    post_history_instructions: str('post_history_instructions'),
    creator_notes:
      typeof data.creator_notes === 'string' ? data.creator_notes : '',
    creator: str('creator'),
    character_version: str('character_version'),
    tags,
  }
}

/**
 * Parse a character card from raw file bytes. Supports PNG (with embedded
 * metadata) and plain JSON. Throws with a user-actionable message when the
 * file carries no usable card data.
 */
export async function parseCharacterCard(
  buffer: ArrayBuffer,
  filename: string
): Promise<ParsedCard> {
  let rawJson: unknown

  if (isPngBuffer(buffer)) {
    const chunks = await extractTextChunks(buffer)
    // ccv3 wins over chara per the V3 spec recommendation.
    const chunk =
      chunks.find((c) => c.keyword === 'ccv3') ??
      chunks.find((c) => c.keyword === 'chara')
    if (!chunk?.text) {
      throw new Error(
        `${filename} has no character data -- the image was likely re-compressed. Try the original PNG or a .json export.`
      )
    }
    rawJson = JSON.parse(decodeBase64(chunk.text))
  } else {
    rawJson = JSON.parse(new TextDecoder().decode(buffer))
  }

  const card = normalizeCardJson(rawJson)
  if (!card) {
    throw new Error(`${filename} does not contain a recognizable character card`)
  }
  return card
}

/** Turn a parsed card into a storable Character with sane defaults. */
export function cardToCharacter(card: ParsedCard): Character {
  return {
    id: generateId(),
    name: card.name || 'Unnamed character',
    created_at: Date.now() / 1000,
    description: card.description ?? '',
    instructions: card.system_prompt ?? '',
    personality: card.personality,
    scenario: card.scenario,
    first_mes: card.first_mes,
    alternate_greetings: card.alternate_greetings?.length
      ? card.alternate_greetings
      : undefined,
    mes_example: card.mes_example,
    system_prompt: card.system_prompt,
    post_history_instructions: card.post_history_instructions || undefined,
    tags: card.tags?.length ? card.tags : undefined,
    creator_notes: card.creator_notes || undefined,
    creator: card.creator || undefined,
    character_version: card.character_version || undefined,
    parameters: {},
  }
}

/**
 * Downscale an uploaded image to a compact square data URL suitable for
 * inline storage in character.json (avatars travel inside the JSON).
 */
export async function fileToAvatarDataUrl(
  file: File,
  size = 256
): Promise<string> {
  const bitmap = await createImageBitmap(file)
  const scale = Math.min(1, size / Math.max(bitmap.width, bitmap.height))
  const w = Math.max(1, Math.round(bitmap.width * scale))
  const h = Math.max(1, Math.round(bitmap.height * scale))
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas unavailable')
  ctx.drawImage(bitmap, 0, 0, w, h)
  bitmap.close()
  return canvas.toDataURL('image/png')
}

/**
 * Legacy migration: map an old Assistant record onto Character. The default
 * Jan assistant is rebranded as the neutral built-in "Assistant" character;
 * everything else keeps its identity. Legacy `instructions` are folded into
 * `description` and dead assistant-only fields (object/tools/file_ids) are
 * dropped, so the description box is the hard truth of what reaches the model.
 */
/**
 * What a character predating the `kind` field must have been. Card fields are
 * the tell: a greeting, a scenario, a personality or example dialogue only
 * ever came from a character card. The built-in Jan assistant is the one
 * hardcoded exception -- it has a long description and nothing else, which is
 * indistinguishable from a hand-written coding assistant.
 *
 * Only consulted when `kind` is absent. Once the user sets it, the stored
 * value wins forever.
 */
export function inferCharacterKind(
  c: Partial<Character> & { id?: string }
): CharacterKind {
  if (c.kind) return c.kind
  if (c.id === 'jan') return 'assistant'
  const hasCardFields = !!(
    c.first_mes?.trim() ||
    c.scenario?.trim() ||
    c.personality?.trim() ||
    c.mes_example?.trim() ||
    c.alternate_greetings?.length
  )
  return hasCardFields ? 'roleplay' : 'assistant'
}

/**
 * Whether roleplay machinery applies to this character. The one gate for
 * greetings, world info, personas and the pills that control them, so a
 * coding assistant can't be given a lorebook it shows no way to remove.
 *
 * No character at all means no roleplay. That case is a thread carrying the
 * `'model-only'` sentinel or an empty `assistants` array -- legacy chats, not
 * anything reachable from the character picker -- and "just the model" is
 * nearer an assistant than a character. Otherwise those threads would silently
 * receive always-active lorebooks and the persona.
 */
export function isRoleplayCharacter(c: Character | undefined): boolean {
  if (!c) return false
  return inferCharacterKind(c) === 'roleplay'
}

export function migrateAssistantToCharacter(a: Assistant): Character {
  const isBuiltIn = a.id === 'jan'
  // Strip keys that only existed on the old assistant record; keep everything
  // else (card fields on already-migrated characters must survive re-runs).
  const { instructions, object, tools, file_ids, ...rest } =
    a as Assistant & Record<string, unknown>
  void object
  void tools
  void file_ids
  const description = [a.description?.trim(), instructions?.trim()]
    .filter(Boolean)
    .join('\n\n')
  return {
    ...(rest as Character),
    name: isBuiltIn ? 'Assistant' : a.name,
    description,
    // Card fields are preserved when present (already-migrated characters
    // carry them); genuinely legacy assistants don't have these keys, so
    // they stay undefined exactly as before. Wiping them here would drop
    // greetings/tags from the in-memory character on every app load.
    first_mes: (a as Partial<Character>).first_mes,
    alternate_greetings: (a as Partial<Character>).alternate_greetings,
    tags: (a as Partial<Character>).tags,
    kind: inferCharacterKind(a as Partial<Character> & { id?: string }),
    instructions: '',
  }
}

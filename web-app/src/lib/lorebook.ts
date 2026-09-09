/**
 * Lorebook (world info) format layer.
 *
 * Two dialects are in circulation and both have to import cleanly:
 *
 *  - **Card-embedded** (`data.character_book` in a V2 card, and what Chub's
 *    lorebook API returns as `definition.embedded_lorebook`): `entries` is an
 *    ARRAY, names are snake_case (`keys`, `secondary_keys`, `insertion_order`),
 *    and the odds and ends live in a per-entry `extensions` bag.
 *  - **SillyTavern world export** (the `.json` files people trade outside
 *    Chub): `entries` is an OBJECT keyed by uid, names are ST's internal
 *    camelCase (`key`, `keysecondary`, `order`, `disable`), and the same odds
 *    and ends are hoisted to the top level of the entry.
 *
 * ST itself keeps a translation table between the two
 * (`originalWIDataKeyMap` in `public/scripts/world-info.js`); this is the same
 * mapping in the direction we need.
 *
 * Everything we don't model yet is kept verbatim in `extra` so a book can be
 * re-exported without losing what it came in with.
 */

/** ST's `world_info_logic`. */
export const LorebookLogic = {
  AND_ANY: 0,
  NOT_ALL: 1,
  NOT_ANY: 2,
  AND_ALL: 3,
} as const

/** ST's `world_info_position`. */
export const LorebookPosition = {
  beforeChar: 0,
  afterChar: 1,
  ANTop: 2,
  ANBottom: 3,
  atDepth: 4,
  EMTop: 5,
  EMBottom: 6,
  outlet: 7,
} as const

export interface LorebookEntry {
  id: string
  /** Trigger keys. Plain text, or `/pattern/flags` for regex. */
  keys: string[]
  secondaryKeys: string[]
  /** One of `LorebookLogic`; only meaningful when secondaryKeys is non-empty. */
  selectiveLogic: number
  content: string
  /** Author's label for the entry. Never sent to the model. */
  comment: string
  enabled: boolean
  /** Always active, no key match required (ST's `constant`). */
  constant: boolean
  /** One of `LorebookPosition`. */
  position: number
  /** Messages from the end, when position is `atDepth`. */
  depth: number
  /** 0 system, 1 user, 2 assistant, when position is `atDepth`. */
  role: number | null
  /** Rank among entries landing in the same slot. Higher wins. */
  order: number
  /** 0-100. */
  probability: number
  caseSensitive: boolean
  /** Fields carried through untouched. */
  extra: Record<string, unknown>
}

export interface Lorebook {
  id: string
  name: string
  description: string
  entries: LorebookEntry[]
  /**
   * Carried through import and export untouched. The scanner does not read
   * them: it runs one pass over every live book, so a per-book scan depth or
   * budget would have to compete with the others rather than simply override
   * the global. Kept so a book we re-export is the book that came in.
   */
  scanDepth: number | null
  tokenBudget: number | null
  /** Entries in this book may trigger each other. Live, unlike the two above. */
  recursiveScanning: boolean
  /**
   * Active in every chat without being attached to a character. Ours, not part
   * of either import dialect, so imports always start false.
   */
  global: boolean
  /** Where this came from, e.g. `chub:lorebooks/creator/slug`. */
  source?: string
}

const num = (v: unknown, fallback: number): number =>
  typeof v === 'number' && Number.isFinite(v) ? v : fallback

const num2 = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null

const bool = (v: unknown, fallback: boolean): boolean =>
  typeof v === 'boolean' ? v : fallback

const strings = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((k): k is string => typeof k === 'string' && k.length > 0) : []

const str = (v: unknown, fallback = ''): string =>
  typeof v === 'string' ? v : fallback

/**
 * Keys we read explicitly; anything else on the entry lands in `extra`.
 * Includes our own field names, since a book we saved is re-read through here:
 * without them every reload copied the entry's own fields into `extra` and
 * nested the previous `extra` inside the new one, burying the ST flags we do
 * read a level deeper each time.
 */
const CONSUMED = new Set([
  'id', 'uid', 'keys', 'key', 'secondary_keys', 'keysecondary', 'secondaryKeys',
  'selectiveLogic', 'content', 'comment', 'name', 'enabled', 'disable',
  'constant', 'position', 'insertion_order', 'order', 'probability',
  'case_sensitive', 'caseSensitive', 'depth', 'role', 'extra', 'extensions',
  'selective',
])

/** Flattens the nested `extra` bags older saves accumulated. */
function unnestExtra(raw: unknown, into: Record<string, unknown>): void {
  if (!raw || typeof raw !== 'object') return
  const bag = raw as Record<string, unknown>
  unnestExtra(bag.extra, into)
  for (const [k, v] of Object.entries(bag)) {
    // Stale copies of the entry's own fields: the nesting is the bug, and the
    // live values are read from the entry itself.
    if (!CONSUMED.has(k)) into[k] = v
  }
}

function normalizeEntry(raw: Record<string, unknown>, index: number): LorebookEntry {
  const ext = (raw.extensions ?? {}) as Record<string, unknown>

  // Card dialect keeps position/depth/role in extensions; ST's export hoists
  // them onto the entry. Prefer whichever is actually present.
  const pick = (cardKey: string, stKey: string, fallback: unknown): unknown =>
    ext[cardKey] !== undefined ? ext[cardKey] : raw[stKey] !== undefined ? raw[stKey] : fallback

  const extra: Record<string, unknown> = {}
  // Deepest first, so a value written more recently wins.
  unnestExtra(raw.extra, extra)
  for (const [k, v] of Object.entries(raw)) {
    if (!CONSUMED.has(k)) extra[k] = v
  }
  for (const [k, v] of Object.entries(ext)) {
    if (!['position', 'depth', 'role', 'probability', 'case_sensitive'].includes(k)) {
      extra[k] = v
    }
  }

  const keys = strings(raw.keys ?? raw.key)
  const constant = bool(raw.constant, false) || keys.length === 0

  return {
    id: String(raw.id ?? raw.uid ?? index),
    keys,
    secondaryKeys: strings(
      raw.secondary_keys ?? raw.keysecondary ?? raw.secondaryKeys
    ),
    selectiveLogic: num(raw.selectiveLogic, LorebookLogic.AND_ANY),
    content: str(raw.content),
    comment: str(raw.comment) || str(raw.name),
    // ST stores the inverse (`disable`); the card dialect stores `enabled`.
    enabled: raw.disable !== undefined ? !raw.disable : bool(raw.enabled, true),
    constant,
    position: num(pick('position', 'position', LorebookPosition.beforeChar), LorebookPosition.beforeChar),
    depth: num(pick('depth', 'depth', 4), 4),
    role: typeof pick('role', 'role', null) === 'number' ? (pick('role', 'role', null) as number) : null,
    order: num(raw.insertion_order ?? raw.order, 100),
    probability: num(ext.probability ?? raw.probability, 100),
    caseSensitive: bool(ext.case_sensitive ?? raw.caseSensitive, false),
    extra,
  }
}

/**
 * Accepts either dialect, plus a bare array of entries, and returns null when
 * the payload isn't a lorebook at all.
 *
 * `allowEmpty` is for reloading books we wrote ourselves: an import with no
 * usable entries is a failed import, but a book the user just created and
 * hasn't filled in yet must still come back from disk.
 */
export function normalizeLorebook(
  raw: unknown,
  opts: {
    id: string
    name?: string
    description?: string
    source?: string
    allowEmpty?: boolean
  }
): Lorebook | null {
  if (!raw || typeof raw !== 'object') return null
  const book = raw as Record<string, unknown>

  const rawEntries = book.entries
  let list: Record<string, unknown>[]
  if (Array.isArray(rawEntries)) {
    list = rawEntries.filter((e): e is Record<string, unknown> => !!e && typeof e === 'object')
  } else if (rawEntries && typeof rawEntries === 'object') {
    // ST export: uid-keyed object. Object key order is insertion order for
    // string keys, but these are numeric-like, so sort by uid to be safe.
    list = Object.entries(rawEntries as Record<string, unknown>)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([uid, e]) => ({ uid, ...(e as Record<string, unknown>) }))
  } else if (rawEntries === undefined && opts.allowEmpty) {
    list = []
  } else {
    return null
  }

  const entries = list
    .map((e, i) => normalizeEntry(e, i))
    // A contentless entry in an import is padding; in one of our own books it
    // is an entry the user has started and not finished.
    .filter((e) => opts.allowEmpty || e.content.trim().length > 0)

  if (entries.length === 0 && !opts.allowEmpty) return null

  return {
    id: opts.id,
    name: str(book.name, opts.name ?? 'Untitled lorebook'),
    description: str(book.description) || opts.description || '',
    entries,
    // Both spellings: the card dialect writes snake_case, and a book of ours
    // is re-read through here with the names we saved it under.
    scanDepth: num2(book.scan_depth ?? book.scanDepth),
    tokenBudget: num2(book.token_budget ?? book.tokenBudget),
    recursiveScanning: bool(
      book.recursive_scanning ?? book.recursiveScanning,
      false
    ),
    global: bool(book.global, false),
    source: opts.source ?? (typeof book.source === 'string' ? book.source : undefined),
  }
}

/**
 * Pulls the book a V2 character card carries in `data.character_book`.
 * Returns null for the many cards that ship without one.
 */
export function lorebookFromCard(
  cardJson: unknown,
  opts: { id: string; name?: string; source?: string }
): Lorebook | null {
  if (!cardJson || typeof cardJson !== 'object') return null
  const root = cardJson as Record<string, unknown>
  const data = (root.data ?? root) as Record<string, unknown>
  const book = data.character_book
  if (!book) return null
  return normalizeLorebook(book, opts)
}

/**
 * Positions kuru actually places. ST's author's note (2/3) and outlet (7) need
 * concepts this app doesn't have, so imported entries carrying them are
 * redirected to the nearest thing that exists — and the entry editor says so
 * rather than letting the difference bite silently at generation time.
 */
export const SUPPORTED_POSITIONS: number[] = [
  LorebookPosition.beforeChar,
  LorebookPosition.afterChar,
  LorebookPosition.atDepth,
  LorebookPosition.EMTop,
  LorebookPosition.EMBottom,
]

export interface EffectivePosition {
  position: number
  /** Set only when the stored position isn't one we place. */
  note?: string
}

export function effectivePosition(entry: LorebookEntry): EffectivePosition {
  switch (entry.position) {
    case LorebookPosition.ANTop:
    case LorebookPosition.ANBottom:
      return {
        position: LorebookPosition.atDepth,
        note: 'Author’s note has no equivalent here; this entry is inserted at depth instead.',
      }
    case LorebookPosition.outlet:
      return {
        position: LorebookPosition.beforeChar,
        note: 'Outlets have no equivalent here; this entry is inserted before the character definition instead.',
      }
    default:
      return SUPPORTED_POSITIONS.includes(entry.position)
        ? { position: entry.position }
        : {
            position: LorebookPosition.beforeChar,
            note: 'Unknown position; this entry is inserted before the character definition instead.',
          }
  }
}

/**
 * Splits the keys input. Commas separate keys, except inside a `/regex/flags`
 * literal where a comma is part of the pattern (`/a{1,3}/i`).
 */
export function parseKeys(input: string): string[] {
  const keys: string[] = []
  let current = ''
  let inRegex = false
  let escaped = false

  for (const ch of input) {
    if (escaped) {
      current += ch
      escaped = false
      continue
    }
    if (ch === '\\') {
      current += ch
      escaped = true
      continue
    }
    if (ch === '/') {
      // A slash opens a literal only at the start of a key.
      if (!inRegex && current.trim() === '') inRegex = true
      else if (inRegex) inRegex = false
      current += ch
      continue
    }
    if (ch === ',' && !inRegex) {
      if (current.trim()) keys.push(current.trim())
      current = ''
      continue
    }
    current += ch
  }
  if (current.trim()) keys.push(current.trim())
  return keys
}

export const formatKeys = (keys: string[]): string => keys.join(', ')

/**
 * Serialize a book back to the card-embedded dialect, the one both Chub and
 * SillyTavern accept on import. Fields we never modelled ride along from
 * `extra`, so a book that came in from elsewhere goes back out intact.
 *
 * The V2 card spec puts a string `position` on the entry while ST reads a
 * numeric one from `extensions`; both are written, since the two importers
 * disagree about which to trust.
 */
export function lorebookToCardBook(book: Lorebook): Record<string, unknown> {
  return {
    name: book.name,
    description: book.description,
    scan_depth: book.scanDepth ?? undefined,
    token_budget: book.tokenBudget ?? undefined,
    recursive_scanning: book.recursiveScanning,
    extensions: {},
    entries: book.entries.map((entry, index) => ({
      ...entry.extra,
      id: index,
      keys: entry.keys,
      secondary_keys: entry.secondaryKeys,
      comment: entry.comment,
      content: entry.content,
      constant: entry.keys.length === 0,
      selective: entry.secondaryKeys.length > 0,
      selectiveLogic: entry.selectiveLogic,
      insertion_order: entry.order,
      enabled: entry.enabled,
      position:
        entry.position === LorebookPosition.beforeChar
          ? 'before_char'
          : 'after_char',
      extensions: {
        position: entry.position,
        depth: entry.depth,
        role: entry.role,
        probability: entry.probability,
        case_sensitive: entry.caseSensitive,
      },
    })),
  }
}

import {
  parseCharacterCard,
  cardToCharacter,
  type ParsedCard,
} from './character-card'
import seedTagFreq from './chub-tags.json'
import tagAliases from './chub-tag-aliases.json'
import { normalizeLorebook, type Lorebook } from './lorebook'

/**
 * Chub.ai (CharacterHub) client — the largest public repository of
 * SillyTavern-format character cards. Read-only search needs no auth;
 * a user token unlocks NSFW/private listings.
 */

const CHUB_SEARCH_URL = 'https://api.chub.ai/search'
const CHUB_API_URL = 'https://api.chub.ai/api'
const CHUB_CARD_URL = 'https://avatars.charhub.io/avatars'

export interface ChubCharacterResult {
  id: number
  name: string
  fullPath: string
  tagline?: string
  description?: string
  avatar_url?: string
  starCount?: number
  nMessages?: number
  nTokens?: number
  rating?: number
  topics?: string[]
}

export interface ChubSearchData {
  count: number
  nodes: ChubCharacterResult[]
}

/**
 * Sort fields the Chub search API accepts (verified against api.chub.ai):
 * default | trending | star_count | rating | last_activity_at | random | name
 */
/**
 * Search results, keyed by the exact request.
 *
 * Switching between the Hub tabs remounts each page, and every mount reissues
 * the default listing — the same 24 cards, from a public API, several times a
 * minute. Entries hold the promise rather than the result so two mounts racing
 * each other share one request, and a rejection is evicted so a retry is a
 * real retry. Short-lived on purpose: this is about tab switching, not about
 * showing yesterday's listing.
 */
const SEARCH_TTL_MS = 5 * 60 * 1000
const searchCache = new Map<string, { at: number; promise: Promise<unknown> }>()

function cachedSearch<T>(key: string, run: () => Promise<T>): Promise<T> {
  const hit = searchCache.get(key)
  if (hit && Date.now() - hit.at < SEARCH_TTL_MS) return hit.promise as Promise<T>
  const promise = run()
  promise.catch(() => {
    if (searchCache.get(key)?.promise === promise) searchCache.delete(key)
  })
  searchCache.set(key, { at: Date.now(), promise })
  return promise
}

/** Drops cached listings, so a deliberate refresh actually hits Chub. */
export function clearChubSearchCache(): void {
  searchCache.clear()
}

export async function searchChubCharacters(
  query: string,
  opts: {
    first?: number
    page?: number
    token?: string
    sort?: string
    /** Tag names that results MUST have (ANDed by Chub). */
    tags?: string[]
    /** Tag names that results must NOT have. */
    excludedTags?: string[]
  } = {}
): Promise<ChubSearchData> {
  const params = new URLSearchParams({
    namespace: 'characters',
    first: String(opts.first ?? 24),
    page: String(opts.page ?? 1),
    nsfw: 'false',
    sort: opts.sort ?? 'default',
  })
  // Empty query = the default listing (top by stars), like the hub's catalog.
  if (query.trim()) params.set('search', query.trim())
  if (opts.tags?.length) params.set('tags', opts.tags.join(','))
  if (opts.excludedTags?.length)
    params.set('excluded_tags', opts.excludedTags.join(','))
  return cachedSearch(`characters:${params.toString()}`, async () => {
    const response = await fetch(`${CHUB_SEARCH_URL}?${params.toString()}`, {
      headers: opts.token ? { Authorization: `Bearer ${opts.token}` } : undefined,
    })
    if (!response.ok) {
      throw new Error(
        `Chub search failed: ${response.status} ${response.statusText}`
      )
    }
    const json: { data?: ChubSearchData } = await response.json()
    if (!json.data) throw new Error('Chub search returned no data')
    return json.data
  })
}

export function chubCardUrl(fullPath: string): string {
  return `${CHUB_CARD_URL}/${fullPath}/chara_card_v2.png`
}

/**
 * Tag autocomplete has two layers:
 * 1. A bundled seed list (chub-tags.json) — the ~2.2k most popular tags,
 *    harvested once from search results (2026-08-22). Counts are sampled
 *    popularity, used only for ranking suggestions.
 * 2. Live probing (probeChubTagCount) — one cheap search call returns the
 *    exact current count for ANY typed string, so users see that "woman"
 *    only matches 55 cards before committing it instead of "female".
 * Chub has no public autocomplete endpoint; matching is exact after
 * lowercasing (no synonyms, no fuzz).
 */
const SEED_LIST = seedTagFreq as [string, number][]
const SEED_BY_KEY = new Map<string, [string, number]>(
  SEED_LIST.map(([tag, count]) => [tag.toLowerCase(), [tag, count]])
)
const TAG_ALIASES = tagAliases as Record<string, string[]>

/** Plain DP Levenshtein — the seed list is small enough that this is free. */
function editDistance(a: string, b: string): number {
  if (Math.abs(a.length - b.length) > 2) return 3
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    const cur = [i]
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(
        prev[j] + 1,
        cur[j - 1] + 1,
        prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      )
    }
    prev = cur
  }
  return prev[b.length]
}

/**
 * Suggestions for the tag filter panel. Empty query → top tags by sampled
 * volume. Otherwise: substring matches, curated alias expansions (semantic
 * near-misses like woman→female) and typo fuzz (edit distance ≤2), all
 * merged and ranked by card volume. Only tags present in the seed list are
 * suggested; anything else can still be committed verbatim via Enter.
 */
export function getSuggestedTags(
  query: string,
  exclude: string[],
  limit = 10
): [string, number][] {
  const excluded = new Set(exclude.map((e) => e.toLowerCase()))
  // Lower tier wins; volume only ranks within the same tier, so an exact
  // or substring match ("dog girl") never sinks below high-volume alias
  // expansions (Female).
  const candidates = new Map<string, [string, number, number]>()
  const addTag = (tag: string, tier: number) => {
    const entry = SEED_BY_KEY.get(tag.toLowerCase())
    if (!entry) return
    const key = entry[0].toLowerCase()
    if (excluded.has(key)) return
    const cur = candidates.get(key)
    if (!cur || tier < cur[2]) candidates.set(key, [...entry, tier])
  }

  const q = query.trim().toLowerCase()
  if (!q) {
    return SEED_LIST.filter(([tag]) => !excluded.has(tag.toLowerCase())).slice(
      0,
      limit
    )
  }

  // Substring matches, graded by how tight they are.
  for (const [tag] of SEED_LIST) {
    const tl = tag.toLowerCase()
    if (tl.includes(q)) {
      addTag(tag, tl === q ? 0 : tl.startsWith(q) ? 1 : 2)
    }
  }
  // Alias expansions — keys match exactly, by prefix, or fuzzily so
  // "menn" still triggers the men/male/yaoi cluster.
  for (const [key, targets] of Object.entries(TAG_ALIASES)) {
    const k = key.toLowerCase()
    if (
      k === q ||
      k.includes(q) ||
      q.includes(k) ||
      editDistance(k, q) <= (k.length >= 4 ? 2 : 1)
    ) {
      for (const target of targets) addTag(target, 3)
    }
  }
  // Typo fuzz against the seed list itself.
  const maxDist = q.length >= 4 ? 2 : 1
  for (const [tag] of SEED_LIST) {
    const tl = tag.toLowerCase()
    if (!candidates.has(tl) && editDistance(tl, q) <= maxDist) {
      addTag(tag, 4)
    }
  }

  return [...candidates.values()]
    .sort((a, b) => a[2] - b[2] || b[1] - a[1])
    .slice(0, limit)
    .map(([tag, count]) => [tag, count])
}

const probeCache = new Map<string, Promise<number>>()

/**
 * Exact current count for a tag, via a minimal search call. Namespaced,
 * because the same tag has wildly different volume across catalogs and
 * plenty of character tags match no lorebooks at all.
 */
export function probeChubTagCount(
  tag: string,
  token?: string,
  namespace: 'characters' | 'lorebooks' = 'characters'
): Promise<number> {
  const key = `${namespace}:${tag.toLowerCase()}`
  const cached = probeCache.get(key)
  if (cached) return cached
  const search =
    namespace === 'lorebooks' ? searchChubLorebooks : searchChubCharacters
  const promise = search('', { first: 1, tags: [tag], token })
    .then((d) => d.count)
    .catch((e) => {
      probeCache.delete(key)
      throw e
    })
  probeCache.set(key, promise)
  return promise
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

/**
 * Downloads a Chub card PNG and runs it through the same parser as manual
 * file imports, so nothing about the pipeline is Chub-specific. The portrait
 * doubles as the avatar, matching local PNG imports.
 *
 * Cached by fullPath: previewing a card and then importing it should not
 * fetch the same megabyte twice. The cache holds the parsed card rather than
 * a built Character, so every import still gets its own id.
 */
const cardCache = new Map<
  string,
  Promise<{ card: ParsedCard; avatar?: string }>
>()

function loadChubCard(
  node: Pick<ChubCharacterResult, 'fullPath' | 'name'>
): Promise<{ card: ParsedCard; avatar?: string }> {
  const cached = cardCache.get(node.fullPath)
  if (cached) return cached

  const promise = (async () => {
    const url = chubCardUrl(node.fullPath)
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(
        `Failed to download "${node.name}": ${response.status} ${response.statusText}`
      )
    }
    const buffer = await response.arrayBuffer()
    let card: ParsedCard
    try {
      card = await parseCharacterCard(buffer, `${node.fullPath}.png`)
    } catch (e) {
      // A deleted Chub character still serves its thumbnail, which carries no
      // embedded card -- so "re-compressed" alone would point at the wrong fix.
      if (e instanceof Error && e.message.includes('no character data')) {
        throw new Error(
          `"${node.name}" has no character data -- the character might be deleted from Chub, or the image was re-compressed. Try a .json export if you have one.`
        )
      }
      throw e
    }
    let avatar: string | undefined
    try {
      avatar = await blobToDataUrl(new Blob([buffer], { type: 'image/png' }))
    } catch {
      // Avatar is optional; keep the card usable without it.
    }
    return { card, avatar }
  })()

  // Don't cache failures -- a retry should actually retry.
  promise.catch(() => cardCache.delete(node.fullPath))
  cardCache.set(node.fullPath, promise)
  return promise
}

/**
 * Everything the preview sheet shows, which is also everything import needs.
 */
export async function loadChubCharacter(
  node: Pick<ChubCharacterResult, 'fullPath' | 'name'>
): Promise<{ character: Character; lorebook: Lorebook | null }> {
  const { card, avatar } = await loadChubCard(node)
  const character = cardToCharacter(card)
  if (avatar) character.avatar = avatar
  return { character, lorebook: card.character_book ?? null }
}

export async function importChubCharacter(
  node: Pick<ChubCharacterResult, 'fullPath' | 'name'>
) {
  const { character } = await loadChubCharacter(node)
  return character
}

/**
 * Pulls `creator/slug` out of a chub.ai character URL, or null for anything
 * else. Creator notes are full of these -- "my V2 is over here" -- and they
 * are worth following in place rather than in a browser.
 */
export function chubCharacterPathFromUrl(url: string): string | null {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    return null
  }
  if (!/(^|\.)chub\.ai$|(^|\.)characterhub\.org$/.test(parsed.hostname)) {
    return null
  }
  const parts = parsed.pathname.split('/').filter(Boolean)
  // /characters/<creator>/<slug>
  if (parts.length !== 3 || parts[0] !== 'characters') return null
  return `${parts[1]}/${parts[2]}`
}

/**
 * The search-result shape for a single card. Same node the search endpoint
 * returns, so anything built from a search result works with it unchanged.
 */
export async function fetchChubCharacterNode(
  fullPath: string,
  token?: string
): Promise<ChubCharacterResult> {
  const response = await fetch(
    `${CHUB_API_URL}/characters/${fullPath}?full=true`,
    { headers: token ? { Authorization: `Bearer ${token}` } : undefined }
  )
  if (!response.ok) {
    throw new Error(
      `Chub has no character at "${fullPath}" (${response.status})`
    )
  }
  const json = await response.json()
  const node = json?.node
  if (!node?.fullPath) throw new Error(`Chub returned no card for "${fullPath}"`)
  return node as ChubCharacterResult
}

/* ------------------------------------------------------------------ */
/* Lorebooks                                                           */
/* ------------------------------------------------------------------ */

export interface ChubLorebookResult {
  id: number
  name: string
  fullPath: string
  tagline?: string
  description?: string
  starCount?: number
  nTokens?: number
  rating?: number
  topics?: string[]
  lastActivityAt?: string
}

/**
 * Same endpoint as characters, different namespace. Sorts and tag filters
 * behave identically; the tag vocabulary does not, so lorebook tags are not
 * fed through the character tag suggester.
 */
export async function searchChubLorebooks(
  query: string,
  opts: {
    first?: number
    page?: number
    token?: string
    sort?: string
    tags?: string[]
    excludedTags?: string[]
  } = {}
): Promise<{ count: number; nodes: ChubLorebookResult[] }> {
  const params = new URLSearchParams({
    namespace: 'lorebooks',
    first: String(opts.first ?? 24),
    page: String(opts.page ?? 1),
    nsfw: 'false',
    sort: opts.sort ?? 'default',
  })
  if (query.trim()) params.set('search', query.trim())
  if (opts.tags?.length) params.set('tags', opts.tags.join(','))
  if (opts.excludedTags?.length)
    params.set('excluded_tags', opts.excludedTags.join(','))
  return cachedSearch(`lorebooks:${params.toString()}`, async () => {
    const response = await fetch(`${CHUB_SEARCH_URL}?${params.toString()}`, {
      headers: opts.token ? { Authorization: `Bearer ${opts.token}` } : undefined,
    })
    if (!response.ok) {
      throw new Error(
        `Chub lorebook search failed: ${response.status} ${response.statusText}`
      )
    }
    const json: { data?: { count: number; nodes: ChubLorebookResult[] } } =
      await response.json()
    if (!json.data) throw new Error('Chub lorebook search returned no data')
    return json.data
  })
}

/**
 * Search results never carry entries (`definition` is null), so the book has
 * to be fetched per item. `fullPath` already starts with `lorebooks/`, which
 * the detail endpoint repeats — strip it rather than doubling it up.
 */
export async function fetchChubLorebook(
  node: Pick<
    ChubLorebookResult,
    'fullPath' | 'name' | 'tagline' | 'description'
  >
): Promise<Lorebook> {
  const path = node.fullPath.replace(/^lorebooks\//, '')
  const response = await fetch(
    `${CHUB_API_URL}/lorebooks/${path}?full=true`
  )
  if (!response.ok) {
    throw new Error(
      `Failed to download "${node.name}": ${response.status} ${response.statusText}`
    )
  }
  const json = await response.json()
  const definition = json?.node?.definition
  // A world export carries no description of its own, so the listing's
  // tagline is the only blurb the book will ever have.
  const book = normalizeLorebook(definition?.embedded_lorebook, {
    id: node.fullPath,
    name: node.name,
    description: node.tagline || node.description || '',
    source: `chub:${node.fullPath}`,
  })
  if (!book) throw new Error(`"${node.name}" has no usable entries`)
  return book
}

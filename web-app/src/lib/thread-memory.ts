/**
 * Thread memory: chapters + canon, ported as settled rules from the
 * SillyTavern Simple Memory prototype (not its code — different codebase).
 *
 * Shape: raw messages -> chapters -> canon. Messages outside the raw window
 * are written into chapters; the oldest chapters fold into canon, a permanent
 * record of durable facts. Canon grows forever and shrinks only by hand.
 *
 * Branching (new design, the prototype has none — it trusts linear indices):
 * - Chapters are lineage-scoped. A chapter covers the current path iff every
 *   one of its message ids is in the path. Deletes/splices don't rewrite
 *   history: the uncovered survivors simply become owed again and refold.
 * - Canon is scoped per root (one story universe per greeting/root). Passes
 *   record their source lineage for audit.
 * - There is no scalar fold watermark. Owed work derives from chapter
 *   coverage of the current path, so branch switches need no per-branch
 *   state and there is nothing to evict, hydrate, or race.
 * - Ordinals (#N) are display-only, resolved live from the active path.
 *   Message ids are canonical.
 */

export interface MemorySizes {
  keepRaw: number
  keepChapters: number
  batch: number
  responseLength: number
}

export const MEMORY_SIZE_PRESETS: Record<string, MemorySizes> = {
  tiny: { keepRaw: 10, keepChapters: 10, batch: 5, responseLength: 150 },
  small: { keepRaw: 15, keepChapters: 15, batch: 8, responseLength: 200 },
  medium: { keepRaw: 25, keepChapters: 25, batch: 10, responseLength: 250 },
  large: { keepRaw: 50, keepChapters: 50, batch: 15, responseLength: 350 },
}

export const DEFAULT_MEMORY_SIZES: MemorySizes = {
  ...MEMORY_SIZE_PRESETS.medium,
}

/** Preset label recomputed from live values, never trusted from storage. */
export function sizePresetFor(vals: MemorySizes): string {
  for (const [name, p] of Object.entries(MEMORY_SIZE_PRESETS)) {
    if (
      vals.keepRaw === p.keepRaw &&
      vals.keepChapters === p.keepChapters &&
      vals.batch === p.batch &&
      vals.responseLength === p.responseLength
    )
      return name
  }
  return 'custom'
}

/**
 * Worst-case chapters-in-context cost: one chapter's token budget per kept
 * chapter. A label for tooltips, not a promise.
 */
export function presetCost(sizes: MemorySizes): number {
  return sizes.keepChapters * sizes.responseLength
}

/** Canon has no hard cap; past this it is worth a look. */
export function canonBudget(sizes: MemorySizes): number {
  return presetCost(sizes)
}

export const CANON_CATEGORIES = [
  'character',
  'relationship',
  'possession',
  'location',
  'goal',
  'status',
  'secret',
  'misc',
] as const

export type CanonCategory = (typeof CANON_CATEGORIES)[number]

/**
 * Where a canon entry came from — the only thing that tells a rebuild what
 * it may replace. 'fold' was derived and can be derived again; 'manual' was
 * written by hand and survives every rebuild, and the model may not revise it.
 * Editing a folded entry leaves it 'fold': promoting on edit would mean a
 * rebuild kept your version *and* re-derived the same fact beside it.
 */
export type CanonSrc = 'fold' | 'manual'

export interface CanonEntry {
  id: string
  cat: string
  text: string
  src: CanonSrc
  created: number
  updated: number
}

export interface CanonPass {
  id: string
  at: number
  /** Active-root id of the lineage this pass folded (one universe per root). */
  rootId: string
  chapterIds: string[]
  /** Canon as it stood when the pass began, non-manual entries only. */
  before: CanonEntry[]
}

export interface MemoryChapter {
  id: string
  /** Active-root id of the lineage this chapter was written on. */
  rootId: string
  /** Exact message ids summarized, in path order. */
  messageIds: string[]
  /** FNV-1a hash of each message's text at write time, for staleness badges. */
  hashes: Record<string, string>
  text: string
  tokens: number
  created: number
  /**
   * Folded into canon. Culled chapters still cover (their messages must not
   * refold) but leave the prompt chapters window.
   */
  culled?: boolean
}

export interface RootMemory {
  chapters: MemoryChapter[]
  canon: CanonEntry[]
  passes: CanonPass[]
}

export interface ThreadMemoryData {
  v: 1
  roots: Record<string, RootMemory>
  /** Pinned message ids: always sent, never folded. Thread-global. */
  pins: string[]
  updatedAt: number
}

export const EMPTY_ROOT_MEMORY: RootMemory = {
  chapters: [],
  canon: [],
  passes: [],
}

export function emptyThreadMemory(): ThreadMemoryData {
  return { v: 1, roots: {}, pins: [], updatedAt: Date.now() }
}

export function rootScope(
  data: ThreadMemoryData | undefined,
  rootId: string
): RootMemory {
  return data?.roots[rootId] ?? EMPTY_ROOT_MEMORY
}

/** One story universe per root; legacy/empty paths share the '' scope. */
export function rootIdOf(path: Array<{ id: string }>): string {
  return path[0]?.id ?? ''
}

export type MemoryAutoMode = 'on' | 'off' | 'default'

export interface ThreadMemoryMeta {
  auto?: MemoryAutoMode
  /** Per-chat size override. Absent means the global defaults. */
  sizes?: MemorySizes
}

/** Effective sizes: the chat's override wins outright, like auto. */
export function effectiveSizes(
  threadMetadata: unknown,
  global: MemorySizes
): MemorySizes {
  const sizes = (threadMetadata as { memory?: ThreadMemoryMeta } | null | undefined)
    ?.memory?.sizes
  return sizes ?? global
}

/** Untouched threads carry no metadata; anything but on/off reads as default. */
export function chatAutoMode(threadMetadata: unknown): MemoryAutoMode {
  const auto = (threadMetadata as { memory?: ThreadMemoryMeta } | null | undefined)
    ?.memory?.auto
  return auto === 'on' || auto === 'off' ? auto : 'default'
}

/** A per-chat on/off always wins outright, in either direction. */
export function effectiveAutoOn(
  threadMetadata: unknown,
  globalDefault: boolean
): boolean {
  const mode = chatAutoMode(threadMetadata)
  if (mode === 'on') return true
  if (mode === 'off') return false
  return globalDefault
}

/** Minimal message view the memory math needs. */
export interface MemoryPathMessage {
  id: string
  role: string
  text: string
}

/** Pins stay in the prompt and empties hold nothing to summarize. */
export function isCountable(m: MemoryPathMessage, pins: Set<string>): boolean {
  return !pins.has(m.id) && m.text.trim().length > 0
}

/**
 * Chapters covering this path: every message id present. Order needs no
 * check — without merges, subset membership on one chain implies order.
 */
export function chaptersForPath(
  chapters: MemoryChapter[],
  pathIds: Set<string>
): MemoryChapter[] {
  return chapters.filter((c) => c.messageIds.every((id) => pathIds.has(id)))
}

/**
 * Where the next fold resumes: index of the first countable message no
 * valid chapter covers. Chapters are whole batches from the path start, so
 * coverage is a prefix over countable messages; pins and empties never
 * break contiguity.
 */
export function coverageCursor(
  path: MemoryPathMessage[],
  chapters: MemoryChapter[],
  pins: Set<string>
): { cursor: number; coveredCount: number } {
  const pathIds = new Set(path.map((m) => m.id))
  const valid = chaptersForPath(chapters, pathIds)
  const covered = new Set<string>()
  for (const c of valid) for (const id of c.messageIds) covered.add(id)
  let coveredCount = 0
  let cursor = path.length
  for (let i = 0; i < path.length; i++) {
    const m = path[i]
    if (!isCountable(m, pins)) continue
    if (!covered.has(m.id)) {
      cursor = i
      break
    }
    coveredCount++
  }
  return { cursor, coveredCount }
}

/** Countable messages from `cursor` to the path end. */
export function owedFrom(
  path: MemoryPathMessage[],
  cursor: number,
  pins: Set<string>
): MemoryPathMessage[] {
  return foldableOwed(path, cursor, pins, 0)
}

/**
 * Foldable messages: countable, uncovered, and outside the raw window. The
 * last `keepRaw` countable messages always stay raw — floors, not ceilings.
 */
export function foldableOwed(
  path: MemoryPathMessage[],
  cursor: number,
  pins: Set<string>,
  keepRaw: number
): MemoryPathMessage[] {
  const countable: MemoryPathMessage[] = []
  for (let i = cursor; i < path.length; i++) {
    if (isCountable(path[i], pins)) countable.push(path[i])
  }
  const keep = Math.max(0, keepRaw)
  return keep > 0 ? countable.slice(0, Math.max(0, countable.length - keep)) : countable
}

/** Whole batches only, in both modes: the remainder stays raw. */
export function plannedBlocks(owedCount: number, batch: number): number {
  return Math.floor(owedCount / Math.max(1, batch))
}

/**
 * Plan chapter ranges from the cursor: consecutive whole batches of
 * foldable messages. Returns id lists, never indices.
 */
export function planChapterBatches(
  path: MemoryPathMessage[],
  cursor: number,
  pins: Set<string>,
  batch: number,
  keepRaw = 0
): string[][] {
  const owed = foldableOwed(path, cursor, pins, keepRaw)
  const n = plannedBlocks(owed.length, batch)
  const out: string[][] = []
  for (let b = 0; b < n; b++) {
    out.push(owed.slice(b * batch, (b + 1) * batch).map((m) => m.id))
  }
  return out
}

/**
 * Folded-history windowing: which message ids leave the prompt and which
 * chapters/canon text carries their meaning instead. Covered messages leave
 * even inside the raw window (they are summarized, not raw); pins always
 * stay. Culled chapters still cover but leave the chapters block.
 */
export function planMemoryWindow(opts: {
  path: MemoryPathMessage[]
  scope: RootMemory
  sizes: MemorySizes
  pins: Set<string>
}): { excludedIds: Set<string>; canon: string; chapters: string } {
  const { path, scope, sizes, pins } = opts
  const pathIds = new Set(path.map((m) => m.id))
  const valid = chaptersForPath(scope.chapters, pathIds)
  const excludedIds = new Set<string>()
  // The root stays in the prompt even once a chapter covers it. kuru sends the
  // opening message folded into the system string (templates reject a leading
  // assistant turn); withhold it and the next surviving reply is folded in its
  // place, so a mid-story line arrives labelled as the character's greeting.
  const rootId = rootIdOf(path)
  for (const c of valid) {
    for (const id of c.messageIds) {
      if (!pins.has(id) && id !== rootId) excludedIds.add(id)
    }
  }
  // A batch boundary lands mid-turn whenever the batch size is odd, leaving
  // an assistant reply first in the window with its user turn folded away.
  // The transport folds every leading assistant into the system string, so
  // that reply would arrive glued to the greeting as a second opening
  // message. Give it its user turn back instead: the raw repeat of one
  // already-summarized message is cheaper than a duplicated character voice.
  const firstKept = () =>
    path.findIndex((m) => m.id !== rootId && !excludedIds.has(m.id))
  let head = firstKept()
  while (
    head > 0 &&
    path[head].role === 'assistant' &&
    excludedIds.has(path[head - 1].id)
  ) {
    excludedIds.delete(path[head - 1].id)
    head = firstKept()
  }

  const canon = scope.canon.length > 0 ? formatCanon(scope.canon) : ''
  const live = valid
    .filter((c) => !c.culled)
    .sort((a, b) => a.created - b.created)
    .slice(-Math.max(1, sizes.keepChapters))
  let chapters = ''
  if (live.length > 0) {
    const ord = new Map(path.map((m, i) => [m.id, i] as const))
    chapters = live
      .map((c) => {
        const ords = c.messageIds
          .map((id) => ord.get(id))
          .filter((n): n is number => n !== undefined)
        const range =
          ords.length > 0
            ? ` (messages ${Math.min(...ords)}-${Math.max(...ords)})`
            : ''
        return `### Chapter${range}\n${c.text.trim()}`
      })
      .join('\n\n')
  }
  return { excludedIds, canon, chapters }
}

/**
 * Canon candidates: the oldest unculled chapters, whole batches only.
 * Keep chapters are kept; chapters past them wait until a full batch is
 * waiting — culling one at a time would pin the count and ask the model
 * for a diff per chapter instead of per window.
 */
export function canonCandidates(
  chapters: MemoryChapter[],
  keepChapters: number,
  take: number
): MemoryChapter[] {
  const live = chapters
    .filter((c) => !c.culled)
    .sort((a, b) => a.created - b.created)
  if (live.length < keepChapters + take) return []
  return live.slice(0, take)
}

/**
 * Canon answer budget: a short diff, never a rewritten record, so a
 * fraction of a chapter's room covers one chapter's changes — but the
 * first pass against an empty record must fit every chapter's facts at
 * once. max_tokens is a ceiling, not a spend.
 */
export function canonPassBudget(responseLength: number, chapters: number): number {
  const raw = Math.round(responseLength * (0.5 + 0.25 * chapters))
  return Math.min(4096, Math.max(120, raw))
}

/**
 * When the next chapter lands, counted in turns. The check runs once the
 * reply has landed, so a turn puts two messages in front of it; counting
 * raw messages would skip warning states and the chapter would arrive
 * unannounced.
 *
 * `uncovered` counts every countable message no chapter covers, the raw
 * window included, and the fold trips at `batch + keepRaw`. Counting only
 * what is already past the window froze the line at its start value for the
 * first `keepRaw` messages of every chat — a countdown that does not count.
 */
export function foldOutlook(opts: {
  uncovered: number
  batch: number
  keepRaw?: number
  perTurn?: number
}): { turns: number; blocks: number; at: number; perTurn: number } {
  const b = Math.max(1, opts.batch)
  const keep = Math.max(0, opts.keepRaw ?? 0)
  const step = Math.max(2, Math.round(opts.perTurn ?? 2) || 2)
  const turns = Math.max(1, Math.ceil((b + keep - opts.uncovered) / step))
  const at = opts.uncovered + step * turns
  return { turns, blocks: plannedBlocks(Math.max(0, at - keep), b), at, perTurn: step }
}

/**
 * Messages one turn actually adds, measured rather than assumed: a solo
 * chat adds two, but other rhythms add more, and a hardcoded two would
 * count down early and skip the "1 turn" warning.
 */
export function messagesPerTurn(roles: string[]): number {
  if (roles.length < 2) return 2
  let lastUser = -1
  for (let i = roles.length - 1; i >= 0; i--) {
    if (roles[i] === 'user') {
      lastUser = i
      break
    }
  }
  if (lastUser < 0) return 2
  if (lastUser === roles.length - 1) {
    let prevUser = -1
    for (let i = lastUser - 1; i >= 0; i--) {
      if (roles[i] === 'user') {
        prevUser = i
        break
      }
    }
    return prevUser < 0 ? 2 : Math.max(2, lastUser - prevUser)
  }
  return Math.max(2, roles.length - lastUser)
}

export interface MemoryStatus {
  sizes: MemorySizes
  scope: RootMemory
  pins: Set<string>
  /** Chapters valid on this path, and the ids they cover. */
  chapters: MemoryChapter[]
  foldedIds: Set<string>
  /** Chapters still in the prompt (culled ones cover but do not show). */
  liveChapters: number
  /** Countable messages no chapter covers, raw window included. */
  uncovered: number
  /** Of those, the ones already past the raw window: what a fold would take. */
  waiting: number
  outlook: ReturnType<typeof foldOutlook>
  /** A whole batch is waiting: the next reply folds. */
  ready: boolean
  auto: MemoryAutoMode
  autoOn: boolean
  /** Roots (other greetings/branch families) holding memory of their own. */
  otherRoots: number
  hasAny: boolean
}

/**
 * The one source for every fold line — chat strip, context sheet, and
 * anything else that reports memory. Two copies of this math drifted apart
 * once already; the prototype keeps one `foldOutlook` for the same reason.
 */
export function memoryStatus(opts: {
  path: MemoryPathMessage[]
  record: ThreadMemoryData | undefined
  threadMetadata: unknown
  autoDefault: boolean
  globalSizes?: MemorySizes
}): MemoryStatus {
  const { path, record, threadMetadata, autoDefault } = opts
  const sizes = effectiveSizes(threadMetadata, opts.globalSizes ?? DEFAULT_MEMORY_SIZES)
  const pins = new Set(record?.pins ?? [])
  const pathIds = new Set(path.map((m) => m.id))
  const rootId = rootIdOf(path)
  const scope = rootScope(record, rootId)
  const chapters = chaptersForPath(scope.chapters, pathIds)
  const foldedIds = new Set<string>()
  for (const c of chapters) for (const id of c.messageIds) foldedIds.add(id)
  const { cursor } = coverageCursor(path, scope.chapters, pins)
  const uncovered = owedFrom(path, cursor, pins).length
  const waiting = Math.max(0, uncovered - sizes.keepRaw)
  const outlook = foldOutlook({
    uncovered,
    batch: sizes.batch,
    keepRaw: sizes.keepRaw,
    perTurn: messagesPerTurn(path.map((m) => m.role)),
  })
  const otherRoots = Object.entries(record?.roots ?? {}).filter(
    ([id, r]) => id !== rootId && (r.chapters.length > 0 || r.canon.length > 0)
  ).length
  return {
    sizes,
    scope,
    pins,
    chapters,
    foldedIds,
    liveChapters: chapters.filter((c) => !c.culled).length,
    uncovered,
    waiting,
    outlook,
    ready: waiting >= sizes.batch,
    auto: chatAutoMode(threadMetadata),
    autoOn: effectiveAutoOn(threadMetadata, autoDefault),
    otherRoots,
    hasAny:
      scope.chapters.length > 0 ||
      scope.canon.length > 0 ||
      pins.size > 0 ||
      otherRoots > 0,
  }
}

/**
 * The fold line itself. Auto off means nothing folds on its own, so the
 * countdown would be a promise nothing keeps — the prototype says Paused
 * there and keeps the waiting count visible.
 */
export function foldLine(status: MemoryStatus): string {
  if (!status.autoOn) {
    return status.waiting > 0
      ? `Paused · ${status.waiting} waiting`
      : 'Paused'
  }
  if (status.outlook.turns <= 1) {
    return status.outlook.blocks > 1
      ? `Next reply writes ${status.outlook.blocks} chapters`
      : 'Next reply writes a chapter'
  }
  return `Next chapter in ${status.outlook.turns} turns`
}

/** 1-based ordinal on the given path; display-only, ids stay canonical. */
export function ordinalOf(path: Array<{ id: string }>, id: string): number {
  const i = path.findIndex((m) => m.id === id)
  return i < 0 ? -1 : i + 1
}

/** Live-resolved display range for a chapter on the current path. */
export function chapterRange(
  chapter: MemoryChapter,
  path: Array<{ id: string }>
): { from: number; to: number } | null {
  const ords = chapter.messageIds
    .map((id) => ordinalOf(path, id))
    .filter((n) => n > 0)
  if (ords.length === 0) return null
  return { from: Math.min(...ords), to: Math.max(...ords) }
}

export interface ChapterHealth {
  edited: string[]
  missing: string[]
}

/** Staleness badges: edited text vs ids gone from the store entirely. */
export function chapterHealth(
  chapter: MemoryChapter,
  lookup: (id: string) => string | undefined
): ChapterHealth {
  const edited: string[] = []
  const missing: string[] = []
  for (const id of chapter.messageIds) {
    const text = lookup(id)
    if (text === undefined) {
      missing.push(id)
      continue
    }
    if (hashText(text) !== chapter.hashes[id]) edited.push(id)
  }
  return { edited, missing }
}

export function hashText(s: string): string {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(16)
}

const NOTHING_DURABLE = /^\(?\s*nothing durable\s*\)?[.!]?$/i
const META_LINE =
  /^(let me know|i hope|feel free|note:|here (is|are)|that('s| is) all|end of (notes|summary)|no (more|further) (plot|new|events|notes|developments))|please (continue|let me know)/i
const NOTHING_NEW = /^\(?\s*nothing new\s*\)?[.!]?$/i

/**
 * The canon pass emits a diff, never a rewritten record, so an entry it
 * does not mention survives byte-exact. Accepts a bare `category | fact`
 * line (known categories only — any prose with a pipe is not a fact), and
 * files unknown categories under misc rather than dropping them.
 */
export function parseCanonDiff(text: string): {
  adds: Array<{ cat: string; text: string }>
  revisions: Array<{ id: string; text: string }>
} {
  const adds: Array<{ cat: string; text: string }> = []
  const revisions: Array<{ id: string; text: string }> = []
  for (const raw of String(text ?? '').split('\n')) {
    const line = raw.trim()
    if (!line || NOTHING_DURABLE.test(line) || META_LINE.test(line)) continue
    const revise = line.match(/^~\s*([A-Za-z]\d+)\s*\|?\s*(.+)$/)
    if (revise) {
      revisions.push({ id: revise[1], text: revise[2].trim() })
      continue
    }
    let add = line.match(/^[+*-]\s*(?:([A-Za-z]+)\s*\|)?\s*(.+)$/)
    if (!add) {
      const bare = line.match(/^([A-Za-z]+)\s*\|\s*(.+)$/)
      if (!bare || !CANON_CATEGORIES.includes(bare[1].toLowerCase() as CanonCategory))
        continue
      add = bare
    }
    const body = add[2].trim()
    if (!body || NOTHING_DURABLE.test(body)) continue
    const cat = String(add[1] ?? '').toLowerCase()
    adds.push({
      cat: CANON_CATEGORIES.includes(cat as CanonCategory) ? cat : 'misc',
      text: body,
    })
  }
  return { adds, revisions }
}

export function nextCanonId(canon: CanonEntry[]): string {
  let max = 0
  for (const e of canon) {
    const m = /^C(\d+)$/.exec(e.id)
    if (m) max = Math.max(max, Number(m[1]))
  }
  return `C${max + 1}`
}

const isManual = (e: CanonEntry): boolean => e.src === 'manual'

/**
 * Applies a parsed diff. Revisions touch folded entries only — a manual
 * entry's wording is a promise, shown to the model for context but never
 * rewritten by it. Adds skip exact-text duplicates (case-insensitive), so
 * two lineages folding the same fact don't stack copies.
 */
export function applyCanonDiff(
  canon: CanonEntry[],
  diff: ReturnType<typeof parseCanonDiff>,
  at: number,
  makeId: (canon: CanonEntry[]) => string = nextCanonId
): { canon: CanonEntry[]; added: number; revised: number } {
  const next = canon.map((e) => ({ ...e }))
  let revised = 0
  for (const r of diff.revisions) {
    const entry = next.find((e) => e.id === r.id)
    if (!entry || isManual(entry)) continue
    entry.text = r.text
    entry.updated = at
    revised++
  }
  let added = 0
  for (const a of diff.adds) {
    const duplicate = next.some(
      (e) => e.text.toLowerCase() === a.text.toLowerCase()
    )
    if (duplicate) continue
    next.push({
      id: makeId(next),
      cat: a.cat,
      text: a.text,
      src: 'fold',
      created: at,
      updated: at,
    })
    added++
  }
  return { canon: next, added, revised }
}

/** Concrete line budget beats "be brief" for smaller models. */
export function maxSummaryLines(
  messageCount: number,
  responseLength: number
): number {
  const perMessage = Math.round(Math.max(1, messageCount) * 0.8)
  return Math.max(3, Math.min(perMessage, 12, Math.floor(responseLength / 15)))
}

export const EMPTY_CHAPTER_TEXT = '- (no notable events)'

/**
 * Forces a summary into one "- " line per event, whatever shape the model
 * returned. Idempotent, so stored chapters can be re-run through it.
 */
export function normalizeSummary(text: string): string {
  const lines = String(text ?? '')
    .split('\n')
    .map((line) =>
      line
        .trim()
        .replace(/^[-*•]\s*/, '')
        .replace(/^\d+[.)]\s*/, '')
        .trim()
    )
    .filter(
      (line) =>
        line &&
        !NOTHING_NEW.test(line) &&
        !META_LINE.test(line) &&
        !/^#{1,6}\s/.test(line)
    )
  return lines.map((l) => `- ${l}`).join('\n')
}

export const DEFAULT_CHAPTER_PROMPT = `You are distilling new story developments from a roleplay chat into concise memory notes.

Write a compact, chronological summary of the messages below, following these rules:
- Record ONLY new plot events, state changes, and decisions not already present in the prior story notes.
- Never restate established personality, appearance, or lore. Never include greetings, filler, or trivial banter.
- Refer to characters by name, not pronouns. Write in past tense, terse prose, no dialogue.
- Output at most {{maxlines}} "- " lines, notes only, no commentary.

## Prior story notes (already summarized - do not repeat)
{{prior}}

## Messages to summarize
{{messages}}

## Notes:`;

export function buildChapterPrompt(
  messagesText: string,
  opts: { maxLines: number; prior?: string; template?: string }
): string {
  let prompt = (opts.template ?? DEFAULT_CHAPTER_PROMPT)
    .replaceAll('{{maxlines}}', String(opts.maxLines))
    .replaceAll('{{messages}}', messagesText)
  if (opts.prior?.trim()) {
    prompt = prompt.replaceAll('{{prior}}', opts.prior.trim())
  } else {
    prompt = prompt.replace(/## Prior story notes[\s\S]*?(?=\n##)/, '')
    prompt = prompt.replaceAll('{{prior}}', '')
  }
  return prompt.replace(/\n{3,}/g, '\n\n')
}

export const DEFAULT_CANON_PROMPT = `You keep the permanent record of a roleplay story. The chapters below are about to be discarded: whatever you do not record here is lost for good.

Record anything a reader would need to follow the story later, under one of these categories:
character, relationship, possession, location, goal, status, secret, misc

Rules:
1. Output only CHANGES to the record. Never repeat something the record already states correctly.
2. A chapter that introduces a person, a place, a bond, a possession, a plan, a promise, an injury or a change in someone's situation has something worth recording. Scenery, mood, weather and one-off banter do not.
3. Facts, not narration. Under 15 words per line. Name the characters.
4. If a chapter completes, cancels or contradicts an existing entry, revise that entry instead of adding a new one.

Format, one change per line:
+ category | the new fact
~ id | the corrected fact

Example:
+ relationship | Mira distrusts the harbor patrol since the reed ambush.
+ goal | Toren must reach the Archivist before the tide turns.
~ c4 | Toren delivered the seal; his debt to Mira is paid.

Answer with "+" and "~" lines only, no other text. If these chapters truly add nothing to the record, answer exactly: (nothing durable)

## Permanent record so far
{{canon}}

## Chapters being discarded
{{chapters}}

## Changes (only "+" and "~" lines):`;

export function buildCanonPrompt(
  canonText: string,
  chaptersText: string,
  template?: string
): string {
  return (template ?? DEFAULT_CANON_PROMPT)
    .replaceAll('{{canon}}', canonText.trim() || '(empty)')
    .replaceAll('{{chapters}}', chaptersText.trim())
}

export function formatCanon(canon: CanonEntry[]): string {
  return canon.map((e) => `[${e.id}] ${e.cat} | ${e.text}`).join('\n')
}

export interface MemoryExportFile {
  kind: 'kuru-thread-memory'
  v: 1
  threadId: string
  exportedAt: number
  chapters: MemoryChapter[]
  canon: CanonEntry[]
}

export function exportMemory(
  threadId: string,
  data: ThreadMemoryData
): MemoryExportFile {
  const chapters = Object.values(data.roots).flatMap((r) => r.chapters)
  const canon = Object.values(data.roots).flatMap((r) => r.canon)
  return {
    kind: 'kuru-thread-memory',
    v: 1,
    threadId,
    exportedAt: Date.now(),
    chapters,
    canon,
  }
}

export function isMemoryExportFile(v: unknown): v is MemoryExportFile {
  const f = v as MemoryExportFile | null | undefined
  return (
    !!f &&
    f.kind === 'kuru-thread-memory' &&
    f.v === 1 &&
    Array.isArray(f.chapters) &&
    Array.isArray(f.canon)
  )
}

/**
 * Cross-thread import: canon imports directly (index-free, always safe;
 * exact-text duplicates skipped like a normal fold) with fresh ids.
 * Chapter *texts* come back for a canon-fold pass; their message ids are
 * never trusted outside their own thread.
 */
export function importCanonEntries(
  canon: CanonEntry[],
  file: MemoryExportFile,
  at: number
): { canon: CanonEntry[]; imported: number } {
  const diff = {
    adds: file.canon.map((e) => ({ cat: e.cat, text: e.text })),
    revisions: [] as Array<{ id: string; text: string }>,
  }
  const { canon: next, added } = applyCanonDiff(canon, diff, at)
  return { canon: next, imported: added }
}

/**
 * Same-thread restore: chapters take the file's version outright (nothing
 * manual about them). Canon takes the file too, but any *current* manual
 * entry whose text isn't already there is carried forward — the one thing
 * nothing automatic may delete.
 */
export function restoreSameThread(
  current: RootMemory,
  file: { chapters: MemoryChapter[]; canon: CanonEntry[] }
): { chapters: MemoryChapter[]; canon: CanonEntry[]; carried: number } {
  const canon = file.canon.map((e) => ({ ...e }))
  let carried = 0
  for (const e of current.canon) {
    if (!isManual(e)) continue
    const present = canon.some(
      (c) => c.text.toLowerCase() === e.text.toLowerCase()
    )
    if (!present) {
      canon.push({ ...e })
      carried++
    }
  }
  return { chapters: file.chapters.map((c) => ({ ...c })), canon, carried }
}

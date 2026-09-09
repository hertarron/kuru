/**
 * The lorebook scanner: decides which entries fire for the next prompt, and
 * where each one lands.
 *
 * Split into pure steps so both the transport and its tests can drive it:
 *
 *  1. `scanLorebooks` walks the live books against the recent chat text and
 *     returns the entries that fired, in prominence order, trimmed to a token
 *     budget.
 *  2. `groupActivations` sorts those into the slots the prompt has: three
 *     inside the system string (before/after the character definition, around
 *     the example dialogue) and one inside the message array (at depth).
 *  3. `applyDepthInjections` puts the at-depth text into the message array.
 *
 * Where this deliberately differs from SillyTavern:
 *
 *  - At-depth text is merged into the message at the depth boundary instead of
 *    being inserted as its own turn. Strict chat templates (Qwen3.5+, several
 *    llama.cpp Jinja presets) require roles to alternate starting with user, so
 *    a standalone system/assistant turn mid-conversation is a hard error there.
 *    The entry's `role` is kept in the data (imports round-trip) but does not
 *    change placement.
 *  - Positions ST supports and kuru does not (author's note, outlet) are
 *    already redirected by `effectivePosition`.
 */

import { estimateTokens } from './context-manager'
import { renderInstructions } from './instructionTemplate'
import {
  LorebookLogic,
  LorebookPosition,
  effectivePosition,
  type Lorebook,
  type LorebookEntry,
} from './lorebook'

export interface LorebookRuntimeSettings {
  /** Master switch. Off means no world info anywhere. */
  enabled: boolean
  /** How many of the most recent messages are scanned for keys. */
  scanDepth: number
  /** Share of the context window world info may take. 0 disables the cap. */
  budgetPercent: number
  /** How many times the scan repeats over newly activated content, for the
   * books that allow it. */
  maxRecursionSteps: number
}

export const defaultLorebookRuntimeSettings: LorebookRuntimeSettings = {
  enabled: true,
  scanDepth: 4,
  budgetPercent: 25,
  maxRecursionSteps: 2,
}

export interface ActivatedEntry {
  entry: LorebookEntry
  book: Lorebook
  /** After `effectivePosition`, so this is a slot that actually exists. */
  position: number
  /** Content with macros resolved -- what reaches the model. */
  text: string
  /** 0 = fired against the chat, 1+ = fired against other entries' content. */
  step: number
}

const REGEX_KEY = /^\/(.+)\/([gimsuy]*)$/

/**
 * `/pattern/flags` becomes a regex, used exactly as written -- that is the
 * point of writing one, and adding flags behind the author's back makes `i`
 * impossible to opt out of. Anything else is a plain-text key: whole-word and
 * case-insensitive, the two defaults that stop `orc` firing on `orchard` and
 * `Rune` missing `rune`. An entry imported with ST's overrides keeps them.
 */
function compileKey(
  key: string,
  caseSensitive: boolean,
  wholeWords: boolean
): RegExp | null {
  const literal = key.match(REGEX_KEY)
  if (literal) {
    try {
      return new RegExp(literal[1], literal[2])
    } catch {
      // A malformed literal is far more likely a key that happens to contain
      // slashes than an intended pattern, so fall through to plain matching.
    }
  }
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  // Word boundaries only do the right thing next to word characters; a key like
  // "!!" or a CJK term would never match with them bolted on.
  const lead = /^\w/.test(key) ? '\\b' : ''
  const tail = /\w$/.test(key) ? '\\b' : ''
  const body = wholeWords ? `${lead}${escaped}${tail}` : escaped
  try {
    return new RegExp(body, caseSensitive ? '' : 'i')
  } catch {
    return null
  }
}

const someKeyMatches = (
  keys: string[],
  haystack: string,
  entry: LorebookEntry,
  wholeWords: boolean
): boolean =>
  keys.some((key) => {
    const re = compileKey(key, entry.caseSensitive, wholeWords)
    return re ? re.test(haystack) : false
  })

const everyKeyMatches = (
  keys: string[],
  haystack: string,
  entry: LorebookEntry,
  wholeWords: boolean
): boolean =>
  keys.every((key) => {
    const re = compileKey(key, entry.caseSensitive, wholeWords)
    return re ? re.test(haystack) : false
  })

/**
 * Plain keys match whole words unless an imported entry says otherwise. ST
 * exposes this per entry; here it is a default with no switch, since a key that
 * should match inside a word is what regex is for.
 */
export function wholeWordsFor(entry: LorebookEntry): boolean {
  const own = entry.extra.matchWholeWords
  return typeof own === 'boolean' ? own : true
}

/**
 * Whether an entry's keys fire against the scanned text. Keyless entries are
 * always on -- that is the one input the entry editor exposes, standing in for
 * ST's separate `constant` flag.
 */
export function entryMatches(entry: LorebookEntry, haystack: string): boolean {
  const whole = wholeWordsFor(entry)
  if (entry.keys.length === 0) return true
  if (!someKeyMatches(entry.keys, haystack, entry, whole)) return false
  if (entry.secondaryKeys.length === 0) return true

  switch (entry.selectiveLogic) {
    case LorebookLogic.NOT_ALL:
      return !everyKeyMatches(entry.secondaryKeys, haystack, entry, whole)
    case LorebookLogic.NOT_ANY:
      return !someKeyMatches(entry.secondaryKeys, haystack, entry, whole)
    case LorebookLogic.AND_ALL:
      return everyKeyMatches(entry.secondaryKeys, haystack, entry, whole)
    case LorebookLogic.AND_ANY:
    default:
      return someKeyMatches(entry.secondaryKeys, haystack, entry, whole)
  }
}

const flag = (entry: LorebookEntry, name: string): boolean =>
  entry.extra[name] === true

/**
 * Scan the live books and return what fires, most prominent first.
 *
 * `chatText` is the recent conversation, newest last, already cut to the scan
 * depth by the caller. `budgetTokens` of 0 or less means no cap.
 */
export function scanLorebooks({
  books,
  chatText,
  settings,
  budgetTokens = 0,
  vars,
  random = Math.random,
}: {
  books: Lorebook[]
  chatText: string
  settings: LorebookRuntimeSettings
  budgetTokens?: number
  vars?: { char?: string; user?: string }
  random?: () => number
}): { activated: ActivatedEntry[]; droppedForBudget: ActivatedEntry[] } {
  if (!settings.enabled || books.length === 0) {
    return { activated: [], droppedForBudget: [] }
  }

  const pending: Array<{ entry: LorebookEntry; book: Lorebook }> = []
  for (const book of books) {
    for (const entry of book.entries) {
      if (!entry.enabled) continue
      if (!entry.content.trim()) continue
      pending.push({ entry, book })
    }
  }

  const hit: ActivatedEntry[] = []
  const taken = new Set<LorebookEntry>()
  let haystack = chatText
  // Recursion is a property of the book: nested lore lives in the book that
  // was written for it, and a book that doesn't want it is unaffected by one
  // that does.
  const anyRecursive = books.some((b) => b.recursiveScanning)
  const maxSteps = anyRecursive ? Math.max(1, settings.maxRecursionSteps) : 1

  for (let step = 0; step < maxSteps; step++) {
    const fired: ActivatedEntry[] = []
    for (const { entry, book } of pending) {
      if (taken.has(entry)) continue
      // ST's `preventRecursion`: the entry may only fire against the chat.
      if (step > 0 && (!book.recursiveScanning || flag(entry, 'preventRecursion')))
        continue
      if (!entryMatches(entry, haystack)) continue
      if (entry.probability < 100 && random() * 100 >= entry.probability) {
        continue
      }
      taken.add(entry)
      fired.push({
        entry,
        book,
        position: effectivePosition(entry).position,
        text: renderInstructions(entry.content.trim(), vars),
        step,
      })
    }
    if (fired.length === 0) break
    hit.push(...fired)
    if (step + 1 >= maxSteps) break
    // ST's `excludeRecursion`: this entry's content must not trigger others.
    const feed = fired
      .filter((a) => a.book.recursiveScanning && !flag(a.entry, 'excludeRecursion'))
      .map((a) => a.text)
    if (feed.length === 0) break
    haystack = `${haystack}\n${feed.join('\n')}`
  }

  // Prominence decides who survives a tight budget, and who leads in a slot.
  hit.sort((a, b) => b.entry.order - a.entry.order)

  if (budgetTokens <= 0) return { activated: hit, droppedForBudget: [] }

  const activated: ActivatedEntry[] = []
  const droppedForBudget: ActivatedEntry[] = []
  let used = 0
  for (const a of hit) {
    const cost = estimateTokens(a.text)
    // Never drop everything: one oversized entry the user wrote on purpose
    // still goes in, and the context trimmer deals with the consequences.
    if (used + cost > budgetTokens && activated.length > 0) {
      droppedForBudget.push(a)
      continue
    }
    used += cost
    activated.push(a)
  }
  return { activated, droppedForBudget }
}

export interface LorebookInjections {
  beforeChar: ActivatedEntry[]
  afterChar: ActivatedEntry[]
  emTop: ActivatedEntry[]
  emBottom: ActivatedEntry[]
  /** Keyed by depth, so one merge per boundary. */
  atDepth: Map<number, ActivatedEntry[]>
}

export const emptyInjections = (): LorebookInjections => ({
  beforeChar: [],
  afterChar: [],
  emTop: [],
  emBottom: [],
  atDepth: new Map(),
})

export function groupActivations(
  activated: ActivatedEntry[]
): LorebookInjections {
  const out = emptyInjections()
  for (const a of activated) {
    switch (a.position) {
      case LorebookPosition.afterChar:
        out.afterChar.push(a)
        break
      case LorebookPosition.EMTop:
        out.emTop.push(a)
        break
      case LorebookPosition.EMBottom:
        out.emBottom.push(a)
        break
      case LorebookPosition.atDepth: {
        const depth = Math.max(0, Math.round(a.entry.depth))
        const bucket = out.atDepth.get(depth)
        if (bucket) bucket.push(a)
        else out.atDepth.set(depth, [a])
        break
      }
      default:
        out.beforeChar.push(a)
    }
  }
  return out
}

export const hasInjections = (i: LorebookInjections): boolean =>
  i.beforeChar.length > 0 ||
  i.afterChar.length > 0 ||
  i.emTop.length > 0 ||
  i.emBottom.length > 0 ||
  i.atDepth.size > 0

export const joinEntries = (entries: ActivatedEntry[]): string =>
  entries
    .map((a) => a.text)
    .filter((t) => t.trim().length > 0)
    .join('\n\n')

/**
 * Text from the most recent `depth` messages, newest last -- the haystack keys
 * are scanned against. Only text parts count: matching on reasoning or tool
 * payloads makes entries fire for words nobody in the chat ever wrote.
 */
export function chatScanText(
  messages: Array<{
    parts?: Array<{ type: string; text?: string }>
  }>,
  depth: number
): string {
  const window = depth > 0 ? messages.slice(-depth) : messages
  return window
    .map((m) =>
      (m.parts ?? [])
        .filter((p) => p.type === 'text')
        .map((p) => p.text ?? '')
        .join('\n')
    )
    .filter((t) => t.trim().length > 0)
    .join('\n')
}

type MergeableMessage = { role: string; content: unknown }

function prependText<T extends MergeableMessage>(message: T, text: string): T {
  const { content } = message
  if (typeof content === 'string') {
    return { ...message, content: `${text}\n\n${content}` }
  }
  if (Array.isArray(content)) {
    return { ...message, content: [{ type: 'text', text }, ...content] }
  }
  return message
}

function appendText<T extends MergeableMessage>(message: T, text: string): T {
  const { content } = message
  if (typeof content === 'string') {
    return { ...message, content: `${content}\n\n${text}` }
  }
  if (Array.isArray(content)) {
    return { ...message, content: [...content, { type: 'text', text }] }
  }
  return message
}

/**
 * Merge at-depth entries into the message array. Depth N puts the text
 * immediately before the Nth message from the end, so depth 1 lands between the
 * character's last reply and the newest user message; depth 0 goes after the
 * last message. See the file header for why this merges into a turn rather than
 * inserting one.
 *
 * When there are no messages to merge into, the entries come back as `unplaced`
 * so the caller can fold them into the system string instead of dropping them.
 */
export function applyDepthInjections<T extends MergeableMessage>(
  messages: T[],
  atDepth: Map<number, ActivatedEntry[]>
): { messages: T[]; unplaced: ActivatedEntry[] } {
  if (atDepth.size === 0) return { messages, unplaced: [] }
  if (messages.length === 0) {
    return { messages, unplaced: [...atDepth.values()].flat() }
  }

  const out = [...messages]
  for (const [depth, entries] of atDepth) {
    const text = joinEntries(entries)
    if (!text) continue
    if (depth === 0) {
      const last = out.length - 1
      out[last] = appendText(out[last], text)
      continue
    }
    // Depth deeper than the window collapses onto the oldest message, which is
    // also what ST does once the chat is shorter than the requested depth.
    const target = Math.min(
      Math.max(0, out.length - depth),
      out.length - 1
    )
    out[target] = prependText(out[target], text)
  }
  return { messages: out, unplaced: [] }
}

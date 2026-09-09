/**
 * Opinionated roleplay text cleanups, applied to the prompt only.
 *
 * Deliberately prompt-side and never display-side: kuru's context visualizer
 * makes what the model received inspectable, so a transform that happens there
 * can be seen and understood. A display-side transform is visible nowhere, and
 * a message that was entirely OOC would render as an empty bubble with no
 * indication anything had been removed.
 *
 * The other half of that rule lives in the caller: these run over *prior*
 * turns only. Stripping the turn being sent would mean an OOC question never
 * reaches the model, which is the one thing OOC is most used for.
 */

/**
 * Out-of-character asides, in the three notations that circulate:
 * `((like this))`, `[OOC: like this]`, and `<ooc>like this</ooc>`.
 *
 * Double parentheses are the risky one — single parens are ordinary prose, so
 * the pattern requires both and refuses to span a blank line, which keeps a
 * stray `((` from eating the rest of a message.
 */
const OOC_PATTERNS: RegExp[] = [
  /\(\((?:(?!\n\s*\n)[\s\S])*?\)\)/g,
  /\[\s*OOC\s*:[^\]]*\]/gi,
  /<ooc>[\s\S]*?<\/ooc>/gi,
]

/** Collapses the blank lines left behind once a block is cut out. */
const tidy = (text: string): string =>
  text
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()

export function stripOoc(text: string): string {
  let out = text
  for (const pattern of OOC_PATTERNS) out = out.replace(pattern, '')
  return tidy(out)
}

/**
 * Drops a dangling final fragment left by a reply that hit the output cap
 * mid-sentence, so the next turn isn't asked to continue from a half-word.
 *
 * Only fires when there is a sentence boundary to fall back to and the
 * fragment is short enough to be a truncation rather than a deliberate
 * unpunctuated ending — a one-line action beat with no full stop is a normal
 * way to write, and must survive.
 */
const SENTENCE_END = /[.!?…"'”’*)\]]\s*$/
const MAX_FRAGMENT = 200

export function trimIncompleteSentence(text: string): string {
  const trimmed = text.trimEnd()
  if (!trimmed || SENTENCE_END.test(trimmed)) return text

  const lastBoundary = Math.max(
    trimmed.lastIndexOf('.'),
    trimmed.lastIndexOf('!'),
    trimmed.lastIndexOf('?'),
    trimmed.lastIndexOf('…')
  )
  if (lastBoundary < 0) return text
  if (trimmed.length - lastBoundary > MAX_FRAGMENT) return text

  // Keep any closing quote/asterisk that follows the punctuation.
  const tail = trimmed.slice(lastBoundary + 1)
  const closers = tail.match(/^["'”’*)\]]+/)?.[0] ?? ''
  return trimmed.slice(0, lastBoundary + 1) + closers
}

export interface RpTextSettings {
  /** Strip OOC asides from prior turns before they reach the model. */
  hideOocFromPrompt: boolean
  /** Drop a truncated trailing fragment from prior assistant turns. */
  trimIncompleteSentence: boolean
}

export const defaultRpTextSettings: RpTextSettings = {
  hideOocFromPrompt: true,
  trimIncompleteSentence: false,
}

/** Both cleanups in one pass, for one prior turn's text. */
export function cleanPriorTurn(
  text: string,
  settings: RpTextSettings
): string {
  let out = text
  if (settings.hideOocFromPrompt) out = stripOoc(out)
  if (settings.trimIncompleteSentence) out = trimIncompleteSentence(out)
  return out
}

type TextPart = { type: string; text?: string }
type CleanableMessage = { role: string; content: unknown }

const cleanContent = (content: unknown, settings: RpTextSettings): unknown => {
  if (typeof content === 'string') return cleanPriorTurn(content, settings)
  if (!Array.isArray(content)) return content
  return (content as TextPart[]).map((part) =>
    part?.type === 'text' && typeof part.text === 'string'
      ? { ...part, text: cleanPriorTurn(part.text, settings) }
      : part
  )
}

/**
 * Applies the cleanups to every turn *before* the current one.
 *
 * The current turn is the last user message, and anything after it (a
 * continue-prefill) belongs to it. Cleaning that would strip an OOC question
 * before the model ever read it, so the boundary is load-bearing rather than
 * conservative. Tool messages are left alone: their content is machine
 * output, and `((...))` in it is data, not an aside.
 */
export function cleanPriorTurns<T extends CleanableMessage>(
  messages: T[],
  settings: RpTextSettings
): T[] {
  if (!settings.hideOocFromPrompt && !settings.trimIncompleteSentence) {
    return messages
  }
  let boundary = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === 'user') {
      boundary = i
      break
    }
  }
  if (boundary <= 0) return messages

  return messages.map((m, i) =>
    i < boundary && m.role !== 'tool'
      ? ({ ...m, content: cleanContent(m.content, settings) } as T)
      : m
  )
}

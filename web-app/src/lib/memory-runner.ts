import { generateText } from 'ai'
import { ulid } from 'ulidx'
import { toast } from 'sonner'
import { ModelFactory } from './model-factory'
import { useModelProvider } from '@/hooks/useModelProvider'
import { useMessages } from '@/hooks/useMessages'
import { useThreads } from '@/hooks/useThreads'
import { useCharacters, resolveThreadCharacter } from '@/hooks/useCharacters'
import { resolveThreadPersona } from '@/hooks/usePersonas'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import { isRoleplayCharacter } from './character-card'
import { computeActivePath } from './message-branching'
import { estimateTokens } from './context-manager'
import { memoryGlobalSettings } from '@/hooks/useMemorySettings'
import { useThreadMemory } from '@/hooks/useThreadMemory'
import { useMemoryRuns } from '@/hooks/useMemoryRuns'
import {
  EMPTY_CHAPTER_TEXT,
  applyCanonDiff,
  buildCanonPrompt,
  buildChapterPrompt,
  canonCandidates,
  canonPassBudget,
  chaptersForPath,
  coverageCursor,
  effectiveAutoOn,
  effectiveSizes,
  formatCanon,
  hashText,
  maxSummaryLines,
  normalizeSummary,
  parseCanonDiff,
  planChapterBatches,
  rootIdOf,
  rootScope,
  DEFAULT_MEMORY_SIZES,
  type CanonEntry,
} from './thread-memory'

export interface MemorySummarizer {
  (
    prompt: string,
    opts: { maxTokens: number; signal: AbortSignal }
  ): Promise<string>
}

export interface FoldRunResult {
  chaptersWritten: number
  canonPasses: number
  canonChanged: number
  stopped: boolean
  didNothing: boolean
}

export interface FoldRunOptions {
  threadId: string
  /** Manual runs skip the auto gate (and the early return, so owed canon still runs). */
  manual?: boolean
  summarize?: MemorySummarizer
  now?: () => number
  makeId?: () => string
}

const inFlight = new Map<string, AbortController>()

export function isMemoryFolding(threadId: string): boolean {
  return inFlight.has(threadId)
}

/** A future Stop button lands here; the runner checks between chapters. */
export function requestMemoryStop(threadId: string): void {
  inFlight.get(threadId)?.abort()
}

const threadText = (m: {
  content?: Array<{ type?: string; text?: { value?: string } }>
}): string =>
  (m.content ?? [])
    .filter((c) => c?.type === 'text')
    .map((c) => c.text?.value ?? '')
    .join('\n')

/**
 * Strip thinking blocks the way titles do: a chapter must be notes, not the
 * model's deliberation about the notes. `enable_thinking: false` covers
 * llama.cpp; this covers everyone else.
 */
export function stripReasoning(text: string): string {
  let out = String(text ?? '').replace(
    /<(think|thinking|reasoning|analysis)[^>]*>[\s\S]*?<\/\1>/gi,
    ''
  )
  const lastClose = out.match(
    /<\/(?:think|thinking|reasoning|analysis)>\s*([\s\S]*)$/i
  )
  if (lastClose) out = lastClose[1]
  // An opening tag with no close is a model that ran out of budget mid-
  // thought. Keeping the tail would file its deliberation as the chapter,
  // so this reads as no answer and the caller retries with more room.
  out = out.replace(/<(?:think|thinking|reasoning|analysis)[^>]*>[\s\S]*$/i, '')
  return out.replace(/<[^>]+>/g, '').trim()
}

/**
 * Fold summarization through the chat model on the reserved background slot
 * (same pinning as thread titles), so a fold can never evict the chat's own
 * KV cache from slot 0.
 */
export async function summarizeWithChatModel(
  prompt: string,
  opts: { maxTokens: number; signal: AbortSignal }
): Promise<string> {
  const { selectedModel, selectedProvider, getProviderByName } =
    useModelProvider.getState()
  if (!selectedModel || !selectedProvider) {
    throw new Error('No model selected for folding')
  }
  const provider = getProviderByName(selectedProvider)
  if (!provider) throw new Error('Model provider not found for folding')

  const params: Record<string, unknown> = {}
  if (selectedProvider === 'llamacpp') {
    params.chat_template_kwargs = { enable_thinking: false }
    const userParallel = Number(
      provider.settings?.find((s) => s.key === 'parallel')?.controller_props
        ?.value ?? 1
    )
    if (Number.isFinite(userParallel) && userParallel > 0) {
      params.id_slot = userParallel
    }
  }
  const model = await ModelFactory.createModel(
    selectedModel.id,
    provider,
    params
  )
  const { text } = await generateText({
    model,
    messages: [{ role: 'user', content: prompt }],
    maxOutputTokens: opts.maxTokens,
    abortSignal: opts.signal,
  })
  const stripped = stripReasoning(text)
  if (!stripped) throw new Error('Empty summary')
  return stripped
}

/** Backoff between summarizer attempts, as in the prototype. */
const RETRY_DELAYS_MS = [2000, 4000, 8000]

const wait = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(timer)
      reject(signal.reason ?? new Error('Aborted'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })

/**
 * One summarizer call, retried on failure or an empty answer.
 *
 * A reasoning model spends the whole budget inside its thinking block and
 * hands back empty content, so every retry also widens the ceiling: the
 * answer is short either way (max_tokens is a ceiling, not a spend), but a
 * model that must think first needs room to finish before it can write.
 */
async function summarizeWithRetry(
  summarize: MemorySummarizer,
  prompt: string,
  maxTokens: number,
  signal: AbortSignal
): Promise<string> {
  let last: unknown
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
    if (signal.aborted) throw signal.reason ?? new Error('Aborted')
    try {
      const text = await summarize(prompt, {
        maxTokens: Math.min(4096, maxTokens * (attempt + 1)),
        signal,
      })
      if (text.trim()) return text
      last = new Error('Empty summary')
    } catch (error) {
      if ((error as Error)?.name === 'AbortError' || signal.aborted) throw error
      last = error
    }
    if (attempt < RETRY_DELAYS_MS.length) {
      console.warn(
        `Memory: summarizer attempt ${attempt + 1} failed, retrying.`,
        last
      )
      await wait(RETRY_DELAYS_MS[attempt], signal)
    }
  }
  throw last instanceof Error ? last : new Error('Summarizer failed')
}

const livePath = (threadId: string) => {
  const thread = useThreads.getState().threads[threadId]
  if (!thread) return undefined
  const messages = useMessages.getState().getMessages(threadId)
  const activeRootId = (thread.metadata as Record<string, unknown> | undefined)
    ?.activeRootId as string | undefined
  return { thread, path: computeActivePath(messages, activeRootId) }
}

/**
 * Run owed chapters, then owed canon passes, for one thread. Everything is
 * id-anchored: a branch switch mid-run only ends the run early (the next
 * trigger recomputes from coverage), and written chapters stay valid on
 * whatever lineage shares their ids.
 */
export async function runMemoryFold(
  opts: FoldRunOptions
): Promise<FoldRunResult> {
  const { threadId, manual = false } = opts
  const idle: FoldRunResult = {
    chaptersWritten: 0,
    canonPasses: 0,
    canonChanged: 0,
    stopped: false,
    didNothing: true,
  }
  if (inFlight.has(threadId)) return idle
  if (useThreads.getState().isLoadingThreads) return idle

  const started = livePath(threadId)
  if (!started) return idle
  const { thread } = started
  const global = memoryGlobalSettings()
  if (!manual && !effectiveAutoOn(thread.metadata, global.autoDefault)) {
    return idle
  }
  const character = resolveThreadCharacter(
    thread,
    useCharacters.getState().characters ?? []
  )
  // v1 prompts are roleplay-flavored; coding threads stay untouched.
  if (!isRoleplayCharacter(character)) return idle

  const sizes = effectiveSizes(thread.metadata, DEFAULT_MEMORY_SIZES)
  const controller = new AbortController()
  inFlight.set(threadId, controller)
  const result: FoldRunResult = { ...idle, didNothing: false }
  // Flipped back if the run turns out to have nothing owed at either level,
  // so "Write now" can say so instead of reporting silence.
  let touched = false
  try {
    const summarize = opts.summarize ?? summarizeWithChatModel
    const now = opts.now ?? Date.now
    const makeId = opts.makeId ?? ulid
    const pins = new Set(useThreadMemory.getState().records[threadId]?.pins ?? [])

    const names = () => {
      const persona = resolveThreadPersona(
        useThreads.getState().threads[threadId]
      )
      return {
        assistant: character?.name ?? 'Assistant',
        user: persona?.name || useGeneralSetting.getState().userName || 'User',
      }
    }

    // Whole chapters only: without a full batch owed there is nothing to
    // write. Manual runs skip this gate so owed canon still runs.
    const first = livePath(threadId)
    if (!first) return result
    const firstLite = first.path.map((m) => ({
      id: m.id,
      role: String(m.role),
      text: threadText(m),
    }))
    const firstScope = rootScope(
      useThreadMemory.getState().records[threadId],
      rootIdOf(firstLite)
    )
    const { cursor } = coverageCursor(firstLite, firstScope.chapters, pins)
    const batches = planChapterBatches(
      firstLite,
      cursor,
      pins,
      sizes.batch,
      sizes.keepRaw
    )
    if (batches.length === 0 && !manual) return result

    const runs = useMemoryRuns.getState()
    if (batches.length > 0) runs.begin(threadId, 'chapter')
    for (const batch of batches) {
      if (controller.signal.aborted) {
        result.stopped = true
        return result
      }
      const live = livePath(threadId)
      if (!live) return result
      const byId = new Map(live.path.map((m) => [m.id, m]))
      // A branch switch or delete mid-run ends the run; coverage recomputes
      // next trigger, and nothing written so far is invalid.
      if (!batch.every((id) => byId.has(id))) return result

      const labels = names()
      const messagesText = batch
        .map((id) => {
          const m = byId.get(id)!
          const speaker =
            String(m.role) === 'assistant'
              ? (labels.assistant ?? 'Assistant')
              : labels.user
          return `${speaker}: ${threadText(m)}`.trim()
        })
        .join('\n\n')

      const scopeNow = rootScope(
        useThreadMemory.getState().records[threadId],
        rootIdOf(live.path.map((m) => ({ id: m.id })))
      )
      const pathIds = new Set(live.path.map((m) => m.id))
      const prior = chaptersForPath(scopeNow.chapters, pathIds)
        .map((c) => c.text)
        .join('\n')
      const prompt = buildChapterPrompt(messagesText, {
        maxLines: maxSummaryLines(batch.length, sizes.responseLength),
        prior,
        template: global.chapterPrompt,
      })
      const raw = await summarizeWithRetry(
        summarize,
        prompt,
        sizes.responseLength,
        controller.signal
      )
      const text = normalizeSummary(raw) || EMPTY_CHAPTER_TEXT
      const at = now()
      const hashes: Record<string, string> = {}
      for (const id of batch) hashes[id] = hashText(threadText(byId.get(id)!))
      const rootId = rootIdOf(live.path.map((m) => ({ id: m.id })))
      useThreadMemory.getState().updateRoot(threadId, rootId, (scope) => ({
        ...scope,
        chapters: [
          ...scope.chapters,
          {
            id: makeId(),
            rootId,
            messageIds: [...batch],
            hashes,
            text,
            tokens: estimateTokens(text),
            created: at,
          },
        ],
      }))
      result.chaptersWritten++
      touched = true
    }

    // Canon waits until every chapter in the run exists: culling between
    // chapters deletes the ones later chapters are written against.
    for (;;) {
      if (controller.signal.aborted) {
        result.stopped = true
        return result
      }
      const live = livePath(threadId)
      if (!live) return result
      const rootId = rootIdOf(live.path.map((m) => ({ id: m.id })))
      const scope = rootScope(
        useThreadMemory.getState().records[threadId],
        rootId
      )
      const doomed = canonCandidates(scope.chapters, sizes.keepChapters, sizes.batch)
      if (doomed.length === 0) {
        result.didNothing = !touched
        return result
      }
      useMemoryRuns.getState().begin(threadId, 'canon')
      const body = doomed
        .map((c) => `### Chapter\n${c.text.trim()}`)
        .join('\n\n')
      const prompt = buildCanonPrompt(formatCanon(scope.canon), body, global.canonPrompt)
      const answer = await summarizeWithRetry(
        summarize,
        prompt,
        canonPassBudget(sizes.responseLength, doomed.length),
        controller.signal
      )
      const diff = parseCanonDiff(answer)
      if (!diff.adds.length && !diff.revisions.length) {
        // The model declined vs the parser dropped it are indistinguishable
        // from the toast alone — keep the raw answer in the log.
        console.warn(
          `Memory: canon pass recorded nothing (thread ${threadId}). Raw answer:\n${answer}`
        )
      }
      const at = now()
      const before: CanonEntry[] = scope.canon
        .filter((e) => e.src !== 'manual')
        .map((e) => ({ ...e }))
      const { canon, added, revised } = applyCanonDiff(scope.canon, diff, at)
      const doomedIds = new Set(doomed.map((c) => c.id))
      useThreadMemory.getState().updateRoot(threadId, rootId, (s) => ({
        canon,
        passes: [
          ...s.passes,
          {
            id: makeId(),
            at,
            rootId,
            chapterIds: [...doomedIds],
            before,
          },
        ],
        chapters: s.chapters.map((c) =>
          doomedIds.has(c.id) ? { ...c, culled: true } : c
        ),
      }))
      result.canonPasses++
      result.canonChanged += added + revised
      touched = true
    }
  } finally {
    if (inFlight.get(threadId) === controller) inFlight.delete(threadId)
    useMemoryRuns.getState().end(threadId)
  }
}

/**
 * Auto-trigger: fire-and-forget after a finished generation. A fold that
 * fails every retry has to say so — silently never writing a chapter looks
 * exactly like memory being switched off.
 */
export function maybeAutoFold(threadId: string): void {
  void runMemoryFold({ threadId }).catch((error) => {
    if ((error as Error)?.name === 'AbortError') return
    console.warn(`Memory: auto fold failed (thread ${threadId}):`, error)
    toast.error('Memory: writing the chapter failed', {
      description:
        (error as Error)?.message ??
        'The summarizer did not answer. It will try again next turn.',
    })
  })
}

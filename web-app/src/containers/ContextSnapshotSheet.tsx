import { useState } from 'react'
import { ThreadMessage } from '@janhq/core'
import { useNavigate } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  IconEye,
  IconChevronRight,
  IconBook,
  IconBrain,
  IconUserCircle,
} from '@tabler/icons-react'
import { cn } from '@/lib/utils'
import { useAppState } from '@/hooks/useAppState'
import { useThreads } from '@/hooks/useThreads'
import { useCharacters } from '@/hooks/useCharacters'
import { isRoleplayCharacter } from '@/lib/character-card'
import { requestMemoryStop, runMemoryFold } from '@/lib/memory-runner'
import { toast } from 'sonner'
import { useTokensCount } from '@/hooks/useTokensCount'
import { useActiveMessagePath } from '@/hooks/useActiveMessagePath'
import { useMemorySettings } from '@/hooks/useMemorySettings'
import { useThreadMemory } from '@/hooks/useThreadMemory'
import { usePendingMemory } from '@/stores/pending-memory'
import { estimateTokens } from '@/lib/context-manager'
import { getVersionInfo } from '@/lib/message-branching'
import {
  DEFAULT_MEMORY_SIZES,
  canonBudget,
  chapterHealth,
  chapterRange,
  effectiveSizes,
  foldLine,
  memoryStatus,
  canonCandidates,
  presetCost,
  sizePresetFor,
  MEMORY_SIZE_PRESETS,
  type MemoryAutoMode,
  type MemorySizes,
  type ThreadMemoryMeta,
} from '@/lib/thread-memory'
import {
  useContextSnapshotStore,
  calibrateTokens,
  selectBranchSnapshot,
  type ContextSnapshotSection,
} from '@/stores/context-snapshot-store'

interface ContextSnapshotSheetProps {
  threadId?: string
  triggerElement?: React.ReactNode
  /** Thread messages, for the measured token readout and the latest reply. */
  messages?: ThreadMessage[]
  /** Whether a turn is in flight, for the live readout. */
  streaming?: boolean
  /** Roleplay gating from the chatbox (covers the home screen, where no thread exists yet). */
  roleplay?: boolean
  /** Controlled mode: when provided, no trigger is rendered. */
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

export function ContextSnapshotSheet({
  threadId,
  triggerElement,
  messages,
  streaming,
  roleplay,
  open,
  onOpenChange,
}: ContextSnapshotSheetProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false)
  const isControlled = open !== undefined
  const isOpen = isControlled ? open : uncontrolledOpen
  const setOpen = (next: boolean) => {
    if (isControlled) onOpenChange?.(next)
    else setUncontrolledOpen(next)
  }

  const defaultTrigger = (
    <button
      type="button"
      className="outline-0 focus:outline-0 flex items-center transition-colors cursor-pointer text-muted-foreground hover:text-foreground"
      aria-label="Context sent to model"
    >
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="flex items-center">
            <IconEye size={16} />
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <p>Context sent to model</p>
        </TooltipContent>
      </Tooltip>
    </button>
  )

  return (
    <Sheet open={isOpen} onOpenChange={setOpen}>
      {!isControlled && (
        <SheetTrigger asChild>{triggerElement || defaultTrigger}</SheetTrigger>
      )}
      <SheetContent className="sm:max-w-lg flex flex-col">
        <ContextSnapshotBody
          threadId={threadId}
          messages={messages}
          streaming={streaming}
          roleplay={roleplay}
        />
      </SheetContent>
    </Sheet>
  )
}

const formatExact = (n: number) => Math.round(n).toLocaleString()

const threadMessageText = (message: ThreadMessage): string =>
  (message.content ?? [])
    .filter((c) => c.type === 'text')
    .map((c) => c.text?.value ?? '')
    .join('\n')
    .trim()

/**
 * Split out so its hooks (model-props polling, live stream stats) only run
 * while the sheet is open — Radix unmounts sheet content when closed.
 */
function ContextSnapshotBody({
  threadId,
  messages,
  streaming,
  roleplay,
}: {
  threadId?: string
  messages?: ThreadMessage[]
  streaming?: boolean
  roleplay?: boolean
}) {
  const capturedForThread = useContextSnapshotStore((s) =>
    threadId ? s.snapshots[threadId] : undefined
  )
  const calibration = useContextSnapshotStore((s) =>
    threadId ? (s.calibration[threadId] ?? 1) : 1
  )
  const liveStats = useAppState((s) =>
    threadId ? s.liveTokenStatsByThread[threadId] : undefined
  )
  const promptProgress = useAppState((s) =>
    threadId ? s.promptProgresses[threadId] : undefined
  )
  const tokenData = useTokensCount(messages ?? [])
  // Branch siblings all live in the message store; only the active reply is
  // part of the conversation the model sees.
  const activePath = useActiveMessagePath(messages ?? [])
  const thread = useThreads((s) => (threadId ? s.threads[threadId] : undefined))
  const updateThread = useThreads((s) => s.updateThread)
  const characters = useCharacters((s) => s.characters)
  // v1 fold prompts are roleplay-flavored, so coding threads get status
  // only — but settings must not vanish on zero-message chats or while the
  // library is still loading. Hide only when definitively non-roleplay
  // (the model-only sentinel or a resolved assistant-kind character); the
  // runner keeps its own strict gate where model calls are at stake.
  const embeddedId = thread?.assistants?.[0]?.id
  const libChar = (characters ?? []).find((c) => c.id === embeddedId)
  const knownNonRp =
    embeddedId === 'model-only' ||
    (libChar !== undefined && !isRoleplayCharacter(libChar))
  const memoryRoleplay = roleplay ?? !knownNonRp
  const [foldRunning, setFoldRunning] = useState(false)
  const navigate = useNavigate()
  const [view, setView] = useState<'context' | 'prompt'>('context')
  const memoryRecord = useThreadMemory((s) =>
    threadId ? s.records[threadId] : undefined
  )
  const globalSizes = DEFAULT_MEMORY_SIZES
  const memoryAutoDefault = useMemorySettings((s) => s.autoDefault)
  const pendingMemory = usePendingMemory((s) => s.pending)
  const patchPendingMemory = usePendingMemory((s) => s.patch)
  // Per-chat settings live on the thread; on the home screen they stage in
  // the pending store and ChatInput copies them onto the thread it creates.
  const metaHost = thread
    ? thread.metadata
    : { memory: pendingMemory }
  // Sizes are per chat: the override wins outright, else Medium.
  // The settings page only holds the auto default and the prompts.
  const sizes: MemorySizes = effectiveSizes(metaHost, globalSizes)
  const hasSizeOverride =
    (metaHost as { memory?: ThreadMemoryMeta } | undefined)?.memory?.sizes !==
    undefined
  const sizePreset = sizePresetFor(sizes)

  // Display-only ordinals: #N is the message's position on the active path,
  // resolved live. Zero-based so the greeting is #0, as in SillyTavern. Ids
  // stay canonical; a branch switch renumbers honestly.
  const ordinals = new Map(activePath.map((m, i) => [m.id, i] as const))
  const byId = new Map((messages ?? []).map((m) => [m.id, m]))

  // Memory standing on the branch being viewed. Chapters count as covering
  // only when every one of their ids is on this path; owed derives from
  // that coverage, so there is no per-branch watermark to go stale.
  const memory = (() => {
    const status = memoryStatus({
      path: activePath.map((m) => ({
        id: m.id,
        role: String(m.role),
        text: threadMessageText(m),
      })),
      record: memoryRecord,
      threadMetadata: metaHost,
      autoDefault: memoryAutoDefault,
      globalSizes,
    })
    return {
      ...status,
      canonTokens: status.scope.canon.reduce(
        (acc, e) => acc + estimateTokens(e.text) + 4,
        0
      ),
      hasAny: !!memoryRecord && status.hasAny,
    }
  })()
  const canonTokenBudget = canonBudget(sizes)

  const patchMemory = (patch: Partial<ThreadMemoryMeta>) => {
    if (thread) {
      const prev = (thread.metadata?.memory ?? {}) as ThreadMemoryMeta
      updateThread(thread.id, {
        metadata: { ...thread.metadata, memory: { ...prev, ...patch } },
      })
    } else {
      patchPendingMemory(patch)
    }
  }
  const setMemoryAuto = (auto: MemoryAutoMode) =>
    patchMemory({ auto: auto === 'default' ? undefined : auto })
  const setMemorySize = (key: keyof MemorySizes, n: number) =>
    patchMemory({ sizes: { ...sizes, [key]: n } })
  const applyMemoryPreset = (name: string) =>
    patchMemory({ sizes: { ...MEMORY_SIZE_PRESETS[name] } })
  const resetMemorySizes = () => {
    if (thread) {
      const prev = { ...(thread.metadata?.memory ?? {}) } as ThreadMemoryMeta
      delete prev.sizes
      updateThread(thread.id, {
        metadata: { ...thread.metadata, memory: prev },
      })
    } else {
      patchPendingMemory({ sizes: undefined })
    }
  }
  // Whole batches only, so under one batch Write now would do nothing —
  // except when chapters are already owed to canon.
  const canWriteNow =
    memory.ready ||
    canonCandidates(memory.scope.chapters, sizes.keepChapters, sizes.batch)
      .length > 0

  const handleWriteNow = async () => {
    if (!thread || foldRunning) return
    setFoldRunning(true)
    try {
      const r = await runMemoryFold({ threadId: thread.id, manual: true })
      if (r.didNothing) {
        toast.info('Nothing owed right now.')
      } else if (r.stopped) {
        toast.info('Stopped. Whatever was written is kept.')
      } else {
        if (r.chaptersWritten > 0) {
          toast.info(
            `Wrote ${r.chaptersWritten} chapter${r.chaptersWritten === 1 ? '' : 's'}.`
          )
        }
        if (r.canonPasses > 0) {
          toast.info(
            r.canonChanged > 0
              ? `Canon updated (${r.canonChanged} record change${r.canonChanged === 1 ? '' : 's'}).`
              : 'Folded chapters into canon, nothing new to record.'
          )
        }
      }
    } catch (error) {
      if ((error as Error)?.name === 'AbortError') {
        toast.info('Stopped. Whatever was written is kept.')
      } else {
        toast.error('Folding failed, will retry on the next message.')
        console.error('Memory fold failed', error)
      }
    } finally {
      setFoldRunning(false)
    }
  }

  const memoryNumberField = (    value: number,
    onChange: (n: number) => void,
    opts: { min: number; max: number }
  ) => (
    <Input
      type="number"
      min={opts.min}
      max={opts.max}
      value={value}
      className="w-20 h-8"
      onChange={(e) => {
        const n = Number(e.target.value)
        if (Number.isNaN(n)) return
        onChange(Math.min(opts.max, Math.max(opts.min, Math.round(n))))
      }}
    />
  )

  // A capture records one request, so the thread keeps one per branch it has
  // sent from: switching replies brings that branch's own breakdown back, and
  // a branch never sent from shows nothing rather than another branch's prompt.
  const pathIds = new Set(activePath.map((m) => m.id))
  const snapshot = selectBranchSnapshot(capturedForThread, pathIds)
  const snapshotIsForAnotherBranch = !snapshot && !!capturedForThread?.length

  const isStreaming = !!streaming
  const measuredTotal = tokenData.tokenCount
  const measuredCompletion = tokenData.outputTokens ?? 0
  const maxTokens = tokenData.maxTokens ?? snapshot?.maxContextTokens ?? null

  // Sections are a character heuristic. Scale them so they add up to the real
  // prompt count: prefer a measurement of this very snapshot (the in-flight
  // stream, or the usage recorded when it finished), and fall back to the
  // thread's last known ratio for a snapshot no turn has measured yet.
  const measuredForSnapshot =
    snapshot?.actualPromptTokens ?? liveStats?.promptTokens
  const scale =
    measuredForSnapshot && snapshot && snapshot.totalTokens > 0
      ? measuredForSnapshot / snapshot.totalTokens
      : calibration
  const scaled = (tokens: number) => calibrateTokens(tokens, scale)
  const promptTotal = snapshot ? scaled(snapshot.totalTokens) : 0
  const isExact = !!measuredForSnapshot

  // A snapshot is captured before the request, so it never contains the reply
  // it produced. Carry that reply as its own row until the next turn folds it
  // into a fresh snapshot — otherwise the panel goes blank on the model's own
  // output, which is the half worth watching while it streams.
  const lastAssistantText = (() => {
    for (let i = activePath.length - 1; i >= 0; i--) {
      if (String(activePath[i].role) === 'assistant')
        return threadMessageText(activePath[i])
    }
    return ''
  })()
  // Once a later turn captures a snapshot, that reply is one of the prompt's
  // own assistant sections (not necessarily the last one, since the user turn
  // follows it) and must not be counted a second time.
  const replyAlreadyInPrompt =
    !!lastAssistantText &&
    !!snapshot?.sections.some(
      (s) => s.role === 'assistant' && s.content.trim() === lastAssistantText
    )
  const replyText = replyAlreadyInPrompt ? '' : lastAssistantText
  const showReply = isStreaming || !!replyText
  const replyTokens = !showReply
    ? 0
    : measuredCompletion > 0
      ? measuredCompletion
      : scaled(estimateTokens(replyText))
  const assistantLabel =
    snapshot?.assistantLabel ??
    snapshot?.sections.find((s) => s.role === 'assistant')?.label ??
    'Reply'
  const replySection: ContextSnapshotSection | null = showReply
    ? {
        id: 'live-reply',
        kind: 'message',
        label: isStreaming ? `${assistantLabel} · replying` : assistantLabel,
        role: 'assistant',
        tokens: replyTokens,
        content: replyText,
      }
    : null

  // Without a snapshot for this branch there is no breakdown to sum, so the
  // header falls back to the usage the branch's own last turn reported.
  const totalTokens = snapshot ? promptTotal + replyTokens : measuredTotal
  // The fallback total is server usage, so it is exact by definition.
  const totalIsMeasured = snapshot ? isExact : true
  const usagePct =
    maxTokens && maxTokens > 0 && totalTokens > 0
      ? Math.min(100, (totalTokens / maxTokens) * 100)
      : null

  // Budget view rolls messages up per role (one row for all assistant turns,
  // one for all user turns); non-message sections keep their own rows. The
  // latest reply stays separate so it can be watched counting up.
  const budgetRows = snapshot
    ? Object.values(
        snapshot.sections.reduce<Record<string, ContextSnapshotSection>>(
          (acc, s) => {
            const row = { ...s, tokens: scaled(s.tokens) }
            if (s.kind !== 'message') {
              acc[s.id] = row
            } else {
              const key = `msg-${s.role ?? 'unknown'}`
              if (acc[key]) {
                acc[key].tokens += row.tokens
              } else {
                acc[key] = { ...row, id: key }
              }
            }
            return acc
          },
          {}
        )
      ).sort((a, b) => b.tokens - a.tokens)
    : []
  const rows = replySection ? [...budgetRows, replySection] : budgetRows

  const promptRows = (
    <>
      {snapshot?.sections.map((s) => {
        if (s.kind !== 'message') {
          return (
            <SystemSection
              key={s.id}
              section={s}
              tokens={scaled(s.tokens)}
              // The opening message is a real message travelling folded into
              // the system string; number it so the send order reads 0,1,2...
              ordinal={s.kind === 'initial-message' ? 0 : undefined}
            />
          )
        }
        const ordinal = s.sourceId ? ordinals.get(s.sourceId) : undefined
        const badges: string[] = []
        if (s.sourceId && memory.foldedIds.has(s.sourceId))
          badges.push('in chapter')
        if (s.sourceId && memory.pins.has(s.sourceId)) badges.push('pinned')
        let version: string | undefined
        const source = s.sourceId ? byId.get(s.sourceId) : undefined
        if (source) {
          const v = getVersionInfo(messages ?? [], source)
          if (v.count > 1) version = `v${v.index}/${v.count}`
        }
        return (
          <MessageSection
            key={s.id}
            section={s}
            tokens={scaled(s.tokens)}
            ordinal={ordinal}
            version={version}
            badges={badges}
          />
        )
      })}
      {replySection && (
        <MessageSection
          section={replySection}
          tokens={replySection.tokens}
          ordinal={(() => {
            for (let i = activePath.length - 1; i >= 0; i--) {
              if (String(activePath[i].role) === 'assistant') return i
            }
            return undefined
          })()}
        />
      )}
    </>
  )

  return (
    <>
      <SheetHeader>
        <SheetTitle>Context sent to model</SheetTitle>
        <SheetDescription>
          {view === 'prompt'
            ? 'The exact messages the model receives, in send order.'
            : isStreaming
              ? 'Streaming now'
              : snapshot
                ? `Last captured ${new Date(snapshot.createdAt).toLocaleTimeString()}`
                : snapshotIsForAnotherBranch
                  ? 'Captured on a different branch.'
                  : 'Nothing captured yet — send a message first.'}
        </SheetDescription>
      </SheetHeader>

      <div className="px-4 pt-2 flex gap-1.5">
        {(['context', 'prompt'] as const).map((v) => (
          <Button
            key={v}
            size="xs"
            variant={view === v ? 'default' : 'outline'}
            className="flex-1"
            onClick={() => setView(v)}
          >
            {v === 'context' ? 'Context' : 'Prompt'}
          </Button>
        ))}
      </div>

      {view === 'prompt' ? (
        <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-4">
          {snapshot ? (
            <div className="space-y-2">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Prompt in send order
              </p>
              {promptRows}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground px-0 pt-2">
              {snapshotIsForAnotherBranch
                ? 'No breakdown for this branch: the last capture belongs to another reply. Send a message to capture this one.'
                : 'No breakdown yet. Send a message to capture what gets sent.'}
            </p>
          )}
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-4">
        {(snapshot || totalTokens > 0 || snapshotIsForAnotherBranch) && (
          <div className="rounded-md border p-3 space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="text-sm font-medium">
                {usagePct !== null
                  ? `${usagePct.toFixed(1)}% of context`
                  : 'Context'}
              </span>
              <span className="text-sm text-muted-foreground tabular-nums font-mono">
                {totalIsMeasured ? '' : '~'}
                {formatExact(totalTokens)}
                {maxTokens ? ` / ${formatExact(maxTokens)}` : ''}
              </span>
            </div>
            {usagePct !== null && (
              <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-500 ease-out',
                    usagePct > 90 ? 'bg-destructive' : 'bg-primary'
                  )}
                  style={{ width: `${usagePct}%` }}
                />
              </div>
            )}
            {snapshot ? (
              <div className="space-y-1 pt-1">
                {rows.map((s) => (
                  <BudgetRow key={s.id} section={s} total={totalTokens} />
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground pt-1">
                {snapshotIsForAnotherBranch
                  ? 'No breakdown for this branch: the last capture belongs to another reply. Send a message to capture this one.'
                  : 'No breakdown yet. Send a message to capture what gets sent.'}
              </p>
            )}
            {isStreaming &&
              promptProgress &&
              promptProgress.total > 0 &&
              promptProgress.processed < promptProgress.total && (
                <div className="pt-1 space-y-1">
                  <div className="flex justify-between text-[11px] text-muted-foreground">
                    <span>Processing prompt</span>
                    <span className="tabular-nums">
                      {formatExact(promptProgress.processed)} /{' '}
                      {formatExact(promptProgress.total)}
                      {promptProgress.cache > 0
                        ? ` · ${formatExact(promptProgress.cache)} cached`
                        : ''}
                    </span>
                  </div>
                  <div className="h-1 rounded-full bg-secondary overflow-hidden">
                    <div
                      className="h-full bg-primary/60 rounded-full"
                      style={{
                        width: `${Math.min(100, (promptProgress.processed / promptProgress.total) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              )}
            <div className="flex flex-wrap gap-x-3 text-[11px] text-muted-foreground pt-1">
              {maxTokens ? (
                <span className="tabular-nums">
                  {formatExact(Math.max(0, maxTokens - totalTokens))} left
                </span>
              ) : null}
              {isStreaming && liveStats?.tokensPerSecond ? (
                <span className="tabular-nums">
                  {liveStats.tokensPerSecond.toFixed(1)} tok/s
                </span>
              ) : null}
              {isStreaming && liveStats?.promptPerSecond ? (
                <span className="tabular-nums">
                  reading {liveStats.promptPerSecond.toFixed(1)} tok/s
                </span>
              ) : null}
              {snapshot ? (
                <span>
                  {isExact
                    ? 'total measured by the tokenizer, rows split by text length'
                    : 'estimated from text length'}
                </span>
              ) : measuredTotal > 0 ? (
                <span>measured on this branch&apos;s last turn</span>
              ) : null}
            </div>
          </div>
        )}

        {memory.hasAny && (
          <div className="rounded-md border p-3 space-y-2">
            <div className="flex items-baseline justify-between">
              <span className="text-sm font-medium flex items-center gap-1.5">
                <IconBrain size={14} className="text-muted-foreground" />
                Memory
              </span>
              <span className="text-sm text-muted-foreground tabular-nums font-mono">
                {memory.liveChapters > 0 &&
                  `${memory.liveChapters} chapter${memory.liveChapters === 1 ? '' : 's'}`}
                {memory.liveChapters > 0 && memory.scope.canon.length > 0 && ' · '}
                {memory.scope.canon.length > 0 &&
                  `${memory.scope.canon.length} canon`}
              </span>
            </div>
            <p className="text-[11px] text-muted-foreground">
              {foldLine(memory)}
              {' · '}
              {`Auto ${memory.auto === 'default' ? `default (${memory.autoOn ? 'on' : 'off'})` : memory.auto}`}
              {memory.otherRoots > 0 &&
                ` · memory on ${memory.otherRoots} other branch${memory.otherRoots === 1 ? '' : 'es'}`}
            </p>
            {memory.chapters.length > 0 && (
              <div className="space-y-1 pt-1">
                {memory.chapters.map((c) => {
                  const range = chapterRange(c, activePath)
                  const health = chapterHealth(c, (id) => {
                    const m = byId.get(id)
                    return m ? threadMessageText(m) : undefined
                  })
                  const badges: string[] = []
                  // A culled chapter still covers its messages but has left
                  // the prompt, so the list must not read as what is sent.
                  if (c.culled) badges.push('in canon')
                  if (health.edited.length > 0) badges.push('edited since')
                  if (health.missing.length > 0) badges.push('moved off path')
                  return (
                    <MessageSection
                      key={c.id}
                      section={{
                        id: c.id,
                        kind: 'message',
                        label: range
                          ? `Chapter #${range.from}–#${range.to}`
                          : 'Chapter (not on this path)',
                        tokens: c.tokens,
                        content: c.text,
                      }}
                      tokens={c.tokens}
                      badges={badges}
                    />
                  )
                })}
              </div>
            )}
            {memory.scope.canon.length > 0 && (
              <CanonSection
                entries={memory.scope.canon}
                tokens={memory.canonTokens}
                overBudget={memory.canonTokens > canonTokenBudget}
              />
            )}
            <p className="text-[11px] text-muted-foreground">
              Chapters and canon travel at the end of the system prompt;
              folded messages leave it.
            </p>
          </div>
        )}

          {/* Compaction settings are always available — SillyTavern-style:
              the runner (not the UI) decides whether a chat folds, so a
              coding chat or a fresh composer never loses the controls. */}
            <div className="rounded-md border p-3 space-y-2">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-medium">
                  Compaction in this chat
                </span>
                {thread && memoryRoleplay && (
                  <Button
                    size="xs"
                    variant={foldRunning ? 'destructive' : 'outline'}
                    // Whatever started a run is that run's Stop.
                    disabled={!foldRunning && !canWriteNow}
                    title={
                      foldRunning
                        ? 'Stop. Whatever is already written is kept.'
                        : canWriteNow
                          ? `Write ${memory.outlook.blocks || Math.floor(memory.waiting / sizes.batch)} chapter(s) now.`
                          : `A chapter takes ${sizes.batch} messages and ${memory.waiting} are waiting.`
                    }
                    onClick={
                      foldRunning
                        ? () => thread && requestMemoryStop(thread.id)
                        : handleWriteNow
                    }
                  >
                    {foldRunning ? 'Stop' : 'Write now'}
                  </Button>
                )}
              </div>
              {!memoryRoleplay && (
                <p className="text-[11px] text-muted-foreground">
                  Folding runs in roleplay chats — these settings wait until
                  one is.
                </p>
              )}
              {!thread && (
                <p className="text-[11px] text-muted-foreground">
                  Saved for the chat you start next.
                </p>
              )}
              <p className="text-[11px] text-muted-foreground">
                {foldLine(memory)}
                {' · '}
                {`~${presetCost(sizes).toLocaleString()} tokens worst case`}
                {' · '}
                <span className="capitalize">
                  {sizePreset}
                  {hasSizeOverride ? ' · this chat' : ' · default'}
                </span>
              </p>
              <div className="space-y-2 pt-1">
                <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                  Auto-fold
                </div>
                <div className="flex gap-1.5">
                  {(['default', 'on', 'off'] as const).map((mode) => (
                    <Button
                      key={mode}
                      size="xs"
                      variant={memory.auto === mode ? 'default' : 'outline'}
                      className="capitalize flex-1"
                      title={
                        mode === 'default'
                          ? `Follow the global setting (currently ${memory.autoOn ? 'on' : 'off'})`
                          : `Always ${mode} in this chat`
                      }
                      onClick={() => setMemoryAuto(mode)}
                    >
                      {mode === 'default'
                        ? `Default (${memory.autoOn ? 'on' : 'off'})`
                        : mode}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="space-y-2 pt-1">
                <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                  Size preset
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {(
                    Object.keys(MEMORY_SIZE_PRESETS) as Array<
                      keyof typeof MEMORY_SIZE_PRESETS
                    >
                  ).map((name) => (
                    <Button
                      key={name}
                      size="xs"
                      variant={sizePreset === name ? 'default' : 'outline'}
                      className="capitalize"
                      title={`~${presetCost(MEMORY_SIZE_PRESETS[name]).toLocaleString()} tokens worst case`}
                      onClick={() => applyMemoryPreset(name)}
                    >
                      {name}
                    </Button>
                  ))}
                  {sizePreset === 'custom' && (
                    <span className="text-xs text-muted-foreground self-center ml-1">
                      Custom
                    </span>
                  )}
                  {hasSizeOverride && (
                    <Button
                      size="xs"
                      variant="ghost"
                      className="ml-auto"
                      onClick={resetMemorySizes}
                    >
                      Reset to Medium
                    </Button>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 pt-1">
                <label className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  Keep raw
                  {memoryNumberField(sizes.keepRaw, (n) => setMemorySize('keepRaw', n), { min: 1, max: 500 })}
                </label>
                <label className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  Keep chapters
                  {memoryNumberField(sizes.keepChapters, (n) => setMemorySize('keepChapters', n), { min: 1, max: 500 })}
                </label>
                <label className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  Batch
                  {memoryNumberField(sizes.batch, (n) => setMemorySize('batch', n), { min: 1, max: 200 })}
                </label>
                <label className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  Summary length
                  {memoryNumberField(sizes.responseLength, (n) => setMemorySize('responseLength', n), { min: 50, max: 2000 })}
                </label>
              </div>
              <div className="pt-1">
                <button
                  type="button"
                  className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
                  onClick={() => navigate({ to: route.settings.memory })}
                >
                  Global default & fold prompts →
                </button>
              </div>
            </div>
        </div>
      )}
    </>
  )
}

function BudgetRow({
  section,
  total,
}: {
  section: ContextSnapshotSection
  total: number
}) {
  const share = Math.round(
    (Math.max(0, section.tokens) / Math.max(1, total)) * 100
  )
  return (
    <div className="text-xs">
      <div className="flex justify-between mb-0.5">
        <span className="truncate">{section.label}</span>
        <span className="shrink-0 text-muted-foreground tabular-nums">
          {section.tokens} tok · {share}%
        </span>
      </div>
      <div className="h-1 rounded-full bg-secondary overflow-hidden">
        <div
          className="h-full bg-primary/60 rounded-full"
          style={{ width: `${share}%` }}
        />
      </div>
    </div>
  )
}

function SystemSection({
  section,
  tokens,
  ordinal,
}: {
  section: ContextSnapshotSection
  tokens: number
  ordinal?: number
}) {
  const [open, setOpen] = useState(false)
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="rounded-md border">
        <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-secondary/50">
          <IconChevronRight
            size={14}
            className={cn('transition-transform', open && 'rotate-90')}
          />
          {section.kind === 'persona' && (
            <IconUserCircle
              size={14}
              className="text-muted-foreground shrink-0"
            />
          )}
          {section.kind === 'world-info' && (
            <IconBook size={14} className="text-muted-foreground shrink-0" />
          )}
          {ordinal !== undefined && (
            <span className="text-muted-foreground tabular-nums text-xs">
              #{ordinal}
            </span>
          )}
          <span className="font-medium">{section.label}</span>
          <span className="ml-auto text-muted-foreground tabular-nums text-xs">
            {tokens} tok
          </span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <pre className="whitespace-pre-wrap break-words px-3 pb-3 text-xs text-muted-foreground max-h-72 overflow-y-auto">
            {section.content}
          </pre>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
}

const ROLE_STYLES: Record<string, string> = {
  user: 'border-blue-500/40',
  assistant: 'border-purple-500/40',
  tool: 'border-orange-500/40',
  system: 'border-red-500/40',
}

function MessageSection({
  section,
  tokens,
  ordinal,
  version,
  badges = [],
}: {
  section: ContextSnapshotSection
  tokens: number
  /** 1-based position on the active path; absent when unresolvable. */
  ordinal?: number
  /** Sibling badge, e.g. v2/3; absent for single-version turns. */
  version?: string
  badges?: string[]
}) {
  const [open, setOpen] = useState(false)
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div
        className={cn(
          'rounded-md border border-l-2',
          ROLE_STYLES[section.role ?? '']
        )}
      >
        <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-secondary/50">
          <IconChevronRight
            size={14}
            className={cn('transition-transform', open && 'rotate-90')}
          />
          {ordinal !== undefined && (
            <span className="tabular-nums text-muted-foreground">
              #{ordinal}
            </span>
          )}
          <span className="font-medium">{section.label}</span>
          {version && (
            <span className="text-[11px] text-muted-foreground tabular-nums">
              {version}
            </span>
          )}
          {badges.map((b) => (
            <span key={b} className="text-[11px] text-muted-foreground">
              {b}
            </span>
          ))}
          <span className="ml-auto text-muted-foreground tabular-nums text-xs">
            {tokens} tok
          </span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <pre className="whitespace-pre-wrap break-words px-3 pb-3 text-xs text-muted-foreground max-h-72 overflow-y-auto">
            {section.content || '(streaming…)'}
          </pre>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
}

function CanonSection({
  entries,
  tokens,
  overBudget,
}: {
  entries: Array<{
    id: string
    cat: string
    text: string
    src: 'fold' | 'manual'
  }>
  tokens: number
  overBudget: boolean
}) {
  const [open, setOpen] = useState(false)
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="rounded-md border">
        <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-secondary/50">
          <IconChevronRight
            size={14}
            className={cn('transition-transform', open && 'rotate-90')}
          />
          <span className="font-medium">Canon</span>
          <span className="ml-auto text-muted-foreground tabular-nums text-xs">
            ~{tokens.toLocaleString()} tok{overBudget ? ' · over budget' : ''}
          </span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="px-3 pb-3 space-y-1 max-h-72 overflow-y-auto">
            {entries.map((e) => (
              <p
                key={e.id}
                className="text-xs text-muted-foreground break-words"
              >
                <span className="tabular-nums font-mono">[{e.id}]</span>{' '}
                {e.cat} | {e.text}
                {e.src === 'manual' && (
                  <span className="text-red-400"> · hand-written</span>
                )}
              </p>
            ))}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
}

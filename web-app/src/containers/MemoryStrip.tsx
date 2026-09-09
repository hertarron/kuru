import { useMemo } from 'react'
import type { ThreadMessage } from '@janhq/core'
import { IconBrain } from '@tabler/icons-react'
import { cn } from '@/lib/utils'
import { useActiveMessagePath } from '@/hooks/useActiveMessagePath'
import { useThreadMemory } from '@/hooks/useThreadMemory'
import { useMemoryRuns } from '@/hooks/useMemoryRuns'
import { useMemorySettings } from '@/hooks/useMemorySettings'
import { foldLine, memoryStatus } from '@/lib/thread-memory'

const textOf = (m: ThreadMessage): string =>
  (m.content ?? [])
    .filter((c) => c?.type === 'text')
    .map((c) => c.text?.value ?? '')
    .join('\n')

/**
 * One-line fold outlook above the send row: what memory holds and when the
 * next chapter lands. Always present in roleplay chats (appearing and
 * disappearing reflows the log every few turns); tapping opens the context
 * view, where the per-chat settings live.
 */
export function MemoryStrip({
  thread,
  messages,
  onOpen,
}: {
  thread?: Thread
  messages: ThreadMessage[]
  onOpen: () => void
}) {
  const record = useThreadMemory((s) =>
    thread ? s.records[thread.id] : undefined
  )
  const phase = useMemoryRuns((s) => (thread ? s.active[thread.id] : undefined))
  const autoDefault = useMemorySettings((s) => s.autoDefault)
  const activePath = useActiveMessagePath(messages)

  const status = useMemo(
    () =>
      memoryStatus({
        path: activePath.map((m) => ({
          id: m.id,
          role: String(m.role),
          text: textOf(m),
        })),
        record,
        threadMetadata: thread?.metadata,
        autoDefault,
      }),
    [activePath, record, thread?.metadata, autoDefault]
  )

  if (!thread) return null

  const armed = status.autoOn && status.outlook.turns <= 1
  const label =
    phase === 'canon'
      ? 'Updating canon…'
      : phase === 'chapter'
        ? 'Writing chapter…'
        : foldLine(status)
  const counts = [
    status.liveChapters > 0
      ? `${status.liveChapters} chapter${status.liveChapters === 1 ? '' : 's'}`
      : null,
    status.scope.canon.length > 0 ? `${status.scope.canon.length} canon` : null,
  ]
    .filter(Boolean)
    .join(' · ')

  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label="Open memory"
      className="flex w-full min-w-0 items-center gap-1.5 px-1 pb-1 text-left cursor-pointer"
    >
      <IconBrain size={12} className="shrink-0 text-muted-foreground" />
      <span
        className={cn(
          'text-[11px] tabular-nums truncate',
          armed || phase !== undefined
            ? 'text-foreground'
            : 'text-muted-foreground',
          phase !== undefined && 'animate-pulse'
        )}
      >
        {label}
      </span>
      {counts && (
        <span className="text-[11px] tabular-nums text-muted-foreground/70 truncate">
          · {counts}
        </span>
      )}
    </button>
  )
}

export default MemoryStrip

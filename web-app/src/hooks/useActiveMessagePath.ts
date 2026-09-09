import { useMemo } from 'react'
import { ThreadMessage } from '@janhq/core'
import { computeActivePath } from '@/lib/message-branching'
import { useThreads } from './useThreads'

/**
 * The messages actually in play for a thread.
 *
 * The message store holds every branch sibling, so a regenerated reply leaves
 * two assistant messages under the same parent. Anything reasoning about "the
 * conversation" — what was sent, what it cost — has to walk the active path
 * instead of the raw list, and has to re-walk it when the user flips branches.
 */
export const useActiveMessagePath = (
  messages: ThreadMessage[] = []
): ThreadMessage[] => {
  const threadId = messages[0]?.thread_id
  const activeRootId = useThreads((s) =>
    threadId
      ? ((s.threads[threadId]?.metadata as Record<string, unknown> | undefined)
          ?.activeRootId as string | undefined)
      : undefined
  )
  return useMemo(
    () => computeActivePath(messages, activeRootId),
    [messages, activeRootId]
  )
}

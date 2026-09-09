import { create } from 'zustand'
import { estimateTokens } from '@/lib/context-manager'

export type ContextSectionKind =
  | 'system-prompt'
  | 'world-info'
  | 'persona'
  | 'canon'
  | 'chapters'
  | 'initial-message'
  | 'files-instruction'
  | 'web-search-instruction'
  | 'message'

export interface ContextSnapshotSection {
  id: string
  kind: ContextSectionKind
  label: string
  role?: 'user' | 'assistant' | 'system' | 'tool'
  tokens: number
  content: string
  /**
   * Thread-message id this row was built from, when known. Lets the
   * visualizer number the row (#N on the active path) and show version
   * badges. Absent for system rows, synthetic rows (compact summaries),
   * and captures taken before ids were recorded.
   */
  sourceId?: string
}

export interface ContextSnapshot {
  threadId: string
  createdAt: number
  sections: ContextSnapshotSection[]
  /** Sum of the per-section character heuristic, before calibration. */
  totalTokens: number
  maxContextTokens: number | null
  /** Display name for assistant turns (the character), when one is attached. */
  assistantLabel?: string
  /**
   * Ids of the messages this prompt was built from, so a viewer can tell
   * whether the snapshot still describes the branch being looked at.
   */
  promptMessageIds?: string[]
  /** Server-reported prompt tokens for this exact snapshot, once the turn finishes. */
  actualPromptTokens?: number
}

type State = {
  /** Per thread, the captures still held, oldest first. */
  snapshots: Record<string, ContextSnapshot[]>
  /** Per-thread ratio of real tokenizer tokens to heuristic tokens. */
  calibration: Record<string, number>
  capture: (snapshot: ContextSnapshot) => void
  recordActualPromptTokens: (threadId: string, actual: number) => void
}

// A ratio outside this range means the snapshot and the reported usage describe
// different requests (retry, aborted turn, provider quirk); ignore it rather
// than poisoning every future section count.
const MIN_CALIBRATION = 0.5
const MAX_CALIBRATION = 2.5

// A thread keeps one capture per branch it has sent from, so switching back to
// an earlier reply brings its own breakdown back. Bounded because every entry
// holds the full prompt text.
const MAX_SNAPSHOTS_PER_THREAD = 20

const branchKey = (snapshot: ContextSnapshot) =>
  snapshot.promptMessageIds?.join('|') ?? ''

export const useContextSnapshotStore = create<State>((set) => ({
  snapshots: {},
  calibration: {},
  capture: (snapshot) =>
    set((s) => {
      const key = branchKey(snapshot)
      const kept = (s.snapshots[snapshot.threadId] ?? []).filter(
        (existing) => branchKey(existing) !== key
      )
      return {
        snapshots: {
          ...s.snapshots,
          [snapshot.threadId]: [...kept, snapshot].slice(
            -MAX_SNAPSHOTS_PER_THREAD
          ),
        },
      }
    }),
  recordActualPromptTokens: (threadId, actual) =>
    set((s) => {
      const list = s.snapshots[threadId]
      const snapshot = list?.[list.length - 1]
      if (!snapshot || !(actual > 0)) return s
      const ratio = actual / Math.max(1, snapshot.totalTokens)
      const usable = ratio >= MIN_CALIBRATION && ratio <= MAX_CALIBRATION
      return {
        snapshots: {
          ...s.snapshots,
          [threadId]: [
            ...list.slice(0, -1),
            { ...snapshot, actualPromptTokens: actual },
          ],
        },
        calibration: usable
          ? { ...s.calibration, [threadId]: ratio }
          : s.calibration,
      }
    }),
}))

/**
 * The newest capture whose prompt is entirely on the given path — the one that
 * describes the branch being viewed. Captures taken before ids were recorded
 * carry no prompt ids and are accepted as a last resort.
 */
export function selectBranchSnapshot(
  snapshots: ContextSnapshot[] | undefined,
  pathIds: Set<string>
): ContextSnapshot | undefined {
  if (!snapshots?.length) return undefined
  for (let i = snapshots.length - 1; i >= 0; i--) {
    const ids = snapshots[i].promptMessageIds
    if (!ids || ids.every((id) => pathIds.has(id))) return snapshots[i]
  }
  return undefined
}

/**
 * Scale a heuristic section count by the thread's measured ratio so the
 * breakdown sums to the number llama.cpp actually reported.
 */
export function calibrateTokens(tokens: number, calibration: number): number {
  return Math.round(tokens * calibration)
}

/** Token overhead per message for role/formatting, mirroring context-manager. */
const MESSAGE_TOKEN_OVERHEAD = 4

export function buildMessageSections(
  messages: Array<{ role: string; content: unknown }>,
  startIndex = 0,
  roleLabels: Partial<Record<string, string>> = {},
  sourceIds: Array<string | undefined> = []
): ContextSnapshotSection[] {
  return messages.map((m, i) => {
    const content = modelMessageToText(m.content)
    return {
      id: `msg-${startIndex + i}`,
      kind: 'message' as const,
      label:
        roleLabels[m.role] || m.role.charAt(0).toUpperCase() + m.role.slice(1),
      role: m.role as ContextSnapshotSection['role'],
      tokens:
        content.length > 0 ? estimateTokens(content) + MESSAGE_TOKEN_OVERHEAD : 0,
      content,
      sourceId: sourceIds[i],
    }
  })
}

function modelMessageToText(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  const parts: string[] = []
  for (const part of content) {
    if (
      part &&
      typeof part === 'object' &&
      'text' in part &&
      typeof (part as { text: unknown }).text === 'string'
    ) {
      parts.push((part as { text: string }).text)
    }
  }
  return parts.join('\n')
}

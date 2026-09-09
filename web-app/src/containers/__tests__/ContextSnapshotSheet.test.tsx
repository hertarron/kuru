import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, act, fireEvent } from '@testing-library/react'
import { ThreadMessage } from '@janhq/core'
import { ContextSnapshotSheet } from '../ContextSnapshotSheet'
import { useContextSnapshotStore } from '@/stores/context-snapshot-store'
import { useThreads } from '@/hooks/useThreads'
import { useCharacters } from '@/hooks/useCharacters'
import { useAppState } from '@/hooks/useAppState'
import { useTokensCount } from '@/hooks/useTokensCount'

vi.mock('@/hooks/useTokensCount', () => ({ useTokensCount: vi.fn() }))

const mockUseTokensCount = vi.mocked(useTokensCount)

const THREAD = 'thread-1'

function mockTokens(overrides: Record<string, unknown> = {}) {
  mockUseTokensCount.mockReturnValue({
    tokenCount: 0,
    maxTokens: 10000,
    calculateTokens: vi.fn(),
    ...overrides,
  } as never)
}

function message(
  id: string,
  role: 'user' | 'assistant',
  text: string,
  metadata: Record<string, unknown> = {}
): ThreadMessage {
  return {
    id,
    object: 'thread.message',
    thread_id: THREAD,
    role,
    content: [{ type: 'text', text: { value: text, annotations: [] } }],
    status: 'ready',
    created_at: 1,
    completed_at: 1,
    metadata,
  } as ThreadMessage
}

function capture(
  overrides: Partial<Parameters<typeof useContextSnapshotStore.getState>[0]> = {}
) {
  act(() => {
    useContextSnapshotStore.getState().capture({
      threadId: THREAD,
      createdAt: Date.now(),
      sections: [
        {
          id: 'system-prompt-0',
          kind: 'system-prompt',
          label: 'System prompt',
          tokens: 100,
          content: 'You are Isekai RPG.',
        },
        {
          id: 'msg-0',
          kind: 'message',
          label: 'Coolpick',
          role: 'user',
          tokens: 20,
          content: 'hello',
        },
      ],
      totalTokens: 120,
      maxContextTokens: null,
      assistantLabel: 'Isekai RPG',
      promptMessageIds: ['u1'],
      ...(overrides as object),
    })
  })
}

function setActiveRoot(activeRootId?: string) {
  act(() => {
    useThreads.setState({
      threads: {
        [THREAD]: { id: THREAD, metadata: { activeRootId } } as never,
      },
    })
  })
}

const renderSheet = (messages: ThreadMessage[], streaming = false) =>
  render(
    <ContextSnapshotSheet
      threadId={THREAD}
      messages={messages}
      streaming={streaming}
      open
      onOpenChange={() => {}}
    />
  )

describe('ContextSnapshotSheet', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockTokens()
    act(() => {
      useContextSnapshotStore.setState({ snapshots: {}, calibration: {} })
      useAppState.setState({ liveTokenStatsByThread: {}, promptProgresses: {} })
    })
    setActiveRoot(undefined)
  })

  it('shows the prompt breakdown for the branch the snapshot was taken on', () => {
    capture()
    renderSheet([message('u1', 'user', 'hello')])
    fireEvent.click(screen.getByRole('button', { name: 'Prompt' }))

    expect(screen.getAllByText('System prompt').length).toBeGreaterThan(0)
    expect(screen.getByText('Prompt in send order')).toBeInTheDocument()
  })

  // Rows carry their thread-message id: the visualizer numbers them live
  // off the active path instead of trusting snapshot positions. Zero-based,
  // so the greeting is #0 as in SillyTavern.
  it('numbers prompt rows off the active path', () => {
    capture({
      sections: [
        {
          id: 'msg-0',
          kind: 'message',
          label: 'Coolpick',
          role: 'user',
          tokens: 20,
          content: 'hello',
          sourceId: 'u1',
        },
      ],
      totalTokens: 20,
    })
    renderSheet([message('u1', 'user', 'hello')])
    fireEvent.click(screen.getByRole('button', { name: 'Prompt' }))

    expect(screen.getByText('#0')).toBeInTheDocument()
  })

  // Compaction lives in the context view, per chat: auto tri-state plus
  // the size preset, with the outlook line telling the truth.
  it('shows per-chat compaction in the context view', () => {
    act(() => {
      useThreads.setState({
        threads: {
          [THREAD]: {
            id: THREAD,
            assistants: [{ id: 'char1' }],
            metadata: {},
          } as never,
        },
      })
      useCharacters.setState({
        characters: [{ id: 'char1', kind: 'roleplay' } as never],
      })
    })
    renderSheet([message('u1', 'user', 'hello')])

    expect(screen.getByText('Compaction in this chat')).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: /Default \(off\)/ })
    ).toBeInTheDocument()
    expect(
      screen.queryByText('Prompt in send order')
    ).not.toBeInTheDocument()
  })

  // Settings must not need messages to show up: an empty chat gets the
  // full compaction section plus the fold line. Auto is off by default, so
  // that line reads Paused rather than promising a chapter nothing writes.
  it('shows compaction on a zero-message chat, with the fold line', () => {
    act(() => {
      useThreads.setState({
        threads: {
          [THREAD]: {
            id: THREAD,
            assistants: [{ id: 'char1' }],
            metadata: {},
          } as never,
        },
      })
      useCharacters.setState({ characters: [] })
    })
    renderSheet([])

    expect(screen.getByText('Compaction in this chat')).toBeInTheDocument()
    expect(screen.getByText(/Paused/)).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: /Default \(off\)/ })
    ).toBeInTheDocument()
  })

  // Before a thread exists the same section stages into the pending store,
  // which ChatInput copies onto the thread it creates. No Write now: there
  // is nothing to fold yet.
  it('stages compaction before the thread exists', () => {
    render(
      <ContextSnapshotSheet
        messages={[]}
        streaming={false}
        roleplay
        open
        onOpenChange={() => {}}
      />
    )

    expect(screen.getByText('Compaction in this chat')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Write now/ })).not.toBeInTheDocument()
  })

  // Settings never hide: a coding chat gets the same section with an honest
  // note instead, and no Write now.
  it('shows compaction with a note on non-roleplay chats', () => {
    act(() => {
      useThreads.setState({
        threads: {
          [THREAD]: {
            id: THREAD,
            assistants: [{ id: 'model-only' }],
            metadata: {},
          } as never,
        },
      })
      useCharacters.setState({ characters: [] })
    })
    renderSheet([])

    expect(screen.getByText('Compaction in this chat')).toBeInTheDocument()
    expect(screen.getByText(/runs in roleplay chats/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Write now/ })).not.toBeInTheDocument()
  })

  // Regression: a snapshot records one request. After switching to a sibling
  // reply (or another greeting tree) it describes a branch the viewer is no
  // longer on, and was being passed off as the current prompt.
  it('withholds a snapshot captured on another branch', () => {
    capture({ promptMessageIds: ['g1', 'u1'] })
    setActiveRoot('g2')
    renderSheet([
      message('g1', 'assistant', 'greeting one', {
        parentId: null,
        activeChildId: 'u1',
      }),
      message('u1', 'user', 'hello', { parentId: 'g1' }),
      message('g2', 'assistant', 'greeting two', { parentId: null }),
    ])

    expect(screen.getByText('Captured on a different branch.')).toBeInTheDocument()
    expect(
      screen.getByText(/No breakdown for this branch/i)
    ).toBeInTheDocument()
    expect(screen.queryByText('Prompt in send order')).not.toBeInTheDocument()
  })

  // A thread with several greeting trees is several conversations. Each branch
  // that has been sent from keeps its own capture, so switching back restores
  // that branch's breakdown instead of showing the placeholder.
  it('restores the breakdown belonging to each branch when switching', () => {
    const section = (label: string) => [
      {
        id: 'system-prompt-0',
        kind: 'system-prompt' as const,
        label,
        tokens: 100,
        content: label,
      },
    ]
    capture({ sections: section('Tree one'), promptMessageIds: ['g1', 'u1'] })
    capture({ sections: section('Tree two'), promptMessageIds: ['g2', 'u2'] })
    const messages = [
      message('g1', 'assistant', 'greeting one', {
        parentId: null,
        activeChildId: 'u1',
      }),
      message('u1', 'user', 'first', { parentId: 'g1' }),
      message('g2', 'assistant', 'greeting two', {
        parentId: null,
        activeChildId: 'u2',
      }),
      message('u2', 'user', 'second', { parentId: 'g2' }),
    ]

    setActiveRoot('g1')
    const { unmount } = renderSheet(messages)
    expect(screen.getAllByText('Tree one').length).toBeGreaterThan(0)
    expect(screen.queryByText('Tree two')).not.toBeInTheDocument()
    unmount()

    setActiveRoot('g2')
    renderSheet(messages)
    expect(screen.getAllByText('Tree two').length).toBeGreaterThan(0)
    expect(screen.queryByText('Tree one')).not.toBeInTheDocument()
  })

  it('reports nothing captured for a thread that has never been sent', () => {
    renderSheet([])
    expect(
      screen.getByText(/Nothing captured yet/i)
    ).toBeInTheDocument()
  })

  // The snapshot is taken before the request, so the reply it produced is not
  // in it: without this row the panel goes blank on the model's own output.
  it('adds the latest reply as its own row, named after the character', () => {
    capture()
    mockTokens({ tokenCount: 300, outputTokens: 180, maxTokens: 10000 })
    renderSheet([
      message('u1', 'user', 'hello'),
      message('a1', 'assistant', 'a reply'),
    ])

    expect(screen.getAllByText('Isekai RPG').length).toBeGreaterThan(0)
    expect(screen.getAllByText(/180 tok/).length).toBeGreaterThan(0)
  })

  it('drops the reply row once a later capture contains that reply', () => {
    capture({
      sections: [
        {
          id: 'msg-0',
          kind: 'message',
          label: 'Coolpick',
          role: 'user',
          tokens: 20,
          content: 'hello',
        },
        {
          id: 'msg-1',
          kind: 'message',
          label: 'Isekai RPG',
          role: 'assistant',
          tokens: 60,
          content: 'a reply',
        },
      ],
      totalTokens: 80,
      promptMessageIds: ['u1', 'a1'],
    })
    mockTokens({ tokenCount: 300, outputTokens: 180, maxTokens: 10000 })
    renderSheet([
      message('u1', 'user', 'hello'),
      message('a1', 'assistant', 'a reply'),
    ])

    expect(screen.queryByText(/180 tok/)).not.toBeInTheDocument()
  })

  it('marks the reply as in flight while streaming', () => {
    capture()
    act(() => {
      useAppState.setState({
        liveTokenStatsByThread: {
          [THREAD]: {
            promptTokens: 240,
            completionTokens: 12,
            tokensPerSecond: 6,
            promptPerSecond: 400,
          },
        },
      })
    })
    mockTokens({ tokenCount: 252, outputTokens: 12, maxTokens: 10000 })
    renderSheet([message('u1', 'user', 'hello')], true)

    expect(screen.getByText('Streaming now')).toBeInTheDocument()
    expect(screen.getAllByText('Isekai RPG · replying').length).toBeGreaterThan(0)
    expect(screen.getByText('6.0 tok/s')).toBeInTheDocument()
  })

  // Rows come from a character heuristic; the thread's measured prompt scales
  // them so the panel never contradicts the tokenizer.
  it('scales the breakdown to the measured prompt', () => {
    capture()
    act(() => {
      useContextSnapshotStore
        .getState()
        .recordActualPromptTokens(THREAD, 60) // half of the 120 estimated
    })
    renderSheet([message('u1', 'user', 'hello')])

    expect(screen.getAllByText(/50 tok/).length).toBeGreaterThan(0)
  })
})

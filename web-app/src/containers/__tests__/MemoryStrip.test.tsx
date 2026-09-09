import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { render, screen, act, fireEvent } from '@testing-library/react'
import { vi } from 'vitest'
import type { ThreadMessage } from '@janhq/core'
import { MemoryStrip } from '../MemoryStrip'
import { useThreadMemory } from '@/hooks/useThreadMemory'
import { useMemoryRuns } from '@/hooks/useMemoryRuns'
import { useMemorySettings } from '@/hooks/useMemorySettings'

const THREAD_ID = 'strip-thread'

const thread = { id: THREAD_ID, metadata: {} } as Thread

function message(id: string, role: 'user' | 'assistant'): ThreadMessage {
  return {
    id,
    object: 'thread.message',
    thread_id: THREAD_ID,
    role,
    content: [{ type: 'text', text: { value: `line ${id}`, annotations: [] } }],
    status: 'ready',
    created_at: 1,
    completed_at: 1,
    metadata: {},
  } as ThreadMessage
}

describe('MemoryStrip', () => {
  beforeEach(() => {
    act(() => {
      useThreadMemory.getState().clearAll()
    })
  })

  afterEach(() => {
    act(() => {
      useThreadMemory.getState().clearAll()
      useMemoryRuns.getState().end(THREAD_ID)
    })
  })

  it('renders nothing without a thread', () => {
    render(<MemoryStrip messages={[]} onOpen={() => {}} />)
    expect(screen.queryByRole('button', { name: 'Open memory' })).not.toBeInTheDocument()
  })

  it('counts down from the first turns, raw window included', () => {
    act(() => {
      useMemorySettings.setState({ autoDefault: true })
    })
    const six = Array.from({ length: 6 }, (_, i) =>
      message(`m${i + 1}`, i % 2 === 0 ? 'user' : 'assistant')
    )
    const { rerender } = render(
      <MemoryStrip thread={thread} messages={six} onOpen={() => {}} />
    )
    // Medium: the fold trips at batch + keepRaw = 35 countable messages.
    expect(screen.getByText('Next chapter in 15 turns')).toBeInTheDocument()

    rerender(
      <MemoryStrip
        thread={thread}
        messages={[
          ...six,
          message('m7', 'user'),
          message('m8', 'assistant'),
        ]}
        onOpen={() => {}}
      />
    )
    expect(screen.getByText('Next chapter in 14 turns')).toBeInTheDocument()
  })

  it('says paused rather than promising a fold when auto is off', () => {
    act(() => {
      useMemorySettings.setState({ autoDefault: false })
    })
    render(<MemoryStrip thread={thread} messages={[]} onOpen={() => {}} />)
    expect(screen.getByText('Paused')).toBeInTheDocument()
  })

  it('announces the fold while one runs', () => {
    act(() => {
      useMemoryRuns.getState().begin(THREAD_ID, 'chapter')
    })
    render(<MemoryStrip thread={thread} messages={[]} onOpen={() => {}} />)

    expect(screen.getByText('Writing chapter…')).toBeInTheDocument()
  })

  it('opens the context view on tap', () => {
    const onOpen = vi.fn()
    render(<MemoryStrip thread={thread} messages={[]} onOpen={onOpen} />)
    fireEvent.click(screen.getByRole('button', { name: 'Open memory' }))
    expect(onOpen).toHaveBeenCalledOnce()
  })
})

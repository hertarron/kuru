import { describe, it, expect } from 'vitest'
import { shouldShowTokenCounter } from '@/lib/tokenCounterVisibility'

const base = {
  hasSelectedModel: true,
  isAgentMode: false,
}

describe('shouldShowTokenCounter', () => {
  it('shows when a model is selected', () => {
    expect(shouldShowTokenCounter(base)).toBe(true)
  })

  it('hides when no model is selected', () => {
    expect(shouldShowTokenCounter({ ...base, hasSelectedModel: false })).toBe(
      false
    )
  })

  it('hides in agent mode', () => {
    expect(shouldShowTokenCounter({ ...base, isAgentMode: true })).toBe(false)
  })

  // Regression: the token counter must NOT depend on whether the model is
  // currently in `activeModels`. Router-mode llama.cpp loads lazily and never
  // writes the model back into `activeModels` during a chat turn, so a gate on
  // "is the model active" hid the counter for llama.cpp entirely while remote
  // providers (which were gated only on model selection) kept showing it.
  // There is no `isModelActive` input at all: the predicate is provider- and
  // load-state-agnostic by construction.
  it('shows for a local (llama.cpp) model regardless of active-model state', () => {
    expect(shouldShowTokenCounter(base)).toBe(true)
  })

  // Regression: the counter is also the context-visualizer trigger, so it is a
  // permanent fixture of the chatbox rather than something that appears once
  // a turn has been sent. The sigma pill on a fresh composer opens the same
  // context view, where per-chat compaction is set before anything is sent.
  it('shows in an empty thread with no messages and no typed text', () => {
    expect(shouldShowTokenCounter(base)).toBe(true)
  })

  it('shows on the new-thread composer, before the first send', () => {
    expect(shouldShowTokenCounter(base)).toBe(true)
  })

  // Cowork keeps no thread messages and mounts a permanently "initial" input,
  // so every one of the thread-shaped conditions is false for it.
  describe('reported usage', () => {
    const cowork = {
      ...base,
      isInitialMessage: true,
      hasMessages: false,
      hasPromptText: false,
    }

    it('shows once a surface reports usage of its own', () => {
      expect(
        shouldShowTokenCounter({ ...cowork, hasReportedUsage: true })
      ).toBe(true)
    })

    it('does not override the model and agent-mode conditions', () => {
      expect(
        shouldShowTokenCounter({
          ...cowork,
          hasReportedUsage: true,
          hasSelectedModel: false,
        })
      ).toBe(false)
      expect(
        shouldShowTokenCounter({
          ...cowork,
          hasReportedUsage: true,
          isAgentMode: true,
        })
      ).toBe(false)
    })
  })
})

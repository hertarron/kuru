export interface TokenCounterVisibilityInput {
  hasSelectedModel: boolean
  isAgentMode: boolean
}

/**
 * Whether the chat-input token counter should mount. It is a permanent
 * fixture of the chatbox (it doubles as the context-visualizer trigger), so
 * it does not wait for messages, typed text, or the first send — the sigma
 * pill on a fresh composer opens the same context view, where per-chat
 * compaction can be set before anything is sent.
 *
 * Deliberately provider- and load-state-agnostic: gating here on model load
 * state (`activeModels`) only ever hid the counter for lazily-loaded local
 * engines. See tokenCounterVisibility.test.ts.
 */
export const shouldShowTokenCounter = ({
  hasSelectedModel,
  isAgentMode,
}: TokenCounterVisibilityInput): boolean =>
  hasSelectedModel && !isAgentMode

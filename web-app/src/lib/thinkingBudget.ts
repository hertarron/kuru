export type ThinkingBudgetLevelKey =
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh'
  | 'unlimited'

// Fractions of the model's context window; null = unlimited (-1, llama.cpp's
// sentinel for "don't cap reasoning"). Resolved against the LIVE (post-fit)
// context size at send time, not the configured/default size at selection
// time, since llama.cpp's --fit can pick a runtime n_ctx far from either.
// Calibrated for roleplay: responses are short (~100-500 tokens), so even the
// lowest level must stay small — at 32k context, low = 1.6k thinking tokens,
// already several times the response length.
export const THINKING_BUDGET_LEVELS: Array<{
  key: ThinkingBudgetLevelKey
  label: string
  ratio: number | null
}> = [
  { key: 'low', label: 'Low', ratio: 0.05 },
  { key: 'medium', label: 'Medium', ratio: 0.1 },
  { key: 'high', label: 'High', ratio: 0.2 },
  { key: 'xhigh', label: 'XHigh', ratio: 0.35 },
  { key: 'unlimited', label: 'Unlimited', ratio: null },
]

export const DEFAULT_THINKING_BUDGET_LEVEL: ThinkingBudgetLevelKey = 'unlimited'

export function tokensForThinkingBudgetLevel(
  level: ThinkingBudgetLevelKey,
  contextSize: number
): number {
  const ratio = THINKING_BUDGET_LEVELS.find((l) => l.key === level)?.ratio
  return ratio == null ? -1 : Math.max(1, Math.round(contextSize * ratio))
}

export function isThinkingBudgetLevelKey(
  value: unknown
): value is ThinkingBudgetLevelKey {
  return (
    typeof value === 'string' &&
    THINKING_BUDGET_LEVELS.some((l) => l.key === value)
  )
}

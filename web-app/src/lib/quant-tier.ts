export type QuantTier = {
  label: string
  className: string
}

/**
 * A rough size/quality band for a quant, read off its filename. Shown beside
 * the quant id so the list can be skimmed without knowing what IQ4_XS means.
 */
export function getQuantTier(modelId: string): QuantTier | null {
  const id = modelId.toLowerCase()
  if (/(^|[-_.])(f32|bf16|f16|q8|q6)([-_.]|$)/.test(id)) {
    return {
      label: 'Large',
      className: 'bg-amber-500/10 text-amber-700 dark:text-amber-400',
    }
  }
  if (/(^|[-_.])(q5|q4_k|iq4)/.test(id)) {
    return {
      label: 'Balanced',
      className: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
    }
  }
  if (/(^|[-_.])(iq2|iq3|q2|q3|q4_0|q4_1)/.test(id)) {
    return {
      label: 'Small',
      className: 'bg-blue-500/10 text-blue-600 dark:text-blue-400',
    }
  }
  return null
}

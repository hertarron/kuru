import { createContext, useCallback, useContext, useLayoutEffect, useRef } from 'react'
import type { PillKey } from '@/lib/pill-labels'

export type Registration = {
  forms: string[]
  /**
   * The pill's live nodes, not a copy of them. A snapshot taken at
   * registration goes stale the moment a fit swaps or drops the label
   * element, and the next fit then measures a detached node as zero.
   */
  nodes: { pill: HTMLElement | null; label: HTMLElement | null }
}

export type PillRegistry = {
  register: (key: PillKey, registration: Registration) => void
  unregister: (key: PillKey) => void
}

/**
 * The registry keeps one identity for the row's life. A pill's registration
 * effect depends on it, so folding the fitted labels in here made every fit
 * re-register every pill, refit, and loop without finishing a commit.
 */
export const PillRegistryContext = createContext<PillRegistry | null>(null)
export const PillLabelsContext = createContext<Map<PillKey, string> | null>(null)

/**
 * Reports this pill's label forms and returns the one to render.
 *
 * Outside a `PillRow` -- the header pills, which stand alone -- there is
 * nothing to fit against and the full label comes back unchanged.
 */
export function usePillLabel(
  key: PillKey,
  forms: string[]
): {
  label: string
  pillRef: (el: HTMLElement | null) => void
  labelRef: (el: HTMLElement | null) => void
} {
  const registry = useContext(PillRegistryContext)
  const fitted = useContext(PillLabelsContext)
  const nodes = useRef<{ pill: HTMLElement | null; label: HTMLElement | null }>({
    pill: null,
    label: null,
  })
  const latestForms = useRef(forms)
  latestForms.current = forms
  // A new array on every render, so the effect keys off its content instead.
  const signature = forms.join('|')

  useLayoutEffect(() => {
    if (!registry) return
    registry.register(key, { forms: latestForms.current, nodes: nodes.current })
    return () => registry.unregister(key)
  }, [registry, key, signature])

  const pillRef = useCallback((el: HTMLElement | null) => {
    nodes.current.pill = el
  }, [])
  const labelRef = useCallback((el: HTMLElement | null) => {
    nodes.current.label = el
  }, [])

  return {
    label: fitted?.get(key) ?? forms[forms.length - 1],
    pillRef,
    labelRef,
  }
}

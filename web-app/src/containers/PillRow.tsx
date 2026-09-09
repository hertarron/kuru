import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { fitPillLabels, type PillKey, type PillMetrics } from '@/lib/pill-labels'
import {
  PillLabelsContext,
  PillRegistryContext,
  type Registration,
} from '@/hooks/usePillLabel'

/** gap-1 between the pills. */
const ROW_GAP = 4
/** gap-1.5 between a pill's icon and its label, gone when the label is. */
const LABEL_GAP = 6
/**
 * Covers the last of the sub-pixel error. Every width here is fractional, but
 * the browser still lays text out with its own rounding, and a form chosen
 * with nothing to spare comes back from that as an ellipsis.
 */
const SLACK = 1

const widthOf = (el: HTMLElement | null | undefined) =>
  el ? el.getBoundingClientRect().width : 0

const sameLabels = (a: Map<PillKey, string>, b: Map<PillKey, string>) =>
  a.size === b.size && [...a].every(([key, label]) => b.get(key) === label)

/**
 * Measures the toolbar pills and picks how much of each label fits.
 *
 * Breakpoints keyed off the chatbox width, so every name yielded at the same
 * point and room a short name did not use went to nobody.
 *
 * `flex-1` keeps the layout effect from looping: the row's width comes from
 * the space beside the button group, not from the pills inside it.
 */
export function PillRow({ children }: { children: React.ReactNode }) {
  const rowRef = useRef<HTMLDivElement>(null)
  const ghostRef = useRef<HTMLSpanElement>(null)
  const entries = useRef(new Map<PillKey, Registration>())
  const [revision, setRevision] = useState(0)
  const [labels, setLabels] = useState<Map<PillKey, string>>(new Map())

  // Stable for the life of the row; see the note in usePillLabel.
  const registry = useMemo(
    () => ({
      register: (key: PillKey, registration: Registration) => {
        entries.current.set(key, registration)
        setRevision((r) => r + 1)
      },
      unregister: (key: PillKey) => {
        entries.current.delete(key)
        setRevision((r) => r + 1)
      },
    }),
    []
  )

  // Bumped by anything that can change the answer. The width itself is read
  // live inside the fit, never cached here: a width cached while the row was
  // still unlaid measured zero, and the fit then had no way back.
  const invalidate = useCallback(() => setRevision((r) => r + 1), [])

  // Watches the row and every pill in it. Watching only the row missed the
  // case that matters most: the character, persona and lorebook labels arrive
  // from disk after the first fit, and loading one changes that pill's width
  // without changing the row's, so nothing refit and the labels stayed whole.
  // A pill that resizes for any reason now refits, whatever the reason was.
  const watched = useRef(new Set<Element>())
  const observer = useRef<ResizeObserver | null>(null)
  const watch = useCallback((elements: Element[]) => {
    const ro = observer.current
    if (!ro) return
    const wanted = new Set(elements)
    for (const el of watched.current) {
      if (!wanted.has(el)) {
        ro.unobserve(el)
        watched.current.delete(el)
      }
    }
    // Observing fires once for each new element, which costs one extra fit
    // that settles on the same labels and stops there.
    for (const el of wanted) {
      if (!watched.current.has(el)) {
        ro.observe(el)
        watched.current.add(el)
      }
    }
  }, [])

  useLayoutEffect(() => {
    const ro = new ResizeObserver(invalidate)
    const seen = watched.current
    observer.current = ro
    if (rowRef.current) watch([rowRef.current])
    return () => {
      ro.disconnect()
      observer.current = null
      seen.clear()
    }
  }, [invalidate, watch])

  // A web font changes every label's width without changing the row's, so
  // nothing else here would refit. The first fit measured the fallback font,
  // which is narrower, and every label looked like it fit.
  useLayoutEffect(() => {
    let live = true
    document.fonts?.ready.then(() => {
      if (live) invalidate()
    })
    return () => {
      live = false
    }
  }, [invalidate])

  useLayoutEffect(() => {
    let frame = 0

    const fit = () => {
      const row = rowRef.current
      const ghost = ghostRef.current
      if (!row || !ghost || entries.current.size === 0) return

      const available = widthOf(row)
      // A row that has not been laid out yet measures zero, which happens
      // when it remounts inside a container that is still collapsed. Wait for
      // a frame that has real geometry: giving up here left the labels whole
      // until the user resized the window by hand.
      if (available === 0) {
        frame = requestAnimationFrame(fit)
        return
      }

      const pills = new Map<PillKey, PillMetrics>()
      for (const [key, entry] of entries.current) {
        const labelWidth = widthOf(entry.nodes.label)
        const pillWidth = widthOf(entry.nodes.pill)
        // Measure in the label's own font rather than a copy of its classes:
        // the ghost sits outside the pill, so anything the pill inherits --
        // family, weight, letter-spacing -- would otherwise skew every form.
        if (entry.nodes.label)
          ghost.style.font = getComputedStyle(entry.nodes.label).font
        pills.set(key, {
          forms: entry.forms,
          formWidths: entry.forms.map((form) => {
            if (!form) return 0
            ghost.textContent = form
            return widthOf(ghost)
          }),
          // The icon-to-label gap exists only while a label does, so it is
          // priced with the label rather than folded into the fixed width.
          fixed: Math.max(
            0,
            pillWidth - labelWidth - (labelWidth > 0 ? LABEL_GAP : 0)
          ),
          labelGap: LABEL_GAP,
        })
      }

      watch([
        row,
        ...[...entries.current.values()]
          .map((entry) => entry.nodes.pill)
          .filter((pill): pill is HTMLElement => !!pill),
      ])

      const next = fitPillLabels(pills, available - SLACK, ROW_GAP)
      setLabels((prev) => (sameLabels(prev, next) ? prev : next))
    }

    fit()
    return () => cancelAnimationFrame(frame)
  }, [revision, watch])

  return (
    <PillRegistryContext.Provider value={registry}>
      <PillLabelsContext.Provider value={labels}>
        <div
          ref={rowRef}
          className="flex flex-1 items-center gap-1 min-w-0 relative"
        >
          {children}
          {/* Measures a candidate label in the row's own font. Absolute, so
              it takes part in no layout of its own. */}
          <span
            ref={ghostRef}
            aria-hidden
            className="absolute left-0 top-0 invisible whitespace-nowrap text-xs font-medium"
          />
        </div>
      </PillLabelsContext.Provider>
    </PillRegistryContext.Provider>
  )
}

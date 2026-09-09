import { useEffect, useMemo, useRef, useState } from 'react'
import { IconFilter, IconMinus, IconPlus, IconX } from '@tabler/icons-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { getSuggestedTags, probeChubTagCount } from '@/lib/chub'

export type TagFilter = { tag: string; mode: 'include' | 'exclude' }

interface ChubTagFilterProps {
  filters: TagFilter[]
  setFilters: React.Dispatch<React.SetStateAction<TagFilter[]>>
  token?: string
  /** Which catalog the live counts are probed against. */
  namespace?: 'characters' | 'lorebooks'
}

/**
 * Include/exclude tag picker for Chub search, shared by the characters and
 * lorebooks hub pages.
 *
 * The seed suggestion list is harvested from character tags. Lorebooks use a
 * narrower vocabulary, so a suggestion there can come back with a count of 0;
 * the live probe is namespaced, which makes that visible before the tag is
 * committed rather than after it empties the results.
 */
export function ChubTagFilter({
  filters,
  setFilters,
  token,
  namespace = 'characters',
}: ChubTagFilterProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const panelRef = useRef<HTMLDivElement>(null)
  const noun = namespace === 'lorebooks' ? 'lorebooks' : 'cards'

  const addFilter = (tag: string, mode: TagFilter['mode'] = 'include') => {
    setFilters((prev) =>
      prev.some((t) => t.tag.toLowerCase() === tag.toLowerCase())
        ? prev
        : [...prev, { tag, mode }]
    )
  }

  // Live probe: exact count for whatever is typed, so near-miss tags
  // ("woman" vs "female") are visible before committing. Debounced; one
  // minimal search call per pause, cached in chub.ts.
  const [probeCount, setProbeCount] = useState<number | null>(null)
  const [probing, setProbing] = useState(false)
  useEffect(() => {
    const typed = query.trim()
    if (!typed) {
      setProbeCount(null)
      setProbing(false)
      return
    }
    setProbing(true)
    const handler = setTimeout(() => {
      probeChubTagCount(typed, token, namespace)
        .then((count) => setProbeCount(count))
        .catch(() => setProbeCount(null))
        .finally(() => setProbing(false))
    }, 400)
    return () => clearTimeout(handler)
  }, [query, token, namespace])

  // Close on outside clicks; closing resets the panel's search box.
  useEffect(() => {
    if (!open) setQuery('')
    const onPointerDown = (e: PointerEvent) => {
      if (!panelRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [open])

  const suggestions = useMemo(
    () => getSuggestedTags(query, filters.map((t) => t.tag)),
    [query, filters]
  )

  // Exact counts fetched lazily once the list settles — seed counts are only
  // a sampled popularity ranking, and are wrong outright for lorebooks.
  const [exactCounts, setExactCounts] = useState<Record<string, number>>({})
  const probedKeysRef = useRef(new Set<string>())
  useEffect(() => {
    probedKeysRef.current.clear()
    setExactCounts({})
  }, [namespace])
  useEffect(() => {
    if (!open || suggestions.length === 0) return
    const handler = setTimeout(() => {
      for (const [tag] of suggestions) {
        const key = tag.toLowerCase()
        if (probedKeysRef.current.has(key)) continue
        probedKeysRef.current.add(key)
        probeChubTagCount(tag, token, namespace)
          .then((count) => setExactCounts((prev) => ({ ...prev, [key]: count })))
          .catch(() => probedKeysRef.current.delete(key))
      }
    }, 300)
    return () => clearTimeout(handler)
  }, [open, suggestions, token, namespace])

  return (
    <div className="relative" ref={panelRef}>
      <Button
        variant={open || filters.length ? 'secondary' : 'outline'}
        size="sm"
        onClick={() => setOpen((v) => !v)}
      >
        <IconFilter size={16} />
        Tags
      </Button>
      {open && (
        <div className="absolute top-full mt-1 z-30 w-72 rounded-md border bg-popover shadow-md py-1">
          <div className="px-2 pt-0.5 pb-1.5">
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setOpen(false)
                if (e.key === 'Enter') {
                  const typed = query.trim()
                  if (typed) {
                    e.preventDefault()
                    addFilter(typed, 'include')
                    setQuery('')
                  }
                }
              }}
              placeholder="Filter tags..."
              className="w-full h-7 rounded-md border border-input bg-transparent px-2 text-xs focus:outline-none focus:border-ring"
            />
            {query.trim() && (
              <p
                className={cn(
                  'px-1 pt-1 text-[10px] tabular-nums',
                  probeCount === 0 ? 'text-destructive' : 'text-muted-foreground'
                )}
              >
                {probing
                  ? 'checking…'
                  : probeCount !== null
                    ? `Enter adds “${query.trim()}” · ${probeCount.toLocaleString()} ${noun}`
                    : `Enter adds “${query.trim()}”`}
              </p>
            )}
          </div>
          <div className="max-h-60 overflow-y-auto">
            {suggestions.length === 0 ? (
              <p className="px-3 py-2 text-xs text-muted-foreground">
                No similar tags — Enter adds “{query.trim()}” as-is.
              </p>
            ) : (
              suggestions.map(([tag]) => {
                const active = filters.find(
                  (t) => t.tag.toLowerCase() === tag.toLowerCase()
                )
                const exact = exactCounts[tag.toLowerCase()]
                return (
                  <div
                    key={tag}
                    className={cn(
                      'flex items-center justify-between pl-2.5 pr-1 py-0.5 hover:bg-accent',
                      exact === 0 && 'opacity-50'
                    )}
                  >
                    <span className="text-xs truncate">{tag}</span>
                    <span className="flex items-center gap-1 shrink-0 ml-2">
                      <span className="text-[10px] text-muted-foreground tabular-nums">
                        {exact !== undefined ? exact.toLocaleString() : '…'}
                      </span>
                      <button
                        onClick={() => addFilter(tag, 'include')}
                        title="Include"
                        className={cn(
                          'flex items-center justify-center size-5 rounded cursor-pointer',
                          active?.mode === 'include'
                            ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                            : 'text-muted-foreground hover:bg-emerald-500/15 hover:text-emerald-600'
                        )}
                      >
                        <IconPlus size={12} />
                      </button>
                      <button
                        onClick={() => addFilter(tag, 'exclude')}
                        title="Exclude"
                        className={cn(
                          'flex items-center justify-center size-5 rounded cursor-pointer',
                          active?.mode === 'exclude'
                            ? 'bg-red-500/20 text-red-600 dark:text-red-400'
                            : 'text-muted-foreground hover:bg-red-500/15 hover:text-red-600'
                        )}
                      >
                        <IconMinus size={12} />
                      </button>
                    </span>
                  </div>
                )
              })
            )}
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * The committed filters, on the title row beside the tag picker.
 *
 * One line rather than a wrapping block, so the row's height does not jump as
 * filters are added; the row scrolls sideways instead, by wheel or by dragging
 * it, since a horizontal scrollbar on a short strip is worse than no affordance.
 */
export function ChubTagChips({
  filters,
  setFilters,
  className,
}: {
  filters: TagFilter[]
  setFilters: React.Dispatch<React.SetStateAction<TagFilter[]>>
  className?: string
}) {
  const scroller = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; left: number; moved: boolean } | null>(null)

  if (filters.length === 0) return null

  const toggleMode = (tag: string) =>
    setFilters((prev) =>
      prev.map((t) =>
        t.tag === tag
          ? { ...t, mode: t.mode === 'include' ? 'exclude' : 'include' }
          : t
      )
    )
  const remove = (tag: string) =>
    setFilters((prev) => prev.filter((t) => t.tag !== tag))

  const onPointerDown = (e: React.PointerEvent) => {
    const el = scroller.current
    if (!el || el.scrollWidth <= el.clientWidth) return
    drag.current = { x: e.clientX, left: el.scrollLeft, moved: false }
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const el = scroller.current
    if (!el || !drag.current) return
    const dx = e.clientX - drag.current.x
    if (Math.abs(dx) > 3) drag.current.moved = true
    el.scrollLeft = drag.current.left - dx
  }
  const endDrag = () => {
    drag.current = null
  }

  return (
    <div
      ref={scroller}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerLeave={endDrag}
      // A drag that scrolled must not also fire the chip's button underneath.
      onClickCapture={(e) => {
        if (drag.current?.moved) {
          e.preventDefault()
          e.stopPropagation()
        }
      }}
      className={cn(
        'flex items-center gap-1.5 overflow-x-auto no-scrollbar [scrollbar-width:none] [&::-webkit-scrollbar]:hidden',
        className
      )}
    >
      {filters.map(({ tag, mode }) => (
        <span
          key={tag}
          className={cn(
            'inline-flex shrink-0 items-center gap-0.5 rounded-full border py-0.5 pl-1 pr-1.5 text-xs',
            mode === 'include'
              ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
              : 'border-red-500/40 bg-red-500/10 text-red-600 dark:text-red-400'
          )}
        >
          <button
            onClick={() => toggleMode(tag)}
            title={
              mode === 'include'
                ? 'Including — click to exclude'
                : 'Excluding — click to include'
            }
            className="flex items-center justify-center size-4 rounded-full hover:bg-foreground/10 cursor-pointer"
          >
            {mode === 'include' ? <IconPlus size={11} /> : <IconMinus size={11} />}
          </button>
          <span className="px-0.5">{tag}</span>
          <button
            onClick={() => remove(tag)}
            title="Remove filter"
            className="flex items-center justify-center size-4 rounded-full hover:bg-foreground/10 cursor-pointer"
          >
            <IconX size={11} />
          </button>
        </span>
      ))}
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import { IconFilter } from '@tabler/icons-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { TagFilter } from '@/containers/ChubTagFilter'

export type Facet = {
  /** Matches the `tag` on a TagFilter, so these share the chip row with Chub tags. */
  id: string
  label: string
  description?: string
}

/**
 * The models tab's answer to the Chub tag picker. Its catalog has no tags, only
 * derived facts — tool support, a vision projector, MLX — so the vocabulary is
 * fixed rather than searched, but it produces the same include/exclude filters
 * and renders through the same chip row.
 */
export function HubFacetFilter({
  facets,
  filters,
  setFilters,
  label = 'Filters',
}: {
  facets: Facet[]
  filters: TagFilter[]
  setFilters: React.Dispatch<React.SetStateAction<TagFilter[]>>
  label?: string
}) {
  const [open, setOpen] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (!panelRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const modeOf = (id: string) =>
    filters.find((f) => f.tag === id)?.mode ?? null

  // Cycles include → exclude → off, so one control covers all three states.
  const cycle = (id: string) => {
    setFilters((prev) => {
      const current = prev.find((f) => f.tag === id)
      if (!current) return [...prev, { tag: id, mode: 'include' as const }]
      if (current.mode === 'include')
        return prev.map((f) =>
          f.tag === id ? { ...f, mode: 'exclude' as const } : f
        )
      return prev.filter((f) => f.tag !== id)
    })
  }

  return (
    <div className="relative shrink-0" ref={panelRef}>
      <Button
        variant={open || filters.length ? 'secondary' : 'outline'}
        size="sm"
        onClick={() => setOpen((v) => !v)}
      >
        <IconFilter size={16} />
        {label}
      </Button>
      {open && (
        <div className="absolute top-full mt-1 z-30 w-64 rounded-md border bg-popover shadow-md py-1">
          {facets.map((facet) => {
            const mode = modeOf(facet.id)
            return (
              <button
                key={facet.id}
                onClick={() => cycle(facet.id)}
                className="w-full text-left px-2 py-1.5 flex items-start gap-2 hover:bg-secondary/60 cursor-pointer"
              >
                <span
                  className={cn(
                    'mt-0.5 size-3.5 shrink-0 rounded-sm border flex items-center justify-center text-[10px] leading-none',
                    mode === 'include' &&
                      'border-emerald-500/60 bg-emerald-500/15 text-emerald-600 dark:text-emerald-400',
                    mode === 'exclude' &&
                      'border-red-500/60 bg-red-500/15 text-red-600 dark:text-red-400'
                  )}
                >
                  {mode === 'include' ? '+' : mode === 'exclude' ? '−' : ''}
                </span>
                <span className="min-w-0">
                  <span className="text-sm block truncate">{facet.label}</span>
                  {facet.description && (
                    <span className="text-xs text-muted-foreground block">
                      {facet.description}
                    </span>
                  )}
                </span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default HubFacetFilter

import { createFileRoute } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { useCallback, useEffect, useRef, useState } from 'react'

import HeaderPage from '@/containers/HeaderPage'
import { HubTabs } from '@/containers/HubTabs'
import { HubSearch } from '@/containers/HubSearch'
import { HubTitleRow } from '@/containers/HubTitleRow'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { Switch } from '@/components/ui/switch'
import { HUB_COLUMN, CARD_GRID } from '@/constants/layout'
import {
  ChubTagChips,
  ChubTagFilter,
  type TagFilter,
} from '@/containers/ChubTagFilter'
import { IconBook, IconChevronRight, IconStar } from '@tabler/icons-react'
import { ChevronsUpDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { cn } from '@/lib/utils'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import {
  searchChubLorebooks,
  fetchChubLorebook,
  type ChubLorebookResult,
} from '@/lib/chub'
import { LorebookPosition, type Lorebook } from '@/lib/lorebook'
import { useLorebooks } from '@/hooks/useLorebooks'
import { ulid } from 'ulidx'

const CHUB_SORTS = [
  { value: 'default', label: 'Popular' },
  { value: 'trending', label: 'Trending' },
  { value: 'star_count', label: 'Most starred' },
  { value: 'rating', label: 'Top rated' },
  { value: 'last_activity_at', label: 'Recently updated' },
] as const

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.hub.lorebooks as any)({
  component: LorebooksContent,
})

const POSITION_LABELS: Record<number, string> = {
  [LorebookPosition.beforeChar]: 'Before character',
  [LorebookPosition.afterChar]: 'After character',
  [LorebookPosition.ANTop]: 'Top of author’s note',
  [LorebookPosition.ANBottom]: 'Bottom of author’s note',
  [LorebookPosition.atDepth]: 'At depth',
  [LorebookPosition.EMTop]: 'Before examples',
  [LorebookPosition.EMBottom]: 'After examples',
  [LorebookPosition.outlet]: 'Outlet',
}

function LorebooksContent() {
  const { t } = useTranslation()
  const chubToken = useGeneralSetting((s) => s.chubToken)
  const [searchValue, setSearchValue] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [sortSelected, setSortSelected] = useState<string>('default')
  const [tagFilters, setTagFilters] = useState<TagFilter[]>([])

  const [results, setResults] = useState<ChubLorebookResult[]>([])
  const [count, setCount] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [loadingMore, setLoadingMore] = useState(false)
  const [loadMoreError, setLoadMoreError] = useState<string | null>(null)

  const [onlyImported, setOnlyImported] = useState(false)
  const [importingPath, setImportingPath] = useState<string | null>(null)
  const [importError, setImportError] = useState<string | null>(null)
  const [previewOf, setPreviewOf] = useState<ChubLorebookResult | null>(null)
  const [preview, setPreview] = useState<Lorebook | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const lorebooks = useLorebooks((s) => s.lorebooks)
  const addLorebook = useLorebooks((s) => s.addLorebook)

  useEffect(() => {
    const id = setTimeout(() => setDebouncedQuery(searchValue), 350)
    return () => clearTimeout(id)
  }, [searchValue])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    setLoadMoreError(null)
    setPage(1)
    searchChubLorebooks(debouncedQuery, {
      token: chubToken ?? undefined,
      sort: sortSelected,
      page: 1,
      first: 24,
      tags: tagFilters.filter((t) => t.mode === 'include').map((t) => t.tag),
      excludedTags: tagFilters
        .filter((t) => t.mode === 'exclude')
        .map((t) => t.tag),
    })
      .then((data) => {
        if (cancelled) return
        setResults(data.nodes)
        setCount(data.count)
      })
      .catch((e) => {
        if (cancelled) return
        setResults([])
        setCount(null)
        setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [debouncedQuery, chubToken, sortSelected, tagFilters])

  // Infinite scroll: fetch the next page when the sentinel below the grid
  // approaches the viewport. Appends are deduped by id — Chub's sort can
  // shift pages under us while scrolling.
  const canLoadMore =
    !loading &&
    !error &&
    !loadMoreError &&
    !loadingMore &&
    count !== null &&
    count > 0 &&
    results.length > 0 &&
    results.length < count

  const loadMore = useCallback(() => {
    if (!canLoadMore) return
    setLoadingMore(true)
    const nextPage = page + 1
    searchChubLorebooks(debouncedQuery, {
      token: chubToken ?? undefined,
      sort: sortSelected,
      page: nextPage,
      first: 24,
      tags: tagFilters.filter((t) => t.mode === 'include').map((t) => t.tag),
      excludedTags: tagFilters
        .filter((t) => t.mode === 'exclude')
        .map((t) => t.tag),
    })
      .then((data) => {
        setResults((prev) => {
          const seen = new Set(prev.map((n) => n.id))
          return [...prev, ...data.nodes.filter((n) => !seen.has(n.id))]
        })
        setPage(nextPage)
        setCount(data.count)
      })
      .catch((e) => {
        setLoadMoreError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => setLoadingMore(false))
  }, [canLoadMore, page, debouncedQuery, chubToken, sortSelected, tagFilters])

  const sentinelRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const el = sentinelRef.current
    // With the Downloaded toggle on, most of each fetched page is filtered
    // out, so the sentinel never leaves the viewport and auto-paging spins.
    // Paging becomes a button there instead.
    if (!el || onlyImported) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) loadMore()
      },
      { rootMargin: '600px' }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [loadMore, onlyImported])

  const openPreview = (node: ChubLorebookResult) => {
    setPreviewOf(node)
    setPreview(null)
    setPreviewError(null)
    fetchChubLorebook(node)
      .then(setPreview)
      .catch((e) => setPreviewError(e instanceof Error ? e.message : String(e)))
  }

  // A book already in the library is matched by where it came from, not by id:
  // the library id is minted locally on import.
  const importedBook = previewOf
    ? lorebooks.find((b) => b.source === `chub:${previewOf.fullPath}`)
    : undefined

  const importPreview = () => {
    if (!preview || importedBook) return
    addLorebook({ ...preview, id: ulid() })
  }

  // The card downloads without opening the sheet, so it has to fetch the book
  // itself rather than reuse the preview.
  const handleChubImport = async (node: ChubLorebookResult) => {
    setImportingPath(node.fullPath)
    setImportError(null)
    try {
      const book = await fetchChubLorebook(node)
      addLorebook({ ...book, id: ulid() })
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e))
    } finally {
      setImportingPath(null)
    }
  }

  // Chub has no "already imported" query, so this filters what has been
  // fetched rather than the catalog: with the toggle on, paging further can
  // still turn up more matches.
  const importedSources = new Set(
    lorebooks.map((b) => b.source).filter(Boolean) as string[]
  )
  const shown = onlyImported
    ? results.filter((n) => importedSources.has(`chub:${n.fullPath}`))
    : results

  // Tagline and description are separate fields on the listing; the tagline is
  // often just the head of the description, so drop it when it is.
  const blurb = [previewOf?.tagline?.trim(), previewOf?.description?.trim()]
    .filter((t): t is string => !!t)
    .filter(
      (t, _i, all) => !all.some((other) => other !== t && other.includes(t))
    )
    .join('\n\n')

  const sortLabel =
    CHUB_SORTS.find((s) => s.value === sortSelected)?.label ?? 'Popular'

  return (
    <div className="flex flex-col h-full w-full">
      <HeaderPage className="h-auto pt-1" bleed>
        <div
          className={cn(
            'pointer-events-none h-8 flex items-center relative z-20',
            HUB_COLUMN
          )}
        >
          <HubTabs className="pointer-events-auto" />
        </div>
      </HeaderPage>

      {/* Search and grid share one scroll box: with the search row outside it,
          the scrollbar narrowed the grid alone and the result count drifted
          out of line with the controls above it. */}
      <div className="flex-1 min-h-0 overflow-y-auto pb-6">
        <div className="sticky top-0 z-10 bg-neutral-50 dark:bg-background">
          <HubSearch
            value={searchValue}
            onChange={setSearchValue}
            placeholder="Search lorebooks on Chub.ai..."
            trailing={
              <>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-between shrink-0"
                    >
                      <span>{sortLabel}</span>
                      <ChevronsUpDown className="size-4 shrink-0 text-muted-foreground ml-2" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-44 max-h-80">
                    {CHUB_SORTS.map((s) => (
                      <DropdownMenuItem
                        key={s.value}
                        className={cn(
                          'cursor-pointer my-0.5',
                          sortSelected === s.value && 'bg-secondary-foreground/8'
                        )}
                        onClick={() => setSortSelected(s.value)}
                      >
                        {s.label}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
                <div className="flex items-center gap-2 shrink-0">
                  <Switch
                    checked={onlyImported}
                    onCheckedChange={setOnlyImported}
                  />
                  <span className="text-xs text-foreground font-medium whitespace-nowrap">
                    {t('hub:downloaded')}
                  </span>
                </div>
              </>
            }
            className="pt-2 pb-2"
          />
        </div>

        <div className={HUB_COLUMN}>
          <HubTitleRow
            className="mt-2 mb-3"
            title={
              debouncedQuery
                ? `Chub.ai results for “${debouncedQuery}”`
                : 'Lorebooks on Chub.ai'
            }
            picker={
              <ChubTagFilter
                filters={tagFilters}
                setFilters={setTagFilters}
                token={chubToken ?? undefined}
                namespace="lorebooks"
              />
            }
            chips={
              <ChubTagChips filters={tagFilters} setFilters={setTagFilters} />
            }
            count={
              count !== null && !error
                ? `${count} lorebook${count === 1 ? '' : 's'}`
                : undefined
            }
          />

          {importError && (
            <div className="mb-4 text-sm text-destructive">{importError}</div>
          )}

          {loading ? (
            <div className={CARD_GRID}>
              {[...Array(8)].map((_, i) => (
                <div
                  key={i}
                  className="rounded-lg border p-4 flex flex-col gap-3 animate-pulse"
                >
                  <div className="h-4 bg-muted rounded w-2/3" />
                  <div className="h-3 bg-muted rounded w-full" />
                  <div className="h-3 bg-muted rounded w-3/4" />
                </div>
              ))}
            </div>
          ) : error ? (
            <div className="py-16 text-center text-sm text-destructive">
              {error}
            </div>
          ) : shown.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 gap-3 text-center">
              <IconBook size={28} className="text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                {onlyImported && results.length > 0
                  ? 'None of these are in your library yet.'
                  : `No lorebooks found${debouncedQuery ? ` for “${debouncedQuery}”` : ''}.`}
              </p>
            </div>
          ) : (
            <div className={CARD_GRID}>
              {shown.map((node) => {
                const imported = importedSources.has(`chub:${node.fullPath}`)
                const importing = importingPath === node.fullPath
                return (
                <div
                  key={node.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => openPreview(node)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') openPreview(node)
                  }}
                  className="text-left rounded-lg border p-4 min-h-44 flex flex-col gap-2 hover:bg-secondary/40 cursor-pointer"
                >
                  <div className="flex items-start gap-2 min-w-0">
                    <div className="size-9 shrink-0 flex items-center justify-center bg-secondary dark:bg-secondary/40 rounded-lg">
                      <IconBook size={18} className="text-muted-foreground" />
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <span className="text-sm font-medium truncate">
                        {node.name}
                      </span>
                      <span className="text-xs text-muted-foreground truncate">
                        {node.fullPath.split('/')[1]}
                      </span>
                    </div>
                    <IconChevronRight
                      size={14}
                      className="shrink-0 text-muted-foreground mt-1"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground line-clamp-2">
                    {node.tagline || node.description || 'No description.'}
                  </p>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground pt-1">
                    <span className="flex items-center gap-1">
                      <IconStar size={12} />
                      {node.starCount ?? 0}
                    </span>
                    {!!node.nTokens && <span>{node.nTokens} tokens</span>}
                  </div>
                  {/* Tags and the download share the last line: on its own row
                      the button pushed the text block up. */}
                  <div className="mt-auto flex items-end gap-2 pt-1">
                    <div className="flex flex-wrap gap-1 flex-1 min-w-0">
                      {node.topics?.slice(0, 3).map((topic) => (
                        <span
                          key={topic}
                          className="text-[11px] px-1.5 py-0.5 rounded-full bg-secondary text-muted-foreground truncate max-w-32"
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                    <Button
                      className="shrink-0"
                      variant={imported ? 'ghost' : 'outline'}
                      size="sm"
                      disabled={imported || importing}
                      onClick={(e) => {
                        e.stopPropagation()
                        handleChubImport(node)
                      }}
                    >
                      {imported
                        ? '✓ In library'
                        : importing
                          ? 'Downloading...'
                          : 'Download'}
                    </Button>
                  </div>
                </div>
                )
              })}
            </div>
          )}

          {!loading && !error && results.length > 0 && (
            <>
              {onlyImported ? (
                canLoadMore && (
                  <div className="flex justify-center py-4">
                    <Button variant="outline" size="sm" onClick={loadMore}>
                      Load more
                    </Button>
                  </div>
                )
              ) : (
                <div ref={sentinelRef} className="h-px" />
              )}
              {loadingMore && (
                <div className="flex justify-center py-4">
                  <span className="text-xs text-muted-foreground">
                    Loading more…
                  </span>
                </div>
              )}
              {loadMoreError && (
                <div className="flex flex-col items-center gap-2 py-4">
                  <p className="text-sm text-destructive">{loadMoreError}</p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setLoadMoreError(null)}
                  >
                    Retry
                  </Button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <Sheet
        open={!!previewOf}
        onOpenChange={(open) => {
          if (!open) setPreviewOf(null)
        }}
      >
        <SheetContent className="sm:max-w-lg flex flex-col">
          <SheetHeader>
            <SheetTitle>{previewOf?.name}</SheetTitle>
            <SheetDescription>
              {preview
                ? `${preview.entries.length} entries`
                : previewError
                  ? 'Could not load this lorebook.'
                  : 'Loading entries…'}
            </SheetDescription>
          </SheetHeader>

          <div className="flex-1 overflow-y-auto px-4 pb-4 flex flex-col gap-3">
            {previewError && (
              <p className="text-sm text-destructive">{previewError}</p>
            )}

            {/* The book file itself carries no blurb, so the listing's is the
                only description this book will ever have. */}
            {blurb && (
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                {blurb}
              </p>
            )}
            {!!previewOf?.topics?.length && (
              <div className="flex flex-wrap gap-1">
                {previewOf.topics.map((topic) => (
                  <span
                    key={topic}
                    className="text-[11px] px-1.5 py-0.5 rounded-full bg-secondary text-muted-foreground"
                  >
                    {topic}
                  </span>
                ))}
              </div>
            )}
            {preview?.entries.map((entry) => (
              <div
                key={entry.id}
                className="rounded-md border p-3 flex flex-col gap-1.5"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium truncate">
                    {entry.comment || entry.keys[0] || 'Untitled entry'}
                  </span>
                  <span className="text-[11px] text-muted-foreground shrink-0">
                    {POSITION_LABELS[entry.position] ?? 'Unknown position'}
                    {entry.position === LorebookPosition.atDepth &&
                      ` ${entry.depth}`}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {entry.constant ? (
                    <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-blue-500/15 text-blue-600 dark:text-blue-400">
                      always on
                    </span>
                  ) : (
                    entry.keys.slice(0, 6).map((key) => (
                      <span
                        key={key}
                        className="text-[11px] px-1.5 py-0.5 rounded-full bg-secondary text-muted-foreground truncate max-w-40"
                      >
                        {key}
                      </span>
                    ))
                  )}
                  {entry.keys.length > 6 && (
                    <span className="text-[11px] text-muted-foreground">
                      +{entry.keys.length - 6}
                    </span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground line-clamp-3 whitespace-pre-wrap">
                  {entry.content}
                </p>
              </div>
            ))}
          </div>

          <div className="border-t p-4 flex items-center justify-between gap-3">
            <p className="text-xs text-muted-foreground">
              {importedBook
                ? 'Already in your library.'
                : 'Adds a copy to your lorebook library.'}
            </p>
            <Button
              size="sm"
              disabled={!preview || !!importedBook}
              onClick={importPreview}
            >
              {importedBook ? 'Imported' : 'Import'}
            </Button>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  )
}

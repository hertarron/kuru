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
import { CharacterPreviewSheet } from '@/containers/CharacterPreviewSheet'
import {
  ChubTagChips,
  ChubTagFilter,
  type TagFilter,
} from '@/containers/ChubTagFilter'
import { IconChevronRight } from '@tabler/icons-react'
import { ChevronsUpDown } from 'lucide-react'
import { useCharacters } from '@/hooks/useCharacters'
import { adoptCardLorebook } from '@/hooks/useLorebooks'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import {
  searchChubCharacters,
  loadChubCharacter,
  type ChubCharacterResult,
} from '@/lib/chub'

const CHUB_SORTS = [
  { value: 'default', label: 'Popular' },
  { value: 'trending', label: 'Trending' },
  { value: 'star_count', label: 'Most starred' },
  { value: 'rating', label: 'Top rated' },
  { value: 'last_activity_at', label: 'Recently updated' },
] as const

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.hub.characters as any)({
  component: CharactersContent,
})

const CHUB_IMPORTED_KEY = 'chub-imported-paths'

function getImportedPaths(): Set<string> {
  try {
    const raw = localStorage.getItem(CHUB_IMPORTED_KEY)
    return new Set(raw ? (JSON.parse(raw) as string[]) : [])
  } catch {
    return new Set()
  }
}

/**
 * Character discovery — the download surface for characters, following the
 * hub's model: browse/search Chub.ai here and import; the installed library
 * itself is managed in Settings > Characters.
 */
function CharactersContent() {
  const { t } = useTranslation()
  const { addCharacter } = useCharacters()
  const chubToken = useGeneralSetting((s) => s.chubToken)
  const [searchValue, setSearchValue] = useState('')
  const [importError, setImportError] = useState<string | null>(null)

  const [previewNode, setPreviewNode] = useState<ChubCharacterResult | null>(
    null
  )
  const [chubResults, setChubResults] = useState<ChubCharacterResult[]>([])
  const [chubCount, setChubCount] = useState<number | null>(null)
  const [chubLoading, setChubLoading] = useState(true)
  const [chubError, setChubError] = useState<string | null>(null)
  const [chubPage, setChubPage] = useState(1)
  const [loadingMore, setLoadingMore] = useState(false)
  const [loadMoreError, setLoadMoreError] = useState<string | null>(null)
  const [sortSelected, setSortSelected] = useState<string>('default')
  const [importedPaths, setImportedPaths] =
    useState<Set<string>>(getImportedPaths)
  const [importingPath, setImportingPath] = useState<string | null>(null)
  const [onlyImported, setOnlyImported] = useState(false)
  // Cards whose topic chips are fully expanded (Chub caps listings at ~10).
  const [expandedTags, setExpandedTags] = useState<Set<number>>(new Set())

  const toggleTagExpand = (id: number) =>
    setExpandedTags((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })

  // Debounced query — empty means the default listing.
  const [debouncedQuery, setDebouncedQuery] = useState('')
  useEffect(() => {
    const handler = setTimeout(() => setDebouncedQuery(searchValue.trim()), 300)
    return () => clearTimeout(handler)
  }, [searchValue])

  // Tag filters: green = include, red = exclude (Chub `tags` / `excluded_tags`).
  const [tagFilters, setTagFilters] = useState<TagFilter[]>([])

  useEffect(() => {
    let cancelled = false
    setChubLoading(true)
    setChubError(null)
    setLoadMoreError(null)
    setChubPage(1)
    searchChubCharacters(debouncedQuery, {
      token: chubToken ?? undefined,
      sort: sortSelected,
      page: 1,
      tags: tagFilters.filter((t) => t.mode === 'include').map((t) => t.tag),
      excludedTags: tagFilters
        .filter((t) => t.mode === 'exclude')
        .map((t) => t.tag),
    })
      .then((data) => {
        if (cancelled) return
        setChubResults(data.nodes)
        setChubCount(data.count)
      })
      .catch((e) => {
        if (cancelled) return
        setChubResults([])
        setChubCount(null)
        setChubError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setChubLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [debouncedQuery, chubToken, sortSelected, tagFilters])

  // Infinite scroll: fetch the next page when the sentinel below the grid
  // approaches the viewport. Appends are deduped by id — Chub's sort can
  // shift pages under us while scrolling.
  const canLoadMore =
    !chubLoading &&
    !chubError &&
    !loadMoreError &&
    !loadingMore &&
    chubCount !== null &&
    chubCount > 0 &&
    chubResults.length > 0 &&
    chubResults.length < chubCount

  const loadMore = useCallback(() => {
    if (!canLoadMore) return
    setLoadingMore(true)
    const nextPage = chubPage + 1
    searchChubCharacters(debouncedQuery, {
      token: chubToken ?? undefined,
      sort: sortSelected,
      page: nextPage,
      tags: tagFilters.filter((t) => t.mode === 'include').map((t) => t.tag),
      excludedTags: tagFilters
        .filter((t) => t.mode === 'exclude')
        .map((t) => t.tag),
    })
      .then((data) => {
        setChubResults((prev) => {
          const seen = new Set(prev.map((n) => n.id))
          return [...prev, ...data.nodes.filter((n) => !seen.has(n.id))]
        })
        setChubPage(nextPage)
        setChubCount(data.count)
      })
      .catch((e) => {
        setLoadMoreError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => setLoadingMore(false))
  }, [
    canLoadMore,
    chubPage,
    debouncedQuery,
    chubToken,
    sortSelected,
    tagFilters,
  ])

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

  // Chub has no "already imported" query, so this filters what has been
  // fetched rather than the catalog: with the toggle on, paging further can
  // still turn up more matches.
  const shownResults = onlyImported
    ? chubResults.filter((n) => importedPaths.has(n.fullPath))
    : chubResults

  const clearSearch = () => {
    setSearchValue('')
    setTagFilters([])
  }

  // Card topic chips feed the same include-filter set as the Tags picker;
  // already-filtered tags are no-ops so re-clicking can't duplicate entries.
  const addTagFilter = (tag: string) =>
    setTagFilters((prev) =>
      prev.some((t) => t.tag.toLowerCase() === tag.toLowerCase())
        ? prev
        : [...prev, { tag, mode: 'include' as const }]
    )

  const markImported = (fullPath: string) => {
    setImportedPaths((prev) => {
      const next = new Set(prev).add(fullPath)
      try {
        localStorage.setItem(CHUB_IMPORTED_KEY, JSON.stringify([...next]))
      } catch {
        // Persistence is best-effort; the in-session set still works.
      }
      return next
    })
  }

  const handleChubImport = async (node: ChubCharacterResult) => {
    setImportingPath(node.fullPath)
    setImportError(null)
    try {
      const { character, lorebook } = await loadChubCharacter(node)
      if (lorebook) character.lorebookIds = [adoptCardLorebook(lorebook)]
      addCharacter(character)
      markImported(node.fullPath)
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e))
    } finally {
      setImportingPath(null)
    }
  }

  return (
    <div className="flex flex-col h-full w-full">
      <HeaderPage className="h-auto pt-1" bleed>
        {/* The strip stays window-draggable; only the tabs opt back into
            clicks. Same column as the cards below, so they line up. */}
        <div
          className={cn(
            'pointer-events-none flex items-center h-8 relative z-20',
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
            onClear={clearSearch}
            placeholder="Search characters on Chub.ai..."
            trailing={
              <>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      className="justify-between shrink-0"
                    >
                      <span>
                        {CHUB_SORTS.find((s) => s.value === sortSelected)
                          ?.label ?? 'Sort'}
                      </span>
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
          {importError && (
            <div className="mb-4 text-sm text-destructive">{importError}</div>
          )}

          <HubTitleRow
            className="mt-2 mb-3"
            title={
              debouncedQuery
                ? `Chub.ai results for “${debouncedQuery}”`
                : 'Characters on Chub.ai'
            }
            picker={
              <ChubTagFilter
                filters={tagFilters}
                setFilters={setTagFilters}
                token={chubToken ?? undefined}
                namespace="characters"
              />
            }
            chips={
              <ChubTagChips filters={tagFilters} setFilters={setTagFilters} />
            }
            count={
              chubCount !== null && !chubError
                ? `${chubCount} character${chubCount === 1 ? '' : 's'}`
                : undefined
            }
          />

          {chubLoading ? (
            <div className={CARD_GRID}>
              {[...Array(8)].map((_, i) => (
                <div
                  key={i}
                  className="rounded-xl border bg-secondary/20 p-4 flex flex-col gap-3 animate-pulse"
                >
                  <div className="flex items-center gap-3">
                    <div className="size-11 rounded-lg bg-muted" />
                    <div className="h-4 bg-muted rounded w-2/3" />
                  </div>
                  <div className="h-3 bg-muted rounded w-full" />
                  <div className="h-3 bg-muted rounded w-3/4" />
                </div>
              ))}
            </div>
          ) : chubError ? (
            <div className="py-16 text-center text-sm text-destructive">
              {chubError}
            </div>
          ) : shownResults.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 gap-3 text-center">
              <span className="text-4xl">🎭</span>
              <p className="text-sm text-muted-foreground max-w-xs">
                {onlyImported && chubResults.length > 0
                  ? 'None of these are in your library yet.'
                  : `No Chub.ai characters found${debouncedQuery ? ` for “${debouncedQuery}”` : ''}.`}
              </p>
            </div>
          ) : (
            <div className={CARD_GRID}>
              {shownResults.map((node) => {
                const imported = importedPaths.has(node.fullPath)
                const importing = importingPath === node.fullPath
                const tagsExpanded = expandedTags.has(node.id)
                const visibleTopics =
                  node.topics && tagsExpanded
                    ? node.topics
                    : (node.topics?.slice(0, 3) ?? [])
                const hiddenTopicCount =
                  (node.topics?.length ?? 0) - visibleTopics.length
                return (
                  <div
                    key={node.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => setPreviewNode(node)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') setPreviewNode(node)
                    }}
                    className="relative rounded-xl border hover:bg-secondary/40 overflow-hidden p-4 min-h-44 flex flex-col gap-3 cursor-pointer text-left"
                  >
                    <IconChevronRight
                      size={14}
                      className="absolute top-4 right-4 shrink-0 text-muted-foreground pointer-events-none"
                    />
                    {/* Chub serves a webp thumbnail here; the full card PNG is
                        a megabyte and only worth fetching on preview. The mask
                        dissolves the art into the card rather than ending on a
                        seam, so it holds up on either hover background and in
                        both themes without matching a fixed color. The fade is
                        long enough that the text can start inside it and still
                        read against near-transparent art. */}
                    {node.avatar_url && (
                      <div className="pointer-events-none absolute inset-y-0 left-0 w-40 [mask-image:linear-gradient(to_right,#000_25%,transparent_95%)]">
                        <img
                          src={node.avatar_url}
                          alt=""
                          className="size-full object-cover object-top"
                          loading="lazy"
                        />
                      </div>
                    )}
                    <div className="relative flex flex-col flex-1 gap-1.5 pl-24">
                      <span className="text-sm font-medium truncate">
                        {node.name}
                      </span>
                      <span className="text-[10px] text-muted-foreground truncate">
                        {[
                          node.starCount != null
                            ? `★ ${node.starCount}`
                            : undefined,
                          node.nTokens != null
                            ? `${node.nTokens} tokens`
                            : undefined,
                          node.nMessages != null
                            ? `${node.nMessages} chats`
                            : undefined,
                        ]
                          .filter(Boolean)
                          .join(' · ')}
                      </span>
                      {(node.tagline || node.description) && (
                        <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed mt-1">
                          {node.tagline || node.description}
                        </p>
                      )}
                      {/* Tags and the download share the last line: on its own
                          row the button pushed the text block up. */}
                      <div className="mt-auto flex items-end gap-2 pt-1">
                        <div className="flex flex-wrap gap-1 flex-1 min-w-0">
                          {visibleTopics.map((topic) => (
                            <button
                              key={topic}
                              title={`Filter by “${topic}”`}
                              onClick={(e) => {
                                e.stopPropagation()
                                addTagFilter(topic)
                              }}
                              className={cn(
                                'text-[10px] px-1.5 py-0.5 rounded-full truncate max-w-28 cursor-pointer',
                                tagFilters.some(
                                  (t) =>
                                    t.tag.toLowerCase() === topic.toLowerCase()
                                )
                                  ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                                  : 'bg-foreground/8 text-muted-foreground hover:bg-emerald-500/15 hover:text-emerald-600'
                              )}
                            >
                              {topic}
                            </button>
                          ))}
                          {(hiddenTopicCount > 0 || tagsExpanded) &&
                            (node.topics?.length ?? 0) > 3 && (
                              <button
                                title={
                                  tagsExpanded
                                    ? 'Show fewer tags'
                                    : `Show ${hiddenTopicCount} more tag${
                                        hiddenTopicCount === 1 ? '' : 's'
                                      }`
                                }
                                onClick={(e) => {
                                  e.stopPropagation()
                                  toggleTagExpand(node.id)
                                }}
                                className="text-[10px] px-1.5 py-0.5 rounded-full cursor-pointer bg-foreground/8 text-muted-foreground hover:bg-foreground/15 hover:text-foreground"
                              >
                                {tagsExpanded ? 'less' : `+${hiddenTopicCount}`}
                              </button>
                            )}
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
                  </div>
                )
              })}
            </div>
          )}

          {!chubLoading && !chubError && chubResults.length > 0 && (
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

      <CharacterPreviewSheet
        source={previewNode ? { kind: 'chub', node: previewNode } : null}
        onOpenChange={(open) => {
          if (!open) setPreviewNode(null)
        }}
        onImport={handleChubImport}
        importState={(fullPath) =>
          importedPaths.has(fullPath)
            ? 'imported'
            : importingPath === fullPath
              ? 'importing'
              : 'idle'
        }
      />
    </div>
  )
}

export default CharactersContent

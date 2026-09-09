/* eslint-disable @typescript-eslint/no-explicit-any */
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { useModelSources } from '@/hooks/useModelSources'
import { cn, sanitizeModelId } from '@/lib/utils'
import { isSpecSidecar } from '@/lib/specDraft'
import {
  useState,
  useMemo,
  useEffect,
  useCallback,
  useRef,
  useTransition,
} from 'react'
import { useModelProvider } from '@/hooks/useModelProvider'
import { Switch } from '@/components/ui/switch'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useServiceHub } from '@/hooks/useServiceHub'
import type { CatalogModel } from '@/services/models/types'
import HeaderPage from '@/containers/HeaderPage'
import { ModelCard } from '@/containers/ModelCard'
import { ModelPreviewSheet } from '@/containers/ModelPreviewSheet'
import { HubTitleRow } from '@/containers/HubTitleRow'
import { HubFacetFilter, type Facet } from '@/containers/HubFacetFilter'
import { ChubTagChips, type TagFilter } from '@/containers/ChubTagFilter'
import { HubTabs } from '@/containers/HubTabs'
import { HubSearch } from '@/containers/HubSearch'
import { HUB_COLUMN, CARD_GRID } from '@/constants/layout'
import { ChevronsUpDown, Loader } from 'lucide-react'
import { useTranslation } from '@/i18n/react-i18next-compat'
import Fuse from 'fuse.js'
import {
  cleanHubSearchQuery,
  prioritizeExactModelMatches,
} from './searchRanking'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import { useShallow } from 'zustand/shallow'
import { Button } from '@/components/ui/button'

type SearchParams = {
  repo: string
}

/** Cards added to the DOM per page of the local catalog. */
const PAGE_SIZE = 60

/**
 * What the models tab filters on instead of tags. Each is derived from the
 * catalog entry, so the vocabulary is fixed rather than searched.
 */
const MODEL_FACET_TESTS: Record<string, (model: CatalogModel) => boolean> = {
  Tools: (m) => !!m.tools,
  Multimodal: (m) => (m.num_mmproj ?? 0) > 0,
  MLX: (m) => !!m.is_mlx,
  GGUF: (m) => !m.is_mlx,
}

const MODEL_FACETS: Facet[] = [
  { id: 'Tools', label: 'Tools', description: 'Supports tool calling' },
  {
    id: 'Multimodal',
    label: 'Multimodal',
    description: 'Ships a vision projector',
  },
  { id: 'GGUF', label: 'GGUF', description: 'Runs on llama.cpp' },
  { id: 'MLX', label: 'MLX', description: 'Apple Silicon only' },
]

export const Route = createFileRoute(route.hub.index as any)({
  component: HubContent,
  validateSearch: (search: Record<string, unknown>): SearchParams => ({
    repo: search.repo as SearchParams['repo'],
  }),
})

function HubContent() {
  const [isPending, startTransition] = useTransition()
  const huggingfaceToken = useGeneralSetting((state) => state.huggingfaceToken)
  const serviceHub = useServiceHub()

  const { t } = useTranslation()

  const sortOptions = [
    { value: 'newest', name: t('hub:sortNewest') },
    { value: 'most-downloaded', name: t('hub:sortMostDownloaded') },
    ...(IS_MACOS
      ? [
          { value: 'mlx', name: 'MLX' },
          { value: 'gguf', name: 'GGUF' },
        ]
      : []),
  ]
  const searchOptions = useMemo(
    () => ({
      includeScore: true,
      // Tighter than the 0.6 default so a precise query (e.g. a full HF
      // namespace) stops surfacing loosely-related catalog models.
      threshold: 0.3,
      // Repo ids are long; without this Fuse penalizes matches far from the
      // string start and misses substrings.
      ignoreLocation: true,
      keys: ['model_name', 'developer', 'quants.model_id'],
    }),
    []
  )

  const { sources, fetchSources, loading } = useModelSources(
    useShallow((state) => ({
      sources: state.sources,
      fetchSources: state.fetchSources,
      loading: state.loading,
    }))
  )

  const [searchValue, setSearchValue] = useState('')
  const [sortSelected, setSortSelected] = useState('newest')
  const [isSearching, setIsSearching] = useState(false)
  const [showOnlyDownloaded, setShowOnlyDownloaded] = useState(false)
  const [facetFilters, setFacetFilters] = useState<TagFilter[]>([])
  const [huggingFaceRepo, setHuggingFaceRepo] = useState<CatalogModel | null>(
    null
  )
  const [isInitialLoad, setIsInitialLoad] = useState(true)
  const addModelSourceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  )

  // Sorting functionality
  const sortedModels = useMemo(() => {
    let sorted = [...sources]

    // Apply MLX/GGUF filter first (only on Mac)
    if (sortSelected === 'mlx') {
      sorted = sorted.filter((m) => m.is_mlx)
    } else if (sortSelected === 'gguf') {
      sorted = sorted.filter((m) => !m.is_mlx)
    }

    // Apply sorting
    if (sortSelected === 'most-downloaded') {
      return sorted.sort((a, b) => (b.downloads || 0) - (a.downloads || 0))
    }
    return sorted.sort(
      (a, b) =>
        new Date(b.created_at || 0).getTime() -
        new Date(a.created_at || 0).getTime()
    )
  }, [sortSelected, sources])

  // Filtered models (debounced search)
  const [debouncedSearchValue, setDebouncedSearchValue] = useState(searchValue)

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedSearchValue(searchValue)
    }, 300)
    return () => clearTimeout(handler)
  }, [searchValue])

  const filteredModels = useMemo(() => {
    // Speculative draft companions (mtp/eagle3/dflash/dspark) are draft models,
    // not standalone variants — move them out of `quants` (so they don't show
    // as downloadable) into `specQuants`, where DownloadButton resolves them
    // against the chosen quant.
    let filtered: CatalogModel[] = sortedModels.map((model) => ({
      ...model,
      quants: model.quants?.filter((q) => !isSpecSidecar(q)),
      specQuants: model.quants?.filter((q) => isSpecSidecar(q)),
    }))
    // Apply search filter
    if (debouncedSearchValue.length) {
      const fuse = new Fuse(filtered, searchOptions)
      // Remove domain from search value (e.g., "huggingface.co/author/model" -> "author/model")
      const cleanedSearchValue = cleanHubSearchQuery(debouncedSearchValue)
      // Fuse scores fuzzy relevance; re-rank so an exact HF id/name is first
      // (see #8447 — precise queries were not always top of list).
      filtered = prioritizeExactModelMatches(
        fuse.search(cleanedSearchValue).map((result) => result.item),
        cleanedSearchValue
      )
    }
    // The catalog has no tags, so the facets are read off the entry itself.
    for (const { tag, mode } of facetFilters) {
      const has = MODEL_FACET_TESTS[tag]
      if (!has) continue
      filtered = filtered.filter((model) =>
        mode === 'include' ? has(model) : !has(model)
      )
    }

    // Apply downloaded filter
    if (showOnlyDownloaded) {
      filtered = filtered
        ?.map((model) => ({
          ...model,
          quants: model.quants?.filter((variant) => {
            // Check both direct match and with developer prefix (like DownloadButton does)
            const isLlamaCppDownloaded = useModelProvider
              .getState()
              .getProviderByName('llamacpp')
              ?.models.some(
                (m: { id: string }) =>
                  m.id === variant.model_id ||
                  m.id ===
                    `${model.developer}/${sanitizeModelId(variant.model_id)}`
              )

            const isMlxDownloaded = useModelProvider
              .getState()
              .getProviderByName('mlx')
              ?.models.some(
                (m: { id: string }) =>
                  m.id === variant.model_id ||
                  m.id ===
                    `${model.developer}/${sanitizeModelId(variant.model_id)}`
              )

            return isLlamaCppDownloaded || isMlxDownloaded
          }),
        }))
        .filter((model) => (model.quants?.length ?? 0) > 0)
    }
    // Add HuggingFace repo at the beginning if available
    if (huggingFaceRepo) {
      filtered = [huggingFaceRepo, ...filtered]
    }
    return filtered
  }, [
    sortedModels,
    debouncedSearchValue,
    showOnlyDownloaded,
    facetFilters,
    huggingFaceRepo,
    searchOptions,
  ])

  // The catalog is filtered in memory, so paging here is only about DOM size:
  // 300+ cards at once is a lot of nodes for a list nobody scrolls to the end
  // of. Grows as the sentinel below the grid comes into view.
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)
  useEffect(() => {
    setVisibleCount(PAGE_SIZE)
  }, [filteredModels])
  const visibleModels = useMemo(
    () => filteredModels.slice(0, visibleCount),
    [filteredModels, visibleCount]
  )

  const sentinelRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const el = sentinelRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setVisibleCount((count) => count + PAGE_SIZE)
        }
      },
      { rootMargin: '600px' }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [visibleModels.length])

  const [previewModel, setPreviewModel] = useState<CatalogModel | null>(null)

  useEffect(() => {
    // Use startTransition to keep UI responsive during data fetch
    startTransition(() => {
      fetchSources()
    })
  }, [fetchSources])

  // Reset initial load state after data loads or on filter change
  useEffect(() => {
    if (!isInitialLoad) return

    // Hide skeleton after a short delay to show loading state
    const timer = setTimeout(() => setIsInitialLoad(false), 150)
    return () => clearTimeout(timer)
  }, [isInitialLoad, filteredModels.length])

  const fetchHuggingFaceModel = async (searchValue: string) => {
    if (
      !searchValue.length ||
      (!searchValue.includes('/') && !searchValue.startsWith('http'))
    ) {
      return
    }

    setIsSearching(true)
    if (addModelSourceTimeoutRef.current) {
      clearTimeout(addModelSourceTimeoutRef.current)
    }

    addModelSourceTimeoutRef.current = setTimeout(async () => {
      try {
        const repoInfo = await serviceHub
          .models()
          .fetchHuggingFaceRepo(searchValue, huggingfaceToken)
        if (repoInfo) {
          const catalogModel = serviceHub
            .models()
            .convertHfRepoToCatalogModel(repoInfo)
          if (
            !sources.some(
              (s) =>
                catalogModel.model_name.trim().split('/').pop() ===
                  s.model_name.trim() &&
                catalogModel.developer?.trim() === s.developer?.trim()
            )
          ) {
            setHuggingFaceRepo(catalogModel)
          }
        }
      } catch (error) {
        console.error('Error fetching repository info:', error)
      } finally {
        setIsSearching(false)
      }
    }, 500)
  }

  const handleSearchChange = (value: string) => {
    setIsSearching(false)
    setSearchValue(value)
    setHuggingFaceRepo(null) // Clear previous repo info

    if (!showOnlyDownloaded) {
      fetchHuggingFaceModel(value)
    }
  }

  const navigate = useNavigate()

  const handleUseModel = useCallback(
    (modelId: string) => {
      navigate({
        to: route.home,
        params: {},
        search: {
          threadModel: {
            id: modelId,
            provider: 'llamacpp',
          },
        },
      })
    },
    [navigate]
  )

  const renderFilter = () => {
    return (
      <>
        {/* Sort dropdown - always visible */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm">
              {
                sortOptions.find((option) => option.value === sortSelected)
                  ?.name
              }
              <ChevronsUpDown className="size-4 shrink-0 text-muted-foreground ml-2" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="bottom" align="end">
            {sortOptions.map((option) => (
              <DropdownMenuItem
                className={cn(
                  'cursor-pointer my-0.5',
                  sortSelected === option.value && 'bg-secondary'
                )}
                key={option.value}
                onClick={() => {
                  setIsInitialLoad(true)
                  setSortSelected(option.value)
                }}
              >
                {option.name}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <div className="flex items-center gap-2">
          <Switch
            checked={showOnlyDownloaded}
            onCheckedChange={(checked) => {
              setIsInitialLoad(true)
              setShowOnlyDownloaded(checked)
              if (checked) {
                setHuggingFaceRepo(null)
              } else {
                // Re-trigger HuggingFace search when switching back to "All models"
                fetchHuggingFaceModel(searchValue)
              }
            }}
          />
          <span className="text-xs text-foreground font-medium whitespace-nowrap">
            {t('hub:downloaded')}
          </span>
        </div>
      </>
    )
  }

  return (
    <div className="flex flex-col h-full w-full">
      <div className="flex flex-col h-full w-full ">
        <HeaderPage className="h-auto pt-1" bleed>
          {/* pointer-events-none lets the blank middle fall through to the
              window drag strip; only the tabs opt back in. Same column as the
              cards below, so they line up. */}
          <div
            className={cn(
              'pointer-events-none h-8 flex items-center relative z-20',
              HUB_COLUMN
            )}
          >
            <HubTabs className="pointer-events-auto" />
          </div>
        </HeaderPage>
        {/* Search and grid share one scroll box: with the search row outside
            it, the scrollbar narrowed the grid alone and the result count
            drifted out of line with the controls above it. */}
        <div className="flex-1 min-h-0 pb-6 overflow-y-auto! first-step-setup-local-provider">
          <div className="sticky top-0 z-10 bg-neutral-50 dark:bg-background">
            <HubSearch
              value={searchValue}
              onChange={handleSearchChange}
              placeholder={t('hub:searchPlaceholder')}
              leading={
                isSearching ? (
                  <Loader className="shrink-0 size-4 animate-spin text-muted-foreground" />
                ) : undefined
              }
              trailing={
                <div className="sm:flex items-center gap-2 shrink-0 hidden">
                  {renderFilter()}
                </div>
              }
              className="pt-2 pb-2"
            />
          </div>
          <div className={HUB_COLUMN}>
            <div className="flex flex-col gap-3 w-full">
              <HubTitleRow
                className="mt-2 -mb-1"
                title={
                  debouncedSearchValue
                    ? `Results for “${debouncedSearchValue}”`
                    : 'Models'
                }
                picker={
                  <HubFacetFilter
                    facets={MODEL_FACETS.filter(
                      (f) => f.id !== 'MLX' || IS_MACOS
                    )}
                    filters={facetFilters}
                    setFilters={setFacetFilters}
                  />
                }
                chips={
                  <ChubTagChips
                    filters={facetFilters}
                    setFilters={setFacetFilters}
                  />
                }
                count={
                  filteredModels.length
                    ? `${filteredModels.length} model${filteredModels.length === 1 ? '' : 's'}`
                    : undefined
                }
              />
              {isInitialLoad || (loading && !filteredModels.length) ? (
                <div className={CARD_GRID}>
                  {[...Array(8)].map((_, i) => (
                    <div
                      key={i}
                      className="rounded-xl border p-4 min-h-44 flex flex-col gap-3 animate-pulse"
                    >
                      <div className="h-4 bg-muted rounded w-2/3" />
                      <div className="h-3 bg-muted rounded w-full" />
                      <div className="h-3 bg-muted rounded w-3/4" />
                      <div className="mt-auto h-8 bg-muted rounded w-24 self-end" />
                    </div>
                  ))}
                </div>
              ) : filteredModels.length === 0 ? (
                <div className="flex items-center justify-center py-24">
                  <div className="text-center text-muted-foreground">
                    {t('hub:noModels')}
                  </div>
                </div>
              ) : (
                <div
                  className={cn(
                    'flex flex-col gap-3 pb-2 mb-2 transition-opacity duration-200',
                    isPending ? 'opacity-70' : 'opacity-100'
                  )}
                >
                  <div className="flex items-center gap-2 justify-end sm:hidden">
                    {renderFilter()}
                  </div>
                  <div className={CARD_GRID}>
                    {visibleModels.map((model) => (
                      <ModelCard
                        key={model.model_name}
                        model={model}
                        onOpen={setPreviewModel}
                        onUseModel={handleUseModel}
                      />
                    ))}
                  </div>
                  {/* The catalog is already in memory, so "loading more" is only
                    about how much of it is in the DOM at once. */}
                  {visibleModels.length < filteredModels.length && (
                    <div ref={sentinelRef} className="h-px" />
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <ModelPreviewSheet
        model={previewModel}
        onOpenChange={(open) => {
          if (!open) setPreviewModel(null)
        }}
      />
    </div>
  )
}

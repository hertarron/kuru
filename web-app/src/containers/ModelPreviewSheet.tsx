import { useEffect, useMemo, useState } from 'react'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { CardItem } from '@/containers/Card'
import { RenderMarkdown } from '@/containers/RenderMarkdown'
import { ModelInfoHoverCard } from '@/containers/ModelInfoHoverCard'
import { ModelDownloadAction } from '@/containers/ModelDownloadAction'
import { MlxModelDownloadAction } from '@/containers/MlxModelDownloadAction'
import {
  IconDownload,
  IconExternalLink,
  IconEye,
  IconFileCode,
  IconTool,
} from '@tabler/icons-react'
import { DEFAULT_MODEL_QUANTIZATIONS } from '@/constants/models'
import { useRecommendedQuant } from '@/hooks/useRecommendedQuant'
import { extractDescription, extractModelName } from '@/lib/models'
import { cleanModelReadme, huggingFaceRepoUrl } from '@/lib/model-readme'
import { getQuantTier } from '@/lib/quant-tier'
import { isSpecSidecar } from '@/lib/specDraft'
import { formatBytes, cn } from '@/lib/utils'
import { sumMlxModelBytes } from '@/lib/modelCompatibility'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import { useTranslation } from '@/i18n/react-i18next-compat'
import type { CatalogModel } from '@/services/models/types'

/**
 * Everything the old /hub/$modelId page showed, in a sheet over the grid.
 *
 * A page meant leaving the Hub and coming back to the top of a virtualized
 * list; the sheet leaves the grid mounted, so closing it puts the reader back
 * exactly where they were.
 */
export function ModelPreviewSheet({
  model,
  onOpenChange,
}: {
  model: CatalogModel | null
  onOpenChange: (open: boolean) => void
}) {
  const { t } = useTranslation()
  const huggingfaceToken = useGeneralSetting((s) => s.huggingfaceToken)
  const [readme, setReadme] = useState<string | null>(null)
  const [readmeState, setReadmeState] = useState<'idle' | 'loading' | 'error'>(
    'idle'
  )

  const readmeUrl = model?.readme

  // Fetched on open rather than with the catalog: it is one request per model
  // and most models are never opened. The cancelled flag keeps a slow response
  // from overwriting a model the reader has since moved on from.
  useEffect(() => {
    setReadme(null)
    if (!readmeUrl) {
      setReadmeState('idle')
      return
    }
    let cancelled = false
    setReadmeState('loading')
    const withToken = huggingfaceToken
      ? { headers: { Authorization: `Bearer ${huggingfaceToken}` } }
      : undefined
    fetch(readmeUrl)
      .then((response) =>
        // A gated repo 401s anonymously but may open with the user's token.
        !response.ok && withToken ? fetch(readmeUrl, withToken) : response
      )
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status))
        return response.text()
      })
      .then((text) => {
        if (cancelled) return
        setReadme(cleanModelReadme(text))
        setReadmeState('idle')
      })
      .catch(() => {
        if (!cancelled) setReadmeState('error')
      })
    return () => {
      cancelled = true
    }
  }, [readmeUrl, huggingfaceToken])

  // MTP companions are draft models paired with a real quant at download time,
  // not something to pick on their own.
  const quants = useMemo(
    () => model?.quants?.filter((q) => !isSpecSidecar(q)) ?? [],
    [model]
  )
  const recommended = useRecommendedQuant(model)
  const recommendedId = recommended?.model_id

  const size = model?.is_mlx
    ? formatBytes(sumMlxModelBytes(model) || undefined)
    : recommended?.file_size

  const repoUrl = huggingFaceRepoUrl(
    model?.readme,
    model?.developer,
    model?.model_name
  )

  const stats = [
    model?.developer ? `${t('hub:by')} ${model.developer}` : undefined,
    size,
    model?.is_mlx ? 'MLX' : undefined,
  ].filter(Boolean)

  return (
    <Sheet
      open={!!model}
      onOpenChange={(open) => {
        if (!open) onOpenChange(false)
      }}
    >
      <SheetContent className="sm:max-w-2xl flex flex-col p-0">
        <SheetHeader className="px-4">
          <div className="flex items-start gap-2 pr-6">
            <SheetTitle className="truncate capitalize flex-1 min-w-0">
              {extractModelName(model?.model_name ?? '') || model?.model_name}
            </SheetTitle>
            {repoUrl && (
              <a
                href={repoUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="shrink-0"
              >
                <Button variant="outline" size="sm">
                  <IconExternalLink size={14} />
                  Hugging Face
                </Button>
              </a>
            )}
          </div>
          <SheetDescription className="truncate">
            {stats.join(' · ')}
          </SheetDescription>
        </SheetHeader>

        {model && (
          <div className="flex-1 overflow-y-auto px-4 pb-4 flex flex-col gap-5">
            <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <IconDownload size={14} />
                {model.downloads || 0}
              </span>
              {!model.is_mlx && (
                <span className="flex items-center gap-1">
                  <IconFileCode size={14} />
                  {quants.length} {t('hub:variants')}
                </span>
              )}
              {(model.num_mmproj ?? 0) > 0 && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-secondary text-foreground/80">
                  <IconEye size={13} />
                  {t('multimodal')}
                </span>
              )}
              {model.tools && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-secondary text-foreground/80">
                  <IconTool size={13} />
                  {t('tools')}
                </span>
              )}
            </div>

            {!!extractDescription(model.description) && (
              <div className="text-sm text-muted-foreground leading-normal">
                <RenderMarkdown
                  className="reset-heading"
                  content={extractDescription(model.description) || ''}
                />
              </div>
            )}

            {quants.length > 0 && (
              <div className="flex flex-col">
                <h3 className="text-sm font-medium font-studio mb-1">
                  {t('hub:variants')}
                </h3>
                {quants.map((variant) => (
                  <CardItem
                    key={variant.model_id}
                    title={
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="break-all">{variant.model_id}</span>
                        {(() => {
                          const tier = getQuantTier(variant.model_id)
                          return tier ? (
                            <span
                              className={cn(
                                'text-xs font-medium px-1.5 py-0.5 rounded',
                                tier.className
                              )}
                            >
                              {tier.label}
                            </span>
                          ) : null
                        })()}
                        {variant.model_id === recommendedId && (
                          <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                            Recommended
                          </span>
                        )}
                      </div>
                    }
                    actions={
                      <div className="flex items-center gap-2">
                        <p className="text-muted-foreground font-medium text-xs">
                          {variant.file_size}
                        </p>
                        <ModelInfoHoverCard
                          model={model}
                          variant={variant}
                          defaultModelQuantizations={DEFAULT_MODEL_QUANTIZATIONS}
                        />
                        {model.is_mlx ? (
                          <MlxModelDownloadAction model={model} />
                        ) : (
                          <ModelDownloadAction variant={variant} model={model} />
                        )}
                      </div>
                    }
                  />
                ))}
              </div>
            )}

            {readmeUrl && (
              <div className="flex flex-col gap-2">
                <h3 className="text-sm font-medium font-studio">README</h3>
                {readmeState === 'loading' && (
                  <p className="text-sm text-muted-foreground">
                    Loading README…
                  </p>
                )}
                {readmeState === 'error' && (
                  <p className="text-sm text-muted-foreground">
                    Could not load the README for this model.
                  </p>
                )}
                {readme && (
                  <div className="text-sm text-muted-foreground leading-normal">
                    <RenderMarkdown className="reset-heading" content={readme} />
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </SheetContent>
    </Sheet>
  )
}

export default ModelPreviewSheet

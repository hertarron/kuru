import { useState } from 'react'
import {
  IconChevronRight,
  IconCpu,
  IconDownload,
  IconEye,
  IconFileCode,
  IconTool,
} from '@tabler/icons-react'
import { DownloadButtonPlaceholder } from '@/containers/DownloadButton'
import { MlxModelDownloadAction } from '@/containers/MlxModelDownloadAction'
import { ModelInfoHoverCard } from '@/containers/ModelInfoHoverCard'
import { DEFAULT_MODEL_QUANTIZATIONS } from '@/constants/models'
import { useRecommendedQuant } from '@/hooks/useRecommendedQuant'
import { extractDescription, extractModelName } from '@/lib/models'
import { getModelLogo } from '@/lib/model-developer-logo'
import { isMtpQuant } from '@/lib/mtp'
import { sumMlxModelBytes } from '@/lib/modelCompatibility'
import { formatBytes, cn } from '@/lib/utils'
import { useTranslation } from '@/i18n/react-i18next-compat'
import type { CatalogModel } from '@/services/models/types'

/** The model family's logo where one ships, and its initial where none does. */
function ModelMark({
  modelName,
  developer,
}: {
  modelName?: string
  developer?: string
}) {
  const logo = getModelLogo(modelName, developer)
  const [failed, setFailed] = useState(false)
  const initial = (modelName || developer || '').trim().charAt(0)

  return (
    <div className="size-9 shrink-0 flex items-center justify-center bg-secondary dark:bg-secondary/40 rounded-lg overflow-hidden">
      {logo && !failed ? (
        <img
          src={logo.src}
          alt=""
          className={cn(
            'size-5 object-contain',
            logo.lightArt && 'invert dark:invert-0'
          )}
          loading="lazy"
          draggable={false}
          onError={() => setFailed(true)}
        />
      ) : initial ? (
        <span className="text-sm font-medium uppercase text-muted-foreground">
          {initial}
        </span>
      ) : (
        <IconCpu size={18} className="text-muted-foreground" />
      )}
    </div>
  )
}

/**
 * One model in the Hub grid, laid out like the character and lorebook cards.
 * The card downloads the recommended quant; every other quant lives in the
 * sheet, which is what clicking the card opens.
 */
export function ModelCard({
  model,
  onOpen,
  onUseModel,
  className,
}: {
  model: CatalogModel
  onOpen: (model: CatalogModel) => void
  onUseModel: (modelId: string) => void
  className?: string
}) {
  const { t } = useTranslation()
  const quants = model.quants?.filter((q) => !isMtpQuant(q)) ?? []
  // Matches what the download button will fetch, so the size and the fit badge
  // describe the same file.
  const recommended = useRecommendedQuant(model)
  const size = model.is_mlx
    ? formatBytes(sumMlxModelBytes(model) || undefined)
    : recommended?.file_size

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onOpen(model)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') onOpen(model)
      }}
      className={cn(
        'relative rounded-lg border hover:bg-secondary/40 p-4 min-h-44 flex flex-col gap-2 cursor-pointer text-left',
        className
      )}
    >
      <div className="flex items-start gap-2 min-w-0">
        <ModelMark
          modelName={extractModelName(model.model_name) || model.model_name}
          developer={model.developer}
        />
        <div className="flex flex-col min-w-0 flex-1">
          <span
            className="text-sm font-medium capitalize truncate"
            title={model.model_name}
          >
            {extractModelName(model.model_name) || model.model_name}
          </span>
          {model.developer && (
            <span className="text-xs text-muted-foreground capitalize truncate">
              {model.developer}
            </span>
          )}
        </div>
        <div className="shrink-0 flex items-center gap-1.5 mt-0.5">
          <ModelInfoHoverCard
            model={model}
            defaultModelQuantizations={DEFAULT_MODEL_QUANTIZATIONS}
            variant={recommended}
            isDefaultVariant
          />
          <IconChevronRight
            size={14}
            className="shrink-0 text-muted-foreground pointer-events-none"
          />
        </div>
      </div>

      <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
        {extractDescription(model.description) || 'No description.'}
      </p>

      <div className="flex items-center gap-3 text-xs text-muted-foreground pt-1">
        {size && <span>{size}</span>}
        <span className="flex items-center gap-1">
          <IconDownload size={12} />
          {model.downloads || 0}
        </span>
        {!model.is_mlx && quants.length > 0 && (
          <span className="flex items-center gap-1" title={t('hub:variants')}>
            <IconFileCode size={12} />
            {quants.length}
          </span>
        )}
      </div>

      {/* Tags and the download share the last line: on its own row the button
          pushed the text block up and left a gap mid-card. */}
      <div className="mt-auto flex items-end gap-2 pt-1">
        <div className="flex flex-wrap gap-1 flex-1 min-w-0">
          {(model.num_mmproj ?? 0) > 0 && (
            <span className="inline-flex items-center gap-1 text-[11px] px-1.5 py-0.5 rounded-full bg-secondary text-muted-foreground">
              <IconEye size={11} />
              {t('multimodal')}
            </span>
          )}
          {model.tools && (
            <span className="inline-flex items-center gap-1 text-[11px] px-1.5 py-0.5 rounded-full bg-secondary text-muted-foreground">
              <IconTool size={11} />
              {t('tools')}
            </span>
          )}
          {model.is_mlx && (
            <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-secondary text-muted-foreground">
              MLX
            </span>
          )}
        </div>
        {/* The download buttons are their own controls; a click on one must not
            also open the sheet behind it. */}
        <div
          className="shrink-0"
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => e.stopPropagation()}
        >
          {model.is_mlx ? (
            <MlxModelDownloadAction model={model} />
          ) : (
            <DownloadButtonPlaceholder
              model={model}
              handleUseModel={onUseModel}
            />
          )}
        </div>
      </div>
    </div>
  )
}

export default ModelCard

import { useHardware } from '@/hooks/useHardware'
import {
  DEFAULT_CTX_LENGTH,
  estimateModelFit,
  parseFileSize,
} from '@/lib/modelCompatibility'
import { selectDefaultQuant } from '@/lib/models'
import { isMtpQuant } from '@/lib/mtp'
import { DEFAULT_MODEL_QUANTIZATIONS } from '@/constants/models'
import type { CatalogModel, ModelQuant } from '@/services/models/types'
import type { HardwareData } from '@/hooks/useHardware'

/**
 * The largest quant of a model that still fits comfortably.
 *
 * Within one model's quants, file size is quality: the biggest file that comes
 * back green is the best answer we can give without a table of quant formats
 * to maintain. Falling back down the list keeps the button useful on a machine
 * where nothing fits comfortably — the largest merely-tight quant, then the
 * smallest one there is.
 *
 * With no hardware reading (the estimator returns `unknown`), this defers to
 * the fixed preference list rather than guessing.
 */
export function pickBestFitQuant(
  model: CatalogModel,
  hardware: HardwareData,
  ctxLength: number = DEFAULT_CTX_LENGTH
): ModelQuant | undefined {
  const quants = model.quants?.filter((q) => !isMtpQuant(q)) ?? []
  if (quants.length === 0) return undefined

  const fallback = selectDefaultQuant(quants, DEFAULT_MODEL_QUANTIZATIONS)
  if (model.is_mlx) return fallback

  const sized = quants
    .map((quant) => ({ quant, bytes: parseFileSize(quant.file_size) }))
    .filter((q): q is { quant: ModelQuant; bytes: number } => q.bytes != null)
    .sort((a, b) => b.bytes - a.bytes)
  if (sized.length === 0) return fallback

  const tiers = sized.map((q) => ({
    ...q,
    tier: estimateModelFit(q.bytes, ctxLength, hardware),
  }))
  if (tiers.every((q) => q.tier === 'unknown')) return fallback

  return (
    tiers.find((q) => q.tier === 'green')?.quant ??
    tiers.find((q) => q.tier === 'yellow')?.quant ??
    tiers[tiers.length - 1].quant
  )
}

/** `pickBestFitQuant` against the machine this is running on. */
export function useRecommendedQuant(
  model: CatalogModel | null | undefined
): ModelQuant | undefined {
  const hardwareData = useHardware((s) => s.hardwareData)
  return model ? pickBestFitQuant(model, hardwareData) : undefined
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { useModelCalibration } from '@/hooks/useModelCalibration'
import { useModelProvider } from '@/hooks/useModelProvider'
import { runCalibrationProbe } from '@/lib/calibrationRunner'
import { useModelPlan } from '@/hooks/useModelPlan'
import { useServiceHub } from '@/hooks/useServiceHub'
import {
  autoBand,
  effectiveCacheType,
  effectiveFlashAttn,
  effectiveParallel,
  effectiveUbatch,
  MIN_USABLE_CONTEXT,
  placementArgs,
  resolveContext,
  type ContextBand,
} from '@/lib/contextPlanner'
import { cn } from '@/lib/utils'

/**
 * The cache types worth offering. llama.cpp accepts more, but the rest are
 * either the same size as one of these or too lossy to recommend.
 */
const KV_CACHE_TYPES = [
  { value: 'f16', label: 'fp16', hint: 'Full quality. Twice the memory of q8.' },
  { value: 'q8_0', label: 'q8', hint: 'Near-identical quality at half the size.' },
  { value: 'q5_1', label: 'q5_1', hint: 'Noticeably smaller. Slight quality cost.' },
  { value: 'iq4_nl', label: 'iq4_nl', hint: 'Smallest. Use when context matters more than quality.' },
] as const

/**
 * Prompt-processing presets. The micro-batch is the one that costs VRAM, so it
 * moves the context ceilings; the batch alone does not.
 */
const BATCH_PRESETS = [
  { label: 'Light', batch: 1024, ubatch: 256, hint: 'Leaves the most memory for context. Slowest on long prompts.' },
  { label: 'Balanced', batch: 2048, ubatch: 512, hint: "llama.cpp's default." },
  { label: 'Fast', batch: 2048, ubatch: 2048, hint: 'Reads long prompts fastest. Costs context.' },
] as const

const GiB = 1024 ** 3
const formatGiB = (bytes: number) => `${(bytes / GiB).toFixed(1)} GB`
const formatMiB = (bytes: number) => `${Math.round(bytes / 1024 ** 2)} MB`

type ContextPlanPanelProps = {
  model: Model
  /** True once the user has edited placement by hand in Advanced. */
  manual: boolean
  /** Applied as one update: context and placement must not land separately. */
  onChange: (patch: Record<string, string | number | boolean>) => void
  onResume: () => void
}

function settingValue(model: Model, key: string): unknown {
  return model.settings?.[key]?.controller_props?.value
}

export function ContextPlanPanel({
  model,
  manual,
  onChange,
  onResume,
}: ContextPlanPanelProps) {
  const kvCacheType = effectiveCacheType(
    settingValue(model, 'cache_type_k'),
    useModelProvider((s) =>
      s
        .getProviderByName('llamacpp')
        ?.settings?.find((x) => x.key === 'cache_type_k')
        ?.controller_props?.value
    )
  )
  const ubatch = effectiveUbatch(
    settingValue(model, 'ubatch_size'),
    useModelProvider((s) =>
      s
        .getProviderByName('llamacpp')
        ?.settings?.find((x) => x.key === 'ubatch_size')
        ?.controller_props?.value
    )
  )
  // Recurrent state and the probe floor both scale with the effective
  // parallel slots (per-model override, else global, else auto), so the
  // bands must read it rather than assume the default.
  const sequences = useModelProvider((s) =>
    effectiveParallel(
      settingValue(model, 'parallel'),
      s.getProviderByName('llamacpp')?.settings?.find((x) => x.key === 'parallel')
        ?.controller_props?.value
    )
  )
  const flashAttn = useModelProvider((s) =>
    effectiveFlashAttn(
      settingValue(model, 'flash_attn'),
      s.getProviderByName('llamacpp')?.settings?.find((x) => x.key === 'flash_attn')
        ?.controller_props?.value
    )
  )

  const devices = (settingValue(model, 'device') as string) || ''
  // Advanced can send the whole cache to RAM. It is not a planner-owned key,
  // so it does not flip the model to manual, but every band turns on where
  // the cache lives — a plan that ignores it reserves VRAM for a cache that
  // is never on the card.
  const noKvOffload = settingValue(model, 'no_kv_offload') === true
  // The vision projector follows its own offload flag (on unless explicitly
  // off), decided beside the shape rather than in the bands.
  const mmprojOffload = settingValue(model, 'offload_mmproj') !== false
  // A toggled card is held here until the plan for it is ready. Writing
  // `device` on its own restarts the router with the placement of the old
  // card set, which names layers on a GPU that is no longer in `--device`.
  const [pendingDevices, setPendingDevices] = useState<string | null>(null)

  const {
    shape,
    bands,
    loading,
    machine,
    gpus,
    plannedFreeVram,
    plannedFreeRam,
    autoSelected,
    vramDrift,
    remeasure,
    computeFloorBytes,
    resident,
    calibrationKey,
    calibration,
    plannerOptions,
  } = useModelPlan(model.id, {
    kvCacheType,
    ubatch,
    sequences,
    flashAttn,
    noKvOffload,
    mmprojOffload,
    devices: pendingDevices ?? devices,
  })

  const storedContext = Number(settingValue(model, 'ctx_len')) || 0
  // Absent means the planner still owns the context. `ctx_len` is seeded at
  // 8192 on import, so the stored number was never evidence of a choice, and
  // honouring it pinned every untouched model to 8192 on a card with room
  // for ten times that. Moving the slider or typing a length sets it false.
  const ctxAuto = settingValue(model, 'ctx_auto') !== false
  const [context, setContext] = useState(storedContext)
  const [draft, setDraft] = useState(String(storedContext || ''))

  // The batch moves no ceilings on its own, so unlike the micro-batch it is
  // only ever written, never planned against. Falls back through the global
  // default the same way.
  const batchCandidates = [
    settingValue(model, 'batch_size'),
    useModelProvider((s) =>
      s
        .getProviderByName('llamacpp')
        ?.settings?.find((x) => x.key === 'batch_size')
        ?.controller_props?.value
    ),
  ]
    .map(Number)
    .filter((n) => Number.isFinite(n) && n > 0)
  const batch = batchCandidates.length ? Math.floor(batchCandidates[0]) : 2048
  const matchedBatchPreset = BATCH_PRESETS.some((p) => p.ubatch === ubatch)
  const [customBatchOpen, setCustomBatchOpen] = useState(false)
  const showCustomBatchInputs = customBatchOpen || !matchedBatchPreset
  const [batchDraft, setBatchDraft] = useState(String(batch))
  const [ubatchDraft, setUbatchDraft] = useState(String(ubatch))

  const commitBatchDraft = () => {
    const parsedBatch = Math.floor(Number(batchDraft))
    const parsedUbatch = Math.floor(Number(ubatchDraft))
    if (!Number.isFinite(parsedBatch) || parsedBatch <= 0) {
      setBatchDraft(String(batch))
      return
    }
    if (!Number.isFinite(parsedUbatch) || parsedUbatch <= 0) {
      setUbatchDraft(String(ubatch))
      return
    }
    // The micro-batch is the physical half of the logical batch and cannot
    // exceed it; raise the batch instead of sending a pair llama.cpp rejects.
    const nextBatch = Math.max(parsedBatch, parsedUbatch)
    setBatchDraft(String(nextBatch))
    setUbatchDraft(String(Math.min(parsedUbatch, nextBatch)))
    onChange({ batch_size: nextBatch, ubatch_size: Math.min(parsedUbatch, nextBatch) })
  }

  const maxContext = bands.length ? bands[bands.length - 1].maxContext : 0

  const resolved = useMemo(
    () => resolveContext(bands, context),
    [bands, context]
  )

  const bandIndex = resolved ? bands.indexOf(resolved.band) : 0

  const apply = useCallback(
    (next: number, band: ContextBand | undefined, pin = false) => {
      setContext(next)
      setDraft(String(next))
      if (!shape || !band || !machine || !plannerOptions) return

      const args = placementArgs(shape, band, machine, plannerOptions)
      // Manual owns the layer counts, but the card split is mechanics, not
      // philosophy: when the device choice itself moves (the toggles above),
      // the split is re-derived for the new set so a hand-placed multi-GPU
      // setup never runs on a stale or missing one. A split hand-set outside
      // the UI is left alone as long as the devices do not move.
      const storedDevice = (settingValue(model, 'device') as string) || ''
      const patch: Record<string, string | number | boolean> = manual
        ? {
            ctx_len: next,
            ...(pin ? { ctx_auto: false } : {}),
            ...(args.device !== storedDevice
              ? {
                  device: args.device,
                  tensor_split: args.tensorSplit ?? '',
                  split_mode: args.splitMode,
                }
              : {}),
          }
        : {
            ctx_len: next,
            ...(pin ? { ctx_auto: false } : {}),
            ngl: args.ngl,
            override_tensor_buffer_t: args.overrideTensorBufferType ?? '',
            device: args.device,
            tensor_split: args.tensorSplit ?? '',
            split_mode: args.splitMode,
            // Superseded by the override rule; left set they would fight it.
            cpu_moe: false,
            n_cpu_moe: '',
          }
      // Writing values that already hold would restart a running model every
      // time the sheet is opened.
      const changed = Object.fromEntries(
        Object.entries(patch).filter(
          ([key, value]) => (settingValue(model, key) ?? '') !== value
        )
      )
      if (Object.keys(changed).length) onChange(changed)
    },
    [plannerOptions, machine, manual, model, onChange, shape]
  )

  // On auto the planner owns the number, so the sheet writes the ceiling of
  // the fastest band and follows it as the bands move. A pinned context is
  // honoured, unless a K/V type or batch change has put it outside the bands,
  // in which case there is nothing to honour. Either way the placement has to
  // match the bands now in force.
  //
  // Fires once per target. `apply` writes through the parent, which rebuilds
  // `model`, which rebuilds `apply`, which re-runs this effect — a cycle that
  // only ended because the patch it recomputed came back empty. That held
  // while the target was the fastest band's ceiling, which is what every
  // fitted model already stored; once the target became a *different* band
  // the first patch was never empty and the cycle ran until React gave up
  // with "Maximum update depth exceeded". The stored value is not a reliable
  // stop condition, so the applied target is.
  const appliedRef = useRef('')
  useEffect(() => {
    if (!bands.length) return
    const usable =
      !ctxAuto &&
      storedContext >= MIN_USABLE_CONTEXT &&
      storedContext <= maxContext
    // The automatic size is the balanced band, not the fastest one: near
    // the fastest, another layer off the card buys context far out of
    // proportion to the traffic it costs.
    const target = usable ? storedContext : (autoBand(bands) ?? bands[0]).maxContext
    const band = resolveContext(bands, target)?.band
    const signature = `${model.id}|${target}|${band?.gpuLayers ?? ''}|${bands.length}|${maxContext}`
    if (appliedRef.current === signature) return
    appliedRef.current = signature
    apply(target, band)
  }, [bands, ctxAuto, storedContext, maxContext, apply, model.id])

  // Released once the stored placement names the same cards, whatever order
  // the planner put them in.
  useEffect(() => {
    if (!pendingDevices) return
    const same = (list: string) =>
      list.split(',').map((id) => id.trim()).filter(Boolean).sort().join(',')
    if (same(devices) === same(pendingDevices)) setPendingDevices(null)
  }, [devices, pendingDevices])

  const serviceHub = useServiceHub()
  const calibrating = useModelCalibration(
    (state) => state.running[model.id] !== undefined
  )
  const endRun = useModelCalibration((state) => state.endRun)
  const [calibrationError, setCalibrationError] = useState<string | null>(null)

  const calibrate = async () => {
    setCalibrationError(null)
    // Without a plan there is no axis to file the measurement under.
    if (!calibrationKey) {
      setCalibrationError('Could not plan this model, so nothing was measured.')
      return
    }
    const outcome = await runCalibrationProbe(
      serviceHub.models(),
      model.id,
      calibrationKey
    )
    if (outcome.status === 'cancelled') {
      setCalibrationError('Fit test cancelled.')
      return
    }
    if (outcome.status === 'unsupported') {
      setCalibrationError('This engine cannot measure a model.')
      return
    }
    if (outcome.status === 'failed') {
      setCalibrationError(outcome.error)
      return
    }
    if (outcome.report.failure) setCalibrationError(outcome.report.failure)
    // The probe held the whole model in VRAM a moment ago; the reading taken
    // when the sheet opened is older than that.
    remeasure()
  }

  // A stuck probe outlives this sheet (the download popover keeps showing
  // it), so cancelling kills the process backend-side; the in-flight
  // `calibrate()` above then rejects and drops the run quietly.
  const cancelCalibrate = async () => {
    try {
      await serviceHub.models().cancelCalibrateModel()
    } catch (error) {
      console.error('Failed to cancel calibration:', model.id, error)
    } finally {
      endRun(model.id)
    }
  }

  const commitDraft = () => {
    const typed = Number(draft)
    if (!Number.isFinite(typed)) {
      setDraft(String(context))
      return
    }
    const clamped = Math.min(Math.max(typed, MIN_USABLE_CONTEXT), maxContext)
    apply(clamped, resolveContext(bands, clamped)?.band, true)
  }

  if (loading) {
    return <p className="text-muted-foreground text-xs">Reading model…</p>
  }

  // Without a readable shape there is nothing to plan against, so the panel
  // below cannot render. When the shape reads but no band fits, the panel
  // still renders: the K/V type, prompt processing, GPU choice and the
  // fit-test button are exactly the levers that make a model fit, and hiding
  // them behind the failure left no way to adjust.
  if (!shape) {
    return (
      <div className="space-y-1">
        <span className="font-medium">Context</span>
        <p className="text-muted-foreground leading-normal text-xs">
          Could not read this model, so context cannot be planned automatically.
        </p>
      </div>
    )
  }

  const selectedGpus = gpus.filter((gpu) => gpu.selected)
  const idleDisplayGpu = gpus.find((gpu) => !gpu.selected && gpu.busyAtRest)
  // No GPUs at all: everything is budgeted on the host, so the copy names
  // the CPU and system RAM instead of zero cards and zero VRAM.
  const hostOnly = (machine?.gpus.length ?? 0) === 0

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <div>
          <span className="font-medium">Measured fit</span>
          <p className="text-muted-foreground leading-normal text-xs">
            Weights and cache are read from the model file exactly. The scratch
            space llama.cpp builds around them is not in the file, so until it
            is measured the plan reserves the largest any model has needed.
          </p>
        </div>
        <div className="flex items-center gap-1">
          <Button
            size="sm"
            variant="ghost"
            disabled={calibrating}
            onClick={calibrate}
          >
            {calibrating
              ? 'Loading memory…'
              : calibration
                ? 'Measure again'
                : 'Measure this model'}
          </Button>
          {calibrating && (
            <Button size="sm" variant="ghost" onClick={cancelCalibrate}>
              Cancel
            </Button>
          )}
        </div>
        <p className="text-muted-foreground leading-normal text-xs">
          {calibrationError
            ? calibrationError
            : calibrating
              ? 'Loading the model once to read what it allocates. It unloads again when the reading is done.'
              : calibration && !calibration.failure && computeFloorBytes !== undefined
                ? `${formatMiB(computeFloorBytes)} of scratch space per card${
                    calibration.measuredAt
                      ? `, measured on ${new Date(calibration.measuredAt).toLocaleDateString()}`
                      : ''
                  }.`
                : calibration?.failure
                  ? calibration.failure
                  : 'Loads the model once and unloads it. Takes about a minute, and usually buys back context.'}
        </p>
      </div>

      {gpus.length > 1 && (
        <div className="space-y-3">
          <div>
            <span className="font-medium">GPUs</span>
            <p className="text-muted-foreground leading-normal text-xs">
              A second card is worth it only when it takes weights off the CPU.
              Splitting a model that already fits costs speed.
            </p>
          </div>
          <div className="space-y-2">
            {gpus.map((gpu) => (
              <div
                key={gpu.id}
                className="flex items-center justify-between gap-4"
              >
                <div className="min-w-0">
                  <div className="truncate">{gpu.name}</div>
                  <div className="text-muted-foreground text-xs">
                    {gpu.id} · {formatGiB(gpu.freeBytes)} free
                  </div>
                </div>
                <Switch
                  checked={gpu.selected}
                  onCheckedChange={() => {
                    const next = gpus
                      .filter((other) =>
                        other.index === gpu.index ? !gpu.selected : other.selected
                      )
                      .map((other) => other.id)
                    // An empty list would hand the choice back to llama.cpp,
                    // which uses every card. Keep at least this one.
                    const chosen = next.length ? next.join(',') : gpu.id
                    // Under a hand-set placement nothing recomputes, so the
                    // choice has nowhere to arrive from and is written here.
                    if (manual) onChange({ device: chosen })
                    else setPendingDevices(chosen)
                  }}
                />
              </div>
            ))}
          </div>
          <p className="text-muted-foreground leading-normal text-xs">
            {autoSelected
              ? selectedGpus.length === gpus.length
                ? 'Chosen automatically: this model does not fit on one card.'
                : `Chosen automatically. ${gpus.length - selectedGpus.length === 1 ? 'The other card is' : 'The other cards are'} left out because the model fits without ${gpus.length - selectedGpus.length === 1 ? 'it' : 'them'}.`
              : 'Chosen by hand.'}
            {idleDisplayGpu &&
              ` ${idleDisplayGpu.name} is holding memory at rest, so it is probably driving a display; using it can make the desktop stutter.`}
          </p>
        </div>
      )}

      <div className="space-y-3">
        <div>
          <span className="font-medium">K/V cache type</span>
          <p className="text-muted-foreground leading-normal text-xs">
            How the attention cache is stored. Smaller types buy context.
          </p>
        </div>
        <div className="grid grid-cols-4 gap-1">
          {KV_CACHE_TYPES.map((type) => (
            <Button
              key={type.value}
              size="sm"
              variant={kvCacheType === type.value ? 'default' : 'ghost'}
              onClick={() => {
                // llama.cpp needs flash attention for a quantized V cache,
                // so a quantized preset turns it on in the same patch.
                // Returning to f16 leaves the switch where it is.
                onChange(
                  type.value === 'f16'
                    ? {
                        cache_type_k: type.value,
                        cache_type_v: type.value,
                      }
                    : {
                        cache_type_k: type.value,
                        cache_type_v: type.value,
                        flash_attn: 'on',
                      }
                )
              }}
            >
              {type.label}
            </Button>
          ))}
        </div>
        <p className="text-muted-foreground leading-normal text-xs">
          {KV_CACHE_TYPES.find((t) => t.value === kvCacheType)?.hint}
        </p>
      </div>

      <div className="space-y-3">
        <div>
          <span className="font-medium">Prompt processing</span>
          <p className="text-muted-foreground leading-normal text-xs">
            How much of a prompt is read at once. Larger is faster and leaves
            less room for context.
          </p>
        </div>
        <div className="grid grid-cols-4 gap-1">
          {BATCH_PRESETS.map((preset) => (
            <Button
              key={preset.label}
              size="sm"
              variant={ubatch === preset.ubatch ? 'default' : 'ghost'}
              onClick={() => {
                setCustomBatchOpen(false)
                onChange({
                  batch_size: preset.batch,
                  ubatch_size: preset.ubatch,
                })
              }}
            >
              {preset.label}
            </Button>
          ))}
          <Button
            size="sm"
            variant={showCustomBatchInputs ? 'default' : 'ghost'}
            onClick={() => {
              setBatchDraft(String(batch))
              setUbatchDraft(String(ubatch))
              setCustomBatchOpen(true)
            }}
          >
            Custom
          </Button>
        </div>
        {showCustomBatchInputs && (
          <div className="flex items-center gap-2">
            <label className="text-muted-foreground text-xs flex-1">
              Batch
              <Input
                className="mt-1 text-right"
                value={batchDraft}
                onChange={(e) => setBatchDraft(e.target.value)}
                onBlur={commitBatchDraft}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitBatchDraft()
                }}
              />
            </label>
            <label className="text-muted-foreground text-xs flex-1">
              Micro-batch
              <Input
                className="mt-1 text-right"
                value={ubatchDraft}
                onChange={(e) => setUbatchDraft(e.target.value)}
                onBlur={commitBatchDraft}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitBatchDraft()
                }}
              />
            </label>
          </div>
        )}
        <p className="text-muted-foreground leading-normal text-xs">
          {BATCH_PRESETS.find((p) => p.ubatch === ubatch)?.hint ??
            `Custom: ${ubatch} of ${batch} tokens at a time. The micro-batch cannot exceed the batch.`}
        </p>
      </div>

      {bands.length ? (
        <div className="space-y-3">
          <div className="flex items-baseline justify-between gap-4">
            <span className="font-medium">Context size</span>
            {ctxAuto ? (
              <span className="text-muted-foreground text-xs">Automatic</span>
            ) : (
              <button
                className="text-muted-foreground text-xs underline"
                onClick={() =>
                  onChange({
                    ctx_auto: true,
                    ctx_len: (autoBand(bands) ?? bands[0]).maxContext,
                  })
                }
              >
                Use the automatic size
              </button>
            )}
            <Input
              className="w-28 text-right"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commitDraft}
              onKeyDown={(e) => {
                if (e.key === 'Enter') commitDraft()
              }}
            />
          </div>
          <Slider
            min={0}
            max={bands.length - 1}
            step={1}
            value={[bandIndex]}
            onValueChange={([index]) => {
              const band = bands[index]
              apply(band.maxContext, band, true)
            }}
          />
          <div className="flex justify-between text-muted-foreground text-xs">
            <span>{bands[0].maxContext.toLocaleString()}</span>
            <span>{maxContext.toLocaleString()}</span>
          </div>
          {resolved && (
            <p className="text-muted-foreground leading-normal text-xs">
              {hostOnly
                ? shape.isMoe
                  ? `All ${shape.layerCount} expert layers on the CPU`
                  : `All ${shape.layerCount} layers on the CPU`
                : shape.isMoe
                  ? `${resolved.band.gpuLayers} of ${shape.layerCount} expert layers on the GPU`
                  : `${resolved.band.gpuLayers} of ${shape.layerCount} layers on the GPU`}
              {resolved.freeHeadroom > 0 &&
                ` · ${resolved.freeHeadroom.toLocaleString()} more tokens available at this speed`}
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <div>
            <span className="font-medium">Context size</span>
            <p className="text-muted-foreground leading-normal text-xs">
              This model does not fit in the memory available, even with every
              layer on the CPU.
              {resident &&
                ' It is loaded, so this already counts its memory as freed — unloading it changes nothing.'}{' '}
              The plan is conservative until this model is measured, so try the
              fit test above first: it usually buys back context.
            </p>
          </div>
          <p className="text-muted-foreground leading-normal text-xs">
            Otherwise free VRAM or RAM and{' '}
            <button className="underline" onClick={remeasure}>
              re-plan
            </button>
            , pick a smaller K/V cache type or the Light prompt preset, or set
            layers and context by hand under Advanced (editing a layer count
            takes manual control of placement).
          </p>
        </div>
      )}

      <div className="space-y-1">
        {manual ? (
          <p className="text-muted-foreground leading-normal text-xs">
            Layer sizes are set by hand in Advanced — only the context size
            above is applied here, with the card split following your
            device choice.{' '}
            <button className="underline" onClick={onResume}>
              Plan it automatically
            </button>
          </p>
        ) : (
          <p className="text-muted-foreground leading-normal text-xs">
            {hostOnly
              ? `Planned against ${formatGiB(plannedFreeRam)} free system RAM.`
              : `Planned against ${formatGiB(plannedFreeVram)} free on ${selectedGpus.map((gpu) => gpu.id).join(' + ') || 'the CPU'}.`}
          </p>
        )}
        {vramDrift !== 0 && (
          <p
            className={cn(
              'leading-normal text-xs',
              vramDrift > 0 ? 'text-muted-foreground' : 'text-destructive'
            )}
          >
            {vramDrift > 0
              ? `${formatGiB(vramDrift)} more is free now.`
              : `${formatGiB(-vramDrift)} less is free now.`}{' '}
            <button className="underline" onClick={remeasure}>
              Re-plan
            </button>
          </p>
        )}
      </div>
    </div>
  )
}

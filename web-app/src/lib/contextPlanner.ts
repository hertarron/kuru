/**
 * Turns a model shape and a machine into the context sizes that machine can
 * actually run, and the llama.cpp placement that reaches each one.
 *
 * The unit of trade is one layer moved off the GPU. That frees the layer's
 * bytes, and those bytes buy KV cache, so each step up in context costs one
 * step down in speed. The steps are called bands here; only the ceiling of a
 * band is worth offering, because every context below it costs the same speed.
 *
 * All of this is arithmetic over GGUF metadata. Nothing is measured and nothing
 * is loaded. Speed estimates are a separate concern and need calibration.
 */

import type { HardwareData } from '@/hooks/useHardware'
import type { DeviceList } from '@/services/hardware/types'
import type { ModelShape } from './gguf'
import { KV_CACHE_TYPE_BYTES } from './gguf'

/**
 * Headroom left on top of the VRAM already in use.
 *
 * This is small on purpose. What the display and other applications hold is
 * measured, not guessed, so this covers only allocator slack and short-lived
 * spikes. A fixed 1 GiB reserve — what `fit_target` defaulted to — is wrong in
 * both directions: too little on the card driving three monitors, and far too
 * much on a compute card whose monitors hang off a second GPU.
 *
 * Measured on an idle RTX 3070 by raising the context until `cudaMalloc` fails:
 * 7664 MiB of its 8192 loads and 7703 does not. So a CUDA process cannot reach
 * about 512 MiB of the card whatever it does.
 *
 * On top of that, a loaded process holds about 194 MiB that no `buffer size`
 * line reports: the CUDA context and the allocator's own working set. Measured
 * on the same card, which reported 7626 MiB in use while llama.cpp's own lines
 * summed to 7432. The planner counts only buffer bytes, so that 194 has to
 * come out of the budget here or every plan overshoots by it. A plan that
 * summed to 7516 MiB of buffers failed at the last allocation for exactly this
 * reason, on a card with nothing else running.
 *
 * llama.cpp's own device list is not this number: it reported 7098 MiB free on
 * the same idle card, 566 MiB below what actually loads, and on the display
 * card it reported *more* free than the system said was unused. So the budget
 * is the card's total less what is in use less this margin, and the device
 * list is used only for names and ids.
 */
// 512 unreachable + 194 of runtime overhead + the slack the 39 MiB
// measurement window leaves.
export const DEFAULT_VRAM_SAFETY_MARGIN_BYTES = 768 * 1024 * 1024

/** Held back from system RAM for the OS and everything else the user is running. */
export const DEFAULT_RAM_RESERVE_BYTES = 3 * 1024 * 1024 * 1024

/**
 * System RAM the plan budgets, after what is held back. Zero on unified
 * memory, where the GPU pool is the system pool and counting both would
 * spend the same bytes twice.
 */
export function usableRamBytes(machine: MachineMemory): number {
  // Measured use wins over the fixed reserve, which assumed an idle machine.
  // On a 32 GiB box idling at 6 GiB used, budgeting total minus 3 GiB
  // overstates free RAM by 3 GiB, and the top band lands exactly in that gap:
  // the load creeps into OS headroom and stalls in allocator strain.
  const ramReserve = machine.ramReserveBytes ?? DEFAULT_RAM_RESERVE_BYTES
  const ramHeldBack = Math.max(machine.ramUsedBytes ?? 0, ramReserve)
  return machine.unified ? 0 : Math.max(0, machine.ramBytes - ramHeldBack)
}

/** Nothing below this is a usable chat, so a band that cannot reach it is dropped. */
export const MIN_USABLE_CONTEXT = 4096

export interface GpuMemory {
  /** llama.cpp's name for the device, such as `CUDA0`. Names the buffer type
   * in an `--override-tensor` rule, so it must come from llama.cpp itself. */
  id: string
  /** The card's own name, for the UI. */
  name: string
  /**
   * CUDA compute capability, when the hardware plugin reports one. A coarse
   * stand-in for speed until per-device bandwidth is measured: free VRAM is
   * not one, because a display card can report more free memory than the
   * faster compute card sitting idle next to it.
   */
  capability?: number
  totalBytes: number
  /**
   * VRAM already spoken for by the display, the compositor and anything else
   * running, including the driver overhead a CUDA process cannot use.
   */
  usedBytes: number
  /**
   * Extra bytes to leave alone on this card, beyond the planner's margin. A
   * card driving a display needs it: llama.cpp reports the memory as free, and
   * it is, right until the desktop asks for it back.
   */
  reserveBytes?: number
  /**
   * Whether the device shares system RAM. Resolved in `machineMemoryFrom`
   * from the hardware enumeration when it names a kind, and left unset when
   * it does not — every consumer falls back to the size heuristic then.
   */
  kind?: 'discrete' | 'integrated'
}

export interface MachineMemory {
  gpus: GpuMemory[]
  ramBytes: number
  /**
   * System RAM already in use when the snapshot was taken, in bytes. The
   * budget holds back the larger of this and the fixed reserve: the reserve
   * alone assumed an idle machine, and a browser left open overstates every
   * RAM figure by gigabytes. Unset when the reading is unavailable — or when
   * the model being planned is itself resident, whose own footprint is part
   * of that reading and is accounted for separately below.
   */
  ramUsedBytes?: number
  /**
   * One physical pool shared by CPU and GPU (Apple silicon, and the newer
   * APU-class machines). The RAM side is then not extra capacity.
   */
  unified: boolean
  /**
   * Which GPUs the model is planned against, in the order llama.cpp will be
   * given them. Their free VRAM is one budget, divided by `--tensor-split` in
   * the same proportion. Defaults to the single card with the most free VRAM,
   * because spreading a model that already fits is a cost with no return.
   */
  gpuIndices?: number[]
  safetyMarginBytes?: number
  ramReserveBytes?: number
}

/** Free VRAM on each GPU after its safety margin, in device order. */
export function freeVramByGpu(machine: MachineMemory): number[] {
  const margin = machine.safetyMarginBytes ?? DEFAULT_VRAM_SAFETY_MARGIN_BYTES
  return machine.gpus.map((gpu) =>
    Math.max(0, gpu.totalBytes - gpu.usedBytes - margin - (gpu.reserveBytes ?? 0))
  )
}

/**
 * Size-heuristic fallback for whether a device is integrated (shares system
 * RAM) rather than discrete. A device reporting more than 60% of system RAM
 * as its own memory is treated as integrated: that memory is system RAM
 * counted twice if it is also budgeted as VRAM. Wrong in both directions on
 * its own — a 24 GB discrete card on 32 GB of RAM reads integrated, and a
 * 512 MB carved iGPU reads discrete — so `isIntegratedGpu` prefers the
 * hardware enumeration's own kind whenever it names one.
 */
export function isIntegratedDevice(
  deviceBytes: number,
  ramBytes: number
): boolean {
  return ramBytes > 0 && deviceBytes > ramBytes * 0.6
}

/**
 * Device kind implied by llama.cpp's own device id, for the machines where
 * the hardware enumeration spells a card differently and the name match
 * fails. CUDA, ROCm/HIP and MUSA enumerate discrete cards only; Vulkan and
 * SYCL are the backends that also report an iGPU, so those stay unclassified
 * and fall through to the size heuristic.
 *
 * Worth having because the fallback's failure is not benign: a 24 GB card on
 * 32 GB of RAM reads integrated by size, which makes the machine look unified
 * and drops every band that puts a layer on the CPU.
 */
export function deviceKindFromId(
  id: string
): 'discrete' | undefined {
  return /^(CUDA|ROCm|HIP|MUSA)\d*$/i.test(id.trim()) ? 'discrete' : undefined
}

/**
 * Whether a GPU shares system RAM: the resolved kind wins, and the size
 * heuristic covers devices the enumeration could not classify.
 */
export function isIntegratedGpu(
  kind: 'discrete' | 'integrated' | undefined,
  deviceBytes: number,
  ramBytes: number
): boolean {
  if (kind === 'integrated') return true
  if (kind === 'discrete') return false
  return isIntegratedDevice(deviceBytes, ramBytes)
}

/**
 * GPU indices in the order a plan should fill them: most capable first, and
 * among equals the one with the most free VRAM.
 */
export function gpusByPriority(machine: MachineMemory): number[] {
  const free = freeVramByGpu(machine)
  return machine.gpus
    .map((_, index) => index)
    .sort((a, b) => {
      const capability =
        (machine.gpus[b].capability ?? 0) - (machine.gpus[a].capability ?? 0)
      return capability !== 0 ? capability : free[b] - free[a]
    })
}

/** The GPU a plan fills first. */
export function defaultMainGpuIndex(machine: MachineMemory): number {
  return gpusByPriority(machine)[0] ?? 0
}

/** The GPUs a plan targets, defaulting to the one with the most free VRAM. */
export function plannedGpuIndices(machine: MachineMemory): number[] {
  if (machine.gpus.length === 0) return []
  const chosen = machine.gpuIndices?.filter((i) => machine.gpus[i])
  return chosen && chosen.length ? chosen : [defaultMainGpuIndex(machine)]
}

export interface PlannerOptions {
  /** `--cache-type-k` / `--cache-type-v`; both are assumed to match. */
  kvCacheType: string
  /** `--ubatch-size`. Drives the compute buffer, which comes out of VRAM. */
  ubatch: number
  /**
   * `--parallel`. Only a recurrent or hybrid model spends VRAM on this, and it
   * spends it per sequence whatever the context. Defaults to llama-server's
   * own choice when Kuru passes no `--parallel`, which is 4.
   */
  sequences?: number
  /**
   * `--no-kv-offload`. Puts the whole KV cache in system RAM instead of on
   * the card. Every band's arithmetic turns on where the cache lives, so a
   * plan that ignores this reserves VRAM for a cache that is never there and
   * charges RAM nothing for one that is.
   */
  noKvOffload?: boolean
  /**
   * The floor of the compute buffer for this model on one card, from a
   * calibration probe. Without one the conservative fallback applies, which
   * costs context on every model that needs less than the largest ever seen.
   */
  computeFloorBytes?: number
  /**
   * `offload_mmproj`. Decides whether the vision projector (`mmprojBytes` on
   * the shape) lands in VRAM or in RAM. Anything but `false` means VRAM.
   */
  mmprojOffload?: boolean
  /**
   * Effective MTP (speculative decoding) state with the draft weights beside
   * it. When on and a draft file exists, the second model's weights land on
   * the cards and are charged to VRAM.
   */
  mtp?: boolean
  /** Draft model file size in bytes, for the MTP charge above. */
  mtpDraftBytes?: number
  /**
   * Draft KV cache elements per token, read from the draft GGUF header with
   * the same shape math as the main model. llama-server allocates the draft
   * its own context beside the main one, so it is charged per token wherever
   * the draft weights land.
   */
  mtpDraftKvPerToken?: number
  /**
   * `--swa-full`. Forces full attention on every layer, switching the
   * sliding-window sizing off. Kuru does not emit this flag, so window layers
   * keep the small cache by default.
   */
  swaFull?: boolean
  /**
   * RoPE scale factor, read from `rope_scale` (or `1 / rope_freq_scale`).
   * Multiplies the trained-context cap it exceeds; 1 leaves the cap alone.
   */
  ropeFactor?: number
}

export const DEFAULT_SEQUENCES = 4

/**
 * A calibration probe report, as far as the planner is concerned. Structural
 * so both the persisted store and the engine's report satisfy it.
 */
export interface CalibrationMeasurement {
  devices: { device: string; computeBytes: number }[]
  /** Set when the probe hit an allocation failure instead of loading. */
  failure?: string
  ubatch: number
  context: number
  /** When the probe ran, for the UI. Absent on a report built by hand. */
  measuredAt?: number
}

/**
 * Axes a measurement is valid for: card set, parallel slots, flash-attention
 * state, backend build, and MTP state. Micro-batch and context scale out
 * arithmetically, so neither is an axis. The backend is: llama.cpp changes
 * its graph builder between builds, and a floor measured on one build is not
 * valid on another. MTP is filed as an axis because a probe reports what it
 * ran, but the floor underneath it is shared — see `calibrationForAxes`.
 */
export interface CalibrationAxes {
  deviceIds: string[]
  parallel: number
  flashAttn: string
  backend: string
  mtp: boolean
}

/** Store key for one measurement. */
export function calibrationKeyFor(axes: CalibrationAxes): string {
  return `${[...axes.deviceIds].sort().join('+')}|par${axes.parallel}|fa${axes.flashAttn}|be${axes.backend}|mtp${axes.mtp ? 'on' : 'off'}`
}

/**
 * The measurement to plan one axis against, and the MTP state it was taken
 * under.
 *
 * MTP is an explicit term of the compute-buffer estimate, so the floor a
 * probe leaves behind is the same measurement whether MTP was on or off: the
 * exact key first, then its MTP sibling. Discarding a valid measurement the
 * moment MTP was switched on dropped the plan back to the conservative
 * fallback, and that is what asked llama.cpp for 1065 MiB of compute buffer
 * on a card the plan had left 712 MiB on.
 */
export function calibrationForAxes(
  calibrations: Record<string, CalibrationMeasurement> | undefined,
  axes: CalibrationAxes
): { measurement?: CalibrationMeasurement; mtpWhenMeasured: boolean } {
  const exact = calibrations?.[calibrationKeyFor(axes)]
  if (exact) return { measurement: exact, mtpWhenMeasured: axes.mtp }
  // A failed probe is a fact about the axis it ran on, not about this one:
  // borrowing it would report a fit test this axis never ran as failed.
  const sibling = calibrations?.[calibrationKeyFor({ ...axes, mtp: !axes.mtp })]
  return sibling && !sibling.failure
    ? { measurement: sibling, mtpWhenMeasured: !axes.mtp }
    : { measurement: undefined, mtpWhenMeasured: axes.mtp }
}

/**
 * Effective parallel slots: per-model override wins, then the global
 * setting, then llama-server's own auto default.
 */
export function effectiveParallel(perModel: unknown, global: unknown): number {
  // Text inputs persist strings; the preset's own normalizer coerces them.
  const pick = (value: unknown) => {
    const n =
      typeof value === 'number' || typeof value === 'string'
        ? Number(value)
        : NaN
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined
  }
  return pick(perModel) ?? pick(global) ?? DEFAULT_SEQUENCES
}

/**
 * Effective flash-attention state: per-model override wins, then the global
 * setting, then llama.cpp's auto default.
 */
export function effectiveFlashAttn(
  perModel: unknown,
  global: unknown
): string {
  const pick = (value: unknown) =>
    value === 'on' || value === 'off' ? value : undefined
  return pick(perModel) ?? pick(global) ?? 'auto'
}

/**
 * Effective micro-batch: per-model override wins, then the global setting,
 * then llama.cpp's default. The compute buffer and the prompt presets scale
 * with it, so the bands must read it rather than assume the default.
 */
export function effectiveUbatch(perModel: unknown, global: unknown): number {
  const pick = (value: unknown) => {
    const n =
      typeof value === 'number' || typeof value === 'string'
        ? Number(value)
        : NaN
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined
  }
  return pick(perModel) ?? pick(global) ?? 512
}

/**
 * Effective KV cache type: per-model override wins, then the global
 * setting, then the planner's own default.
 */
export function effectiveCacheType(
  perModel: unknown,
  global: unknown
): string {
  const pick = (value: unknown) =>
    typeof value === 'string' && value.trim().length > 0
      ? value.trim()
      : undefined
  return pick(perModel) ?? pick(global) ?? 'q8_0'
}

/**
 * Effective backend id (`version/backend`, as the calibration report
 * records it): the composed setting wins, else version plus backend, else
 * empty (an unconfigured engine, which matches nothing).
 */
export function effectiveBackend(
  versionBackend: unknown,
  version: unknown,
  backend: unknown
): string {
  if (typeof versionBackend === 'string' && versionBackend.trim()) {
    return versionBackend.trim()
  }
  if (
    typeof version === 'string' &&
    typeof backend === 'string' &&
    version.trim() &&
    backend.trim()
  ) {
    return `${version.trim()}/${backend.trim()}`
  }
  return ''
}

/**
 * Effective MTP state: the `mtp` toggle in model.yml. A floor measured with
 * MTP off must not be reused with it on (and back), because the draft model
 * changes what a load allocates.
 */
export function effectiveMtp(perModel: unknown): boolean {
  return perModel === true
}

/**
 * Effective RoPE scale factor. Only applies while scaling is on; without a
 * method the cap stays at the trained context. Read from `rope_scale`, or
 * from `1 / rope_freq_scale` when only that key carries a value (the control
 * shows one knob at a time, so a default 1.0 scale must not shadow a set
 * frequency scale).
 */
export function effectiveRopeFactor(
  method: unknown,
  scale: unknown,
  freqScale: unknown
): number {
  if (method === 'none' || method === undefined || method === null || method === '') {
    return 1
  }
  const s = Number(scale)
  if (Number.isFinite(s) && s > 0 && s !== 1) return s
  const f = Number(freqScale)
  if (Number.isFinite(f) && f > 0 && f !== 1) return 1 / f
  if (Number.isFinite(s) && s > 0) return s
  return 1
}

/**
 * The compute buffer llama.cpp reserved, turned back into the floor this
 * planner scales, with a margin.
 *
 * Only valid for the cards it was measured on. A split graph is not the
 * graph one card runs: the same model measured 604 MiB on the busier of two
 * cards and 559 MiB on one card at a smaller context, and a 1B model
 * measured 82 MiB a card split against 54 MiB alone. So a measurement taken
 * on a different set of cards is ignored rather than scaled.
 *
 * The margin is because nothing here notices a backend upgrade, and
 * llama.cpp's graph changes between builds. Ten percent of a floor is tens
 * of megabytes; an undersized one is a failed load.
 */
export function computeFloorForPlacement(
  shape: ModelShape,
  machine: MachineMemory,
  calibration: CalibrationMeasurement | undefined,
  mtpWhenMeasured = false
): number | undefined {
  if (!calibration || calibration.failure) return undefined
  const planned = plannedGpuIndices(machine)
    .map((index) => machine.gpus[index]?.id)
    .filter(Boolean)
    .sort()
    .join(',')
  const measuredOn = calibration.devices
    .map((device) => device.device)
    .sort()
    .join(',')
  if (planned !== measuredOn) return undefined

  const measured = calibration.devices.reduce(
    (most, device) => Math.max(most, device.computeBytes),
    0
  )
  if (measured <= 0) return undefined
  return (
    computeFloorFromMeasurement(
      shape,
      measured,
      calibration.ubatch,
      calibration.context,
      mtpWhenMeasured
    ) * 1.1
  )
}

/**
 * Reserved for the compute buffer of a model nobody has calibrated.
 *
 * Not derivable from metadata. llama.cpp's graph decides it, and three models
 * measured at the same context and micro-batch came out at 441, 92 and 156 MiB
 * with no ordering by embedding, layer count or expert count. So this is the
 * largest seen, rounded up: wrong by a little context on a small model, never
 * wrong by a failed load.
 */
export const COMPUTE_FLOOR_FALLBACK_BYTES = 512 * 1024 * 1024

/** Micro-batch term of the compute buffer, per embedding element. */
const COMPUTE_BYTES_PER_UBATCH_ELEMENT = 20

/**
 * What the MTP head adds to that term, per embedding element.
 *
 * An embedded MTP block (`nextn_predict_layers`) costs no weights the layer
 * totals do not already carry and no cache of its own — measured: a 65-block
 * Qwen3.5 hybrid reported 1215 MiB of K/V at 36,501 tokens, which is its 16
 * full-attention layers exactly, with nothing for the nextn block. What it
 * does cost is a second graph over the micro-batch.
 *
 * Read off the pair of loads that exist: the same model measured 670.19 MiB
 * of compute buffer at ubatch 1024 with MTP off (floor 499 MiB once the
 * micro-batch and mask terms come out), and asked for 1065.31 MiB at ubatch
 * 2048 with MTP on, where that floor predicts 826 MiB. The 239 MiB difference
 * over 2048 x 5120 elements is 23.9 bytes; 28 is that with the same headroom
 * the term above carries.
 *
 * One model, one pair of points: the shape is right (the block runs the same
 * tensors over the same micro-batch) but the intercept is unproven, so this
 * is deliberately generous rather than exact.
 */
const MTP_COMPUTE_BYTES_PER_UBATCH_ELEMENT = 28

/**
 * The recurrent state cache, which the engine reports as `RS buffer size`.
 *
 * Zero on a pure attention model. On a hybrid it is charged once per recurrent
 * layer per sequence and does not move with the context, so unlike the KV
 * cache it cannot be traded for a smaller context.
 */
export function recurrentStateBytes(
  shape: ModelShape,
  sequences: number = DEFAULT_SEQUENCES
): number {
  return shape.recurrentStateBytesPerSeq * shape.recurrentLayerCount * sequences
}

function recurrentStatePerLayer(
  shape: ModelShape,
  options: PlannerOptions
): number {
  return (
    shape.recurrentStateBytesPerSeq * (options.sequences ?? DEFAULT_SEQUENCES)
  )
}

export interface ContextBand {
  /**
   * Layers resident on the GPU. For a MoE model this counts layers whose
   * expert tensors stay on the GPU; attention always does.
   */
  gpuLayers: number
  /** Largest context this placement supports. The only value worth offering. */
  maxContext: number
  /** Weight bytes read from VRAM per token at this placement. */
  gpuActiveBytes: number
  /** Weight bytes read over the bus per token at this placement. */
  cpuActiveBytes: number
}

export interface PlacementArgs {
  ngl: number
  overrideTensorBufferType?: string
  /** `--device`: the cards the plan budgeted, comma separated. */
  device: string
  /** `--tensor-split`, omitted when only one card is used. */
  tensorSplit?: string
  /**
   * `--split-mode`. `none` registers only the main device as a backend, so a
   * tensor named for any other card aborts the load with `pre-allocated tensor
   * ... in a buffer (CUDA1) that cannot run the operation`, whatever the
   * override rules say.
   */
  splitMode: 'none' | 'layer'
}

export interface PlanInput {
  shape: ModelShape
  machine: MachineMemory
  options: PlannerOptions
}

/**
 * The planned cards in fill order: fastest first, so a slower card takes only
 * the overflow. The plan and the placement must walk them in the same order
 * or the plan validates a split the placement does not produce.
 */
function fillOrder(machine: MachineMemory): number[] {
  const priority = gpusByPriority(machine)
  return plannedGpuIndices(machine).sort(
    (a, b) => priority.indexOf(a) - priority.indexOf(b)
  )
}

/**
 * Scratch space llama.cpp reserves on the GPU, minus the part that grows with
 * the context.
 *
 * Calibrated on b9967 by loading Qwen3.6-35B-A3B (embedding 2048) four times
 * on one card and reading the card's own memory use, at every combination of
 * ctx 8192/20920 and micro-batch 512/2048. The micro-batch term came out at
 * 17.3 and 17.6 bytes per embedding element; 20 is that with headroom. It was
 * 64 before, which reserved 256 MiB at ub 2048 for something that costs 70.
 *
 * The floor is not separable from the weights by this method, so a model that
 * has not been calibrated gets the conservative fallback.
 */
export function estimateComputeBufferBytes(
  shape: ModelShape,
  ubatch: number,
  floorBytes?: number,
  mtp = false
): number {
  const floor = floorBytes ?? COMPUTE_FLOOR_FALLBACK_BYTES
  return (
    floor +
    ubatch *
      shape.embeddingLength *
      (COMPUTE_BYTES_PER_UBATCH_ELEMENT +
        (mtp ? MTP_COMPUTE_BYTES_PER_UBATCH_ELEMENT : 0))
  )
}

/**
 * The floor a measured compute buffer implies: the engine's own figure with
 * the two terms this planner already knows how to scale taken back out.
 *
 * The measurement is per device, and both terms are charged per device, so
 * this inverts one card's figure into one card's floor. Clamped at zero
 * because a model whose whole buffer is the micro-batch term would otherwise
 * come out negative.
 */
export function computeFloorFromMeasurement(
  shape: ModelShape,
  measuredBytes: number,
  ubatch: number,
  context: number,
  mtpWhenMeasured = false
): number {
  return Math.max(
    0,
    measuredBytes -
      ubatch *
        shape.embeddingLength *
        (COMPUTE_BYTES_PER_UBATCH_ELEMENT +
          (mtpWhenMeasured ? MTP_COMPUTE_BYTES_PER_UBATCH_ELEMENT : 0)) -
      contextScratchBytesPerToken(shape, ubatch) * context
  )
}

/**
 * The part of the compute buffer that scales with the context length.
 *
 * llama.cpp reserves an attention mask over the whole cache for each
 * micro-batch, so the buffer grows with context times micro-batch. Ignoring
 * this is what made a plan that fit on paper fail to load: at 262,144 tokens
 * it is over a gigabyte.
 *
 * The same four-point calibration gives 2.04 bytes per token per micro-batch
 * element, and no measurable term that grows with context alone: subtracting
 * the KV cache and this mask from the per-token slope left 17 bytes at one
 * micro-batch and -42 at the other. The `embeddingLength * 1.5` that used to
 * stand here was reserving 64 MiB per card at 20k tokens for nothing.
 */
export function contextScratchBytesPerToken(
  _shape: ModelShape,
  ubatch: number
): number {
  return ubatch * 2
}

function kvElementBytes(kvCacheType: string): number {
  return KV_CACHE_TYPE_BYTES[kvCacheType] ?? KV_CACHE_TYPE_BYTES.f16
}

function kvBytesPerToken(shape: ModelShape, kvCacheType: string): number {
  return shape.kvElementsPerToken * kvElementBytes(kvCacheType)
}

/** Bytes of a layer that move when that layer is offloaded. */
function movableBytes(shape: ModelShape, layerIndex: number): number {
  const layer = shape.layers[layerIndex]
  if (!layer) return 0
  // A MoE layer offloads only its experts; attention and norms are small and
  // stay on the GPU, which is what makes the trade cheap. A dense layer has no
  // experts, so the whole layer moves.
  return shape.isMoe ? layer.experts : layer.total
}

/** Bytes of a layer that stay on the GPU whatever the placement. */
function pinnedBytes(shape: ModelShape, layerIndex: number): number {
  const layer = shape.layers[layerIndex]
  if (!layer) return 0
  return shape.isMoe ? layer.total - layer.experts : 0
}

/**
 * Weight bytes read to produce one token. Non-expert weights are read in full;
 * expert weights are read in proportion to how many experts fire.
 */
function activeBytes(shape: ModelShape, layerIndex: number): number {
  const layer = shape.layers[layerIndex]
  if (!layer) return 0
  const dense = layer.total - layer.experts
  return dense + layer.experts * shape.expertActiveRatio
}

/**
 * Every band the machine supports, fastest first (most layers on the GPU,
 * least context). Bands that reach the same ceiling are collapsed to the
 * fastest one, and bands that cannot reach a usable context are dropped.
 */
export function planContextBands(input: PlanInput): ContextBand[] {
  const { shape, machine, options } = input
  if (shape.layerCount === 0) return []

  const free = freeVramByGpu(machine)
  const gpuIndices = plannedGpuIndices(machine)
  const fillIndices = fillOrder(machine)
  // No GPU at all: the host-only path below charges the compute buffer, the
  // output head and the whole KV cache to RAM instead of producing no bands.
  const hostOnly = gpuIndices.length === 0
  // `--tensor-split` divides the model in proportion to these budgets, so the
  // selected cards behave as one pool.
  const usableVram = gpuIndices.reduce((sum, i) => sum + (free[i] ?? 0), 0)
  const deviceCount = gpuIndices.length
  const usableRam = usableRamBytes(machine)

  const kvElementBytesNow = kvElementBytes(options.kvCacheType)
  // Sliding-window layers keep only `window + ubatch` of cache (unless
  // `--swa-full` forces full attention), so only the global layers scale
  // with the context. Without a window key this is the old math exactly.
  const swaActive =
    options.swaFull !== true &&
    shape.layers.some((l) => (l.swaWindow ?? 0) > 0)
  // Zero on a purely recurrent model, which has no cache that grows with the
  // context. Then neither memory limit binds and the trained context is free.
  const perToken = swaActive
    ? shape.layers.reduce(
        (sum, l) =>
          sum +
          (l.hasKvCache && !(l.swaWindow! > 0)
            ? shape.kvElementsPerLayerToken * kvElementBytesNow
            : 0),
        0
      )
    : kvBytesPerToken(shape, options.kvCacheType)
  // Each device runs its own share of the graph and reserves its own compute
  // buffer, so both parts are charged once per card.
  const computeBuffer =
    estimateComputeBufferBytes(
      shape,
      options.ubatch,
      options.computeFloorBytes,
      options.mtp === true
    ) * deviceCount
  const scratchPerToken =
    contextScratchBytesPerToken(shape, options.ubatch) * deviceCount

  // A vision projector (`mmproj`) and an MTP draft model ride on the GPU
  // with the weights when their offload is on, and on the host otherwise.
  const mmprojBytes = shape.mmprojBytes ?? 0
  const mmprojOnGpu = options.mmprojOffload !== false
  const mtpExtraBytes =
    options.mtp === true && (options.mtpDraftBytes ?? 0) > 0
      ? options.mtpDraftBytes!
      : 0
  const mtpDraftPerToken =
    mtpExtraBytes > 0
      ? (options.mtpDraftKvPerToken ?? 0) * kvElementBytesNow
      : 0

  // A MoE placement keeps attention and norms on the GPU at every band, and
  // the output head with them. The token embedding is pushed to the CPU by the
  // override rule, so it is not charged here.
  // The recurrent state follows its layer, and a MoE placement keeps every
  // layer on the GPU, so all of it is in VRAM at every band.
  let moeFixedGpu =
    shape.outputBytes +
    computeBuffer +
    recurrentStateBytes(shape, options.sequences)
  for (let i = 0; i < shape.layerCount; i++) moeFixedGpu += pinnedBytes(shape, i)
  const totalPinned = shape.layers.reduce(
    (sum, _, i) => sum + pinnedBytes(shape, i),
    0
  )

  const totalMovable = shape.layers.reduce(
    (sum, _, i) => sum + movableBytes(shape, i),
    0
  )
  const totalActive = shape.layers.reduce(
    (sum, _, i) => sum + activeBytes(shape, i),
    0
  )

  // Attention layers among the first N, for every N. The KV of a dense layer
  // follows its weights, so this is what decides the share of the cache that
  // lands in VRAM. Counted over global layers only: a recurrent layer carries
  // no cache, and a sliding-window layer's cache is capped (see below), so
  // neither must dilute the share. Without a window key this counts every
  // attention layer, as before.
  const isGlobalKvLayer = (index: number) => {
    const layer = shape.layers[index]
    return (
      !!layer?.hasKvCache && (!swaActive || !((layer.swaWindow ?? 0) > 0))
    )
  }
  const kvLayersBefore: number[] = [0]
  for (let i = 0; i < shape.layerCount; i++) {
    kvLayersBefore.push(kvLayersBefore[i] + (isGlobalKvLayer(i) ? 1 : 0))
  }
  const kvLayers = kvLayersBefore[shape.layerCount]

  // Window layers hold at most `window + ubatch` of cache each, whatever the
  // context, so their share is a fixed cost rather than a per-token one.
  // Sized at the cap: below it the real cache is smaller, which charges more
  // than needed — the safe direction.
  const windowCapBefore: number[] = [0]
  for (let i = 0; i < shape.layerCount; i++) {
    const layer = shape.layers[i]
    const window = swaActive && layer?.hasKvCache ? (layer.swaWindow ?? 0) : 0
    windowCapBefore.push(
      windowCapBefore[i] +
        (window > 0
          ? shape.kvElementsPerLayerToken *
            (window + options.ubatch) *
            kvElementBytesNow
          : 0)
    )
  }
  const windowCapTotal = windowCapBefore[shape.layerCount]

  // The same running count for recurrent layers, which carry state instead.
  const recurrentLayersBefore: number[] = [0]
  for (let i = 0; i < shape.layerCount; i++) {
    recurrentLayersBefore.push(
      recurrentLayersBefore[i] + (shape.layers[i]?.hasKvCache ? 0 : 1)
    )
  }
  const recurrentLayers = recurrentLayersBefore[shape.layerCount]

  /**
   * The layer handed back when the band drops from `n + 1` to `n`. Both
   * placements take from the front: llama.cpp fills the GPU with the *last*
   * layers, and `placementArgs` names `blk.0..cpuLayers-1` in the MoE
   * override rule. So the layer released is the same index either way.
   *
   * This read `n` for MoE, which walked the layers in the opposite order and
   * charged the band for the wrong set. Expert blocks are not interchangeable
   * -- in Qwen3.6-35B one is 498 MiB on a full-attention layer and 432 on a
   * recurrent one -- so a band under-charged itself by 66 MiB, and a MoE with
   * dense front layers (DeepSeek, GLM, Llama-4) would be out by far more.
   */
  const releasedLayer = (n: number) => shape.layerCount - 1 - n

  const bands: ContextBand[] = []
  let movableOnGpu = totalMovable
  let activeOnGpu = totalActive

  for (let gpuLayers = shape.layerCount; gpuLayers >= 0; gpuLayers--) {
    // Without a GPU only the fully-hosted placement exists; every other
    // band would budget VRAM that is not there.
    if (hostOnly && gpuLayers > 0) continue
    if (gpuLayers < shape.layerCount) {
      const released = releasedLayer(gpuLayers)
      movableOnGpu -= movableBytes(shape, released)
      activeOnGpu -= shape.isMoe
        ? shape.layers[released].experts * shape.expertActiveRatio
        : activeBytes(shape, released)
    }

    const offloaded = totalMovable - movableOnGpu
    const cpuLayers = shape.layerCount - gpuLayers

    // llama.cpp puts a layer's KV cache wherever that layer's weights are. A
    // MoE placement keeps every layer on the GPU and moves only expert
    // tensors, so all of the KV stays in VRAM. A dense placement moves whole
    // layers, so its KV splits in the same proportion.
    // Counted over global attention layers only: a hybrid model's recurrent
    // layers carry no cache, and a window layer's cache is capped below, so
    // neither must dilute the share.
    // `--no-kv-offload` sends the whole cache to host RAM whatever the
    // placement, and so does the host-only path, so none of it is charged to
    // VRAM and all of it to RAM.
    const gpuKvShare =
      hostOnly || options.noKvOffload
        ? 0
        : shape.isMoe
          ? 1
          : kvLayers > 0
            ? (kvLayers - kvLayersBefore[shape.layerCount - gpuLayers]) / kvLayers
            : 1
    // The capped window cache of the resident layers (VRAM) and of the
    // offloaded ones (RAM). A MoE placement keeps every layer resident.
    const windowVramCap =
      swaActive && !hostOnly && !options.noKvOffload
        ? shape.isMoe
          ? windowCapTotal
          : windowCapTotal - windowCapBefore[cpuLayers]
        : 0
    const windowRamCap = swaActive
      ? hostOnly || options.noKvOffload
        ? windowCapTotal
        : shape.isMoe
          ? 0
          : windowCapBefore[cpuLayers]
      : 0

    // The output head rides on the GPU from `-ngl 1` upward, which every dense
    // band asks for, so it is charged at all of them. The recurrent state
    // follows its own layer, so a dense band pays for only the part of it that
    // is still resident.
    const recurrentOnGpu =
      recurrentLayers > 0
        ? recurrentStateBytes(shape, options.sequences) *
          ((recurrentLayers -
            recurrentLayersBefore[shape.layerCount - gpuLayers]) /
            recurrentLayers)
        : 0
    const fixedGpu =
      (shape.isMoe
        ? moeFixedGpu
        : computeBuffer + shape.outputBytes + recurrentOnGpu) +
      (mmprojOnGpu ? mmprojBytes : 0) +
      mtpExtraBytes +
      windowVramCap

    const vramForKv = usableVram - fixedGpu - movableOnGpu
    if (!hostOnly && vramForKv <= 0) continue

    // The host runs the CPU share of the graph and reserves its own compute
    // for it, but no probe line reports host bytes (calibration drops the
    // CPU/CUDA_Host buffers), so the plan estimates them the same way as the
    // device side, scaled by the share of weights living on the CPU. At
    // ubatch 2048 and 131k context that is on the order of a gigabyte the old
    // math never counted — the load that creeps past 30 of 32 GiB and stalls.
    const cpuShare = totalMovable > 0 ? offloaded / totalMovable : 0
    // The projector and the output head have nowhere but RAM on the host-only
    // path (a MoE's pinned attention weights with them); elsewhere the
    // projector follows its own offload flag, unscaled by the CPU share.
    const hostExtras = hostOnly
      ? shape.outputBytes + mmprojBytes + (shape.isMoe ? totalPinned : 0)
      : mmprojOnGpu
        ? 0
        : mmprojBytes
    const hostFixed =
      estimateComputeBufferBytes(
        shape,
        options.ubatch,
        options.computeFloorBytes,
        options.mtp === true
      ) * cpuShare +
      hostExtras +
      windowRamCap
    const ramForKv = usableRam - offloaded - hostFixed
    // Only the weights and the host graph have to fit; whether anything is
    // left over for cache is the limit below, not a reason to drop the band.
    // Requiring headroom here dropped a band that puts nothing on the CPU --
    // a fully resident model needs no RAM at all, and a MoE keeps all of its
    // cache in VRAM -- so a busy machine lost placements it could run.
    if (ramForKv < 0) continue

    // The context-scaled part of the compute buffer is bought per token out of
    // the same VRAM as the cache, so it belongs on this side of the division.
    // The draft's own context lives on the cards beside the main one.
    const offCard = hostOnly || options.noKvOffload
    const vramPerToken =
      perToken * gpuKvShare + scratchPerToken + (offCard ? 0 : mtpDraftPerToken)
    const vramLimit = vramPerToken > 0 ? vramForKv / vramPerToken : Infinity
    // The host owes the same per-token scratch for its own share of the
    // cache. Only where a cache exists at all, so a purely recurrent model
    // keeps its unbounded context. Single pool, not per card.
    const hostScratchPerToken =
      perToken > 0
        ? contextScratchBytesPerToken(shape, options.ubatch) *
          (options.noKvOffload ? 1 : cpuShare)
        : 0
    const ramPerToken =
      perToken * (1 - gpuKvShare) +
      hostScratchPerToken +
      (offCard ? mtpDraftPerToken : 0)
    const ramLimit = ramPerToken > 0 ? ramForKv / ramPerToken : Infinity

    let maxContext = Math.floor(Math.min(vramLimit, ramLimit))
    // RoPE scaling exists to run past the trained context, so the cap moves
    // with the factor. Without scaling this is the old cap exactly.
    const ropeFactor = options.ropeFactor ?? 1
    if (shape.trainedContext > 0) {
      maxContext = Math.min(
        maxContext,
        Math.floor(shape.trainedContext * (ropeFactor > 0 ? ropeFactor : 1))
      )
    }
    if (maxContext < MIN_USABLE_CONTEXT) continue
    // The arithmetic above spends VRAM as one pool, which no split can
    // realise: a layer lives entirely on one card, so the fastest card stops
    // up to a layer short of its own budget and every leftover byte lands on
    // the last one. On a 7424 + 5024 MiB pair that put 158 MiB more on the
    // GTX 1080 than it had, while the RTX 3070 sat 159 MiB idle — the pooled
    // need matched the pooled budget exactly and neither card was right. So
    // the ceiling is lowered until the fill the placement actually performs
    // fits every card.
    if (fillIndices.length > 1) {
      maxContext = perCardCeiling(
        shape,
        gpuLayers,
        maxContext,
        machine,
        options,
        fillIndices
      )
      if (maxContext < MIN_USABLE_CONTEXT) continue
    }

    const previous = bands[bands.length - 1]
    // A later band reaching no more context than the one before it is strictly
    // worse: same context, more of the model off the GPU.
    if (previous && maxContext <= previous.maxContext) continue

    bands.push({
      gpuLayers,
      maxContext,
      gpuActiveBytes: activeOnGpu,
      cpuActiveBytes: totalActive - activeOnGpu,
    })
  }

  return bands
}

/**
 * How much more traffic over the bus an automatic plan will spend to reach a
 * roomier band. Bus bytes per token are the speed proxy the bands already
 * carry, and near the fastest band they buy context far out of proportion:
 * on a 48-layer MoE that fits 27 layers on the card, giving up two more
 * layers took the ceiling from 13k to 41k for ten percent more traffic.
 */
export const AUTO_BAND_BUS_TOLERANCE = 0.15

/**
 * The band an unpinned model opens at.
 *
 * `bands[0]` is the fastest placement, and taking it unconditionally left
 * models at a fraction of the context the machine could hold, because
 * `bands[0]` is not "everything on the GPU" — on a model that does not fit it
 * already has layers on the CPU, and the next one or two cost almost nothing.
 *
 * A model that *does* fit entirely reads zero bus bytes, and the first layer
 * off is the expensive one: it turns no PCIe traffic into some. So a fully
 * resident plan never gives up a layer here, and the ratio test applies only
 * once the model is already partly on the CPU.
 */
export function autoBand(bands: ContextBand[]): ContextBand | undefined {
  const fastest = bands[0]
  if (!fastest || fastest.cpuActiveBytes <= 0) return fastest
  const ceiling = fastest.cpuActiveBytes * (1 + AUTO_BAND_BUS_TOLERANCE)
  return bands.reduce(
    (best, band) => (band.cpuActiveBytes <= ceiling ? band : best),
    fastest
  )
}

/** The fastest band that reaches `context`, or undefined if none does. */
export function bandForContext(
  bands: ContextBand[],
  context: number
): ContextBand | undefined {
  return bands.find((b) => b.maxContext >= context)
}

export interface ResolvedContext {
  band: ContextBand
  /** What to pass as the context length. Not rounded up to the band ceiling. */
  contextLength: number
  /**
   * Context still available at this placement and this speed. Non-zero means
   * the request landed short of a ceiling, so the difference is going unused.
   */
  freeHeadroom: number
}

/**
 * Resolves a typed context length against the plan.
 *
 * The slider snaps to band ceilings, but a typed value must not: sitting just
 * under a ceiling is how you keep the faster placement, and rounding the
 * request up would silently take that choice away. So the requested length is
 * passed through as-is and only the placement is derived from it.
 *
 * `freeHeadroom` is what the UI needs to say "you can have this much more for
 * free", since a request between two ceilings leaves the difference unused.
 */
export function resolveContext(
  bands: ContextBand[],
  requested: number
): ResolvedContext | undefined {
  if (bands.length === 0) return undefined
  const contextLength = Math.max(MIN_USABLE_CONTEXT, Math.floor(requested))
  const band = bandForContext(bands, contextLength)
  if (!band) return undefined
  return {
    band,
    contextLength,
    freeHeadroom: band.maxContext - contextLength,
  }
}

/**
 * llama.cpp arguments for a band.
 *
 * A dense model is placed with `-ngl` and split across the chosen cards by
 * `--tensor-split`. A MoE model keeps every layer on the GPU and moves only
 * the expert tensors of the offloaded layers, which is what makes a large MoE
 * usable on a small card: the experts are most of the bytes but only a
 * fraction of them are read per token. `--device` is always emitted, because
 * llama.cpp otherwise spreads the model over every card it can see and spends
 * memory the plan never counted.
 */
export function placementArgs(
  shape: ModelShape,
  band: ContextBand,
  machine: MachineMemory,
  options: PlannerOptions
): PlacementArgs {
  // Priority order, so the fastest card is filled first and a slower one takes
  // only the overflow. Proportional splitting put more on the owner's GTX 1080
  // than on the RTX 3070, because llama.cpp reports the display card as having
  // the most free memory.
  const indices = fillOrder(machine)
  // No GPU: everything stays on the host. `ngl 0` keeps even the output head
  // off the cards, and no `--device` line is emitted (never a `CUDA0`
  // fallback, which would name a card that is not there).
  if (indices.length === 0) {
    return { ngl: 0, device: '', splitMode: 'none' }
  }
  const device = indices.map((i) => machine.gpus[i]?.id ?? 'CUDA0').join(',')

  const counts = splitLayersByBytes(shape, band, machine, options, indices)
  const gpuLayers = counts.reduce((sum, n) => sum + n, 0)
  // A single card needs no split, and llama.cpp reads an all-zero list as an
  // error rather than as "no preference".
  const tensorSplit =
    indices.length > 1 && gpuLayers > 0
      ? counts.map((n) => (n / gpuLayers).toFixed(3)).join(',')
      : undefined

  // `-ngl` counts the output head as one of its layers, so N transformer
  // layers on the GPU is N + 1. Measured: `-ngl 1` offloads the head and no
  // layer at all.
  const splitMode = indices.length > 1 ? 'layer' : 'none'

  if (!shape.isMoe) {
    return { ngl: band.gpuLayers + 1, device, tensorSplit, splitMode }
  }

  const cpuLayers = shape.layerCount - band.gpuLayers
  const rules: string[] = ['token_embd.weight=CPU']
  if (cpuLayers > 0) {
    rules.push(`blk\\.(${layerRange(0, cpuLayers)})\\.ffn_.*_exps.*=CPU`)
  }
  // Every remaining tensor of a layer is named for one card. A layer whose
  // expert tensors sit on one GPU while the rest of it runs on another aborts
  // the load with `pre-allocated tensor ... in a buffer (CUDA1) that cannot
  // run the operation`, so the whole layer is pinned rather than left to
  // `--tensor-split`, whose share is counted in layers and not in bytes.
  if (indices.length > 1) {
    let next = 0
    counts.forEach((count, slot) => {
      if (count === 0) return
      const id = machine.gpus[indices[slot]]?.id ?? 'CUDA0'
      rules.push(`blk\\.(${layerRange(next, count)})\\..*=${id}`)
      next += count
    })
  }

  return {
    ngl: shape.layerCount + 1,
    overrideTensorBufferType: rules.join(','),
    device,
    tensorSplit,
    splitMode,
  }
}

export interface AutofitRequest {
  shape: ModelShape
  /** Machine without `gpuIndices`; the device selection is resolved here. */
  machine: MachineMemory
  kvCacheType: string
  ubatch: number
  /** Effective parallel slots (per-model override, else global, else auto). */
  sequences: number
  /** Effective flash-attention state, for the calibration key. */
  flashAttn: string
  /**
   * Active llama.cpp build (`version/backend`), for the calibration key.
   * Empty when the engine has not reported one; matches nothing stored.
   */
  backend?: string
  /** Effective MTP state, for the calibration key. */
  mtp?: boolean
  /** MTP draft file size in bytes, charged to VRAM beside the shape. */
  mtpDraftBytes?: number
  /** MTP draft KV elements per token, charged per token beside the shape. */
  mtpDraftKvPerToken?: number
  /** `offload_mmproj`. Anything but `false` budgets the projector on GPU. */
  mmprojOffload?: boolean
  /** `--swa-full`. Switches the sliding-window sizing off. */
  swaFull?: boolean
  /** RoPE scale factor. Multiplies the trained-context cap it exceeds. */
  ropeFactor?: number
  /** `--no-kv-offload`: the cache lives in RAM, not on the cards. */
  noKvOffload?: boolean
  /** Stored device selection, comma-separated ids. Empty lets the planner pick. */
  devices: string
  /** Stored `ctx_len`, 0 when never set. */
  storedCtxLen: number
  /**
   * Whether the stored request is still the planner's to choose.
   *
   * `ctx_len` is seeded at 8192 on every import, so a stored value is not
   * evidence of a decision, and honouring it left every untouched model at
   * 8192 on a card with room for ten times that. The flag is the evidence:
   * absent means nobody has pinned a context and the top band's ceiling is
   * written, and it is cleared the moment the user moves the slider or types
   * a length.
   */
  ctxAuto: boolean
  /** Measurements by axis key; the matching entry is selected here. */
  calibrations?: Record<string, CalibrationMeasurement>
  /** Stored value of a placement key, for the change guard. */
  currentValue: (key: string) => unknown
}

export interface AutofitResult {
  /**
   * Placement to write (`ngl`, `device`, `tensor_split`, `split_mode`,
   * `override_tensor_buffer_t`, `cpu_moe`, `n_cpu_moe` — plus `ctx_len`,
   * unless the user has pinned one). Already guarded: keys whose stored
   * value matches are omitted, so an empty patch means already fitted. A
   * pinned request is never rewritten: a shortfall loads best-effort with a
   * notice instead.
   */
  patch: Record<string, string | number | boolean>
  band: ContextBand
  requested: number
  /** Set when the request exceeds every band. */
  shortfall?: { requested: number; fits: number }
}

/**
 * Fits a stored context request against a fresh machine snapshot: the same
 * arithmetic the settings sheet runs, without a sheet. Pure metadata math —
 * nothing is measured and nothing is loaded — so it is cheap enough to run
 * on every model load. Undefined when nothing fits at a usable context.
 */
export function autofitPlacement(req: AutofitRequest): AutofitResult | undefined {
  const {
    shape,
    kvCacheType,
    ubatch,
    sequences,
    flashAttn,
    backend = '',
    mtp,
    mtpDraftBytes,
    mtpDraftKvPerToken,
    mmprojOffload,
    swaFull,
    ropeFactor,
    noKvOffload,
  } = req

  // A stored device list is the user's own choice and wins. Without one the
  // planner picks, with the conservative floor: this decides which cards to
  // plan against, and the measured floor is only valid once that is decided.
  const stored = req.devices
    .split(',')
    .map((id) => id.trim())
    .filter(Boolean)
    .map((id) => req.machine.gpus.findIndex((gpu) => gpu.id === id))
    .filter((index) => index >= 0)
  const gpuIndices = stored.length
    ? stored
    : recommendedGpuIndices(shape, req.machine, {
        kvCacheType,
        ubatch,
        mtp,
        mtpDraftBytes,
        mtpDraftKvPerToken,
        mmprojOffload,
      })
  const machine: MachineMemory = { ...req.machine, gpuIndices }

  const { measurement, mtpWhenMeasured } = calibrationForAxes(
    req.calibrations,
    {
      deviceIds: gpuIndices.map((i) => machine.gpus[i]?.id ?? ''),
      parallel: sequences,
      flashAttn,
      backend,
      mtp: mtp === true,
    }
  )
  const computeFloorBytes = computeFloorForPlacement(
    shape,
    machine,
    measurement,
    mtpWhenMeasured
  )
  const planOptions: PlannerOptions = {
    kvCacheType,
    ubatch,
    computeFloorBytes,
    sequences,
    noKvOffload,
    mtp,
    mtpDraftBytes,
    mtpDraftKvPerToken,
    mmprojOffload,
    swaFull,
    ropeFactor,
  }
  const bands = planContextBands({ shape, machine, options: planOptions })
  if (!bands.length) return undefined

  const pinned = !req.ctxAuto && req.storedCtxLen >= MIN_USABLE_CONTEXT
  const auto = autoBand(bands) ?? bands[0]
  const requested = pinned ? Math.floor(req.storedCtxLen) : auto.maxContext
  const resolved = resolveContext(bands, requested)
  // Best effort: the closest achievable placement, with the request passed
  // through unclamped. Either the (conservative) plan was pessimistic and
  // the load succeeds, or it OOMs and the notice says why.
  const band = resolved?.band ?? bands[bands.length - 1]
  const args = placementArgs(shape, band, machine, planOptions)

  const candidate: Record<string, string | number | boolean> = {
    ngl: args.ngl,
    override_tensor_buffer_t: args.overrideTensorBufferType ?? '',
    device: args.device,
    tensor_split: args.tensorSplit ?? '',
    split_mode: args.splitMode,
    // Superseded by the override rule; left set they would fight it.
    cpu_moe: false,
    n_cpu_moe: '',
  }
  if (!pinned) {
    candidate.ctx_len = auto.maxContext
  }
  const patch = Object.fromEntries(
    Object.entries(candidate).filter(
      ([key, value]) => (req.currentValue(key) ?? '') !== value
    )
  )
  return {
    patch,
    band,
    requested,
    shortfall: resolved
      ? undefined
      : { requested, fits: bands[bands.length - 1].maxContext },
  }
}

/**
 * How many consecutive layers each card takes.
 *
 * Layers are not interchangeable once experts have been moved off: a MoE layer
 * that kept its experts is several times the size of one that did not. Dealing
 * by count, which is all `--tensor-split` can express, gave one card every
 * heavy layer and OOMed it while the other sat half empty. So the layers are
 * walked in order and each card is filled to its own free VRAM, in priority
 * order, and the counts are converted back into a split.
 */
function splitLayersByBytes(
  shape: ModelShape,
  band: ContextBand,
  machine: MachineMemory,
  options: PlannerOptions,
  indices: number[]
): number[] {
  return fillCards(shape, band.gpuLayers, band.maxContext, machine, options, indices)
    .counts
}

/**
 * The largest context at this placement that the per-card fill can hold.
 *
 * Binary search rather than arithmetic: the fill is a loop with a branch in
 * it, so there is no closed form for where it stops. It is monotonic though —
 * more context is more KV on every resident layer, so a context that
 * overflows means every larger one does too — which is what makes the search
 * valid. About twenty passes over the layer list per band, and it runs only
 * for a multi-card plan.
 */
function perCardCeiling(
  shape: ModelShape,
  gpuLayers: number,
  ceiling: number,
  machine: MachineMemory,
  options: PlannerOptions,
  indices: number[]
): number {
  const fits = (context: number) =>
    fillCards(shape, gpuLayers, context, machine, options, indices).overflow <= 0
  if (fits(ceiling)) return ceiling
  let low = 0
  let high = ceiling
  while (high - low > 1) {
    const mid = Math.floor((low + high) / 2)
    if (fits(mid)) low = mid
    else high = mid
  }
  return low
}

/**
 * The greedy fill behind both the placement and the per-card ceiling.
 *
 * Walks the resident layers in order, filling each card to its own free VRAM
 * in priority order, and reports the bytes each one ends up holding. The last
 * card takes whatever is left whether it fits or not — there is nowhere else
 * to put it — so `overflow` is how far past its budget that leaves it.
 */
function fillCards(
  shape: ModelShape,
  gpuLayers: number,
  context: number,
  machine: MachineMemory,
  options: PlannerOptions,
  indices: number[]
): { counts: number[]; overflow: number } {
  const free = freeVramByGpu(machine)
  // The placement has to survive anywhere in its band, so it is sized at the
  // ceiling.
  const ceiling = context
  const mmprojOnGpu = options.mmprojOffload !== false
  const mtpExtraBytes =
    options.mtp === true && (options.mtpDraftBytes ?? 0) > 0
      ? options.mtpDraftBytes!
      : 0
  // Nothing of the cache lands on a card under `--no-kv-offload`, so the
  // split is decided by weights alone.
  const kvBpe = kvElementBytes(options.kvCacheType)
  // Every card runs its own share of the graph and reserves its own compute
  // buffer and attention mask, so these two are charged against each budget.
  const fixed =
    estimateComputeBufferBytes(
      shape,
      options.ubatch,
      options.computeFloorBytes,
      options.mtp === true
    ) +
    contextScratchBytesPerToken(shape, options.ubatch) * ceiling
  // The projector, the draft weights and the draft's own cache are single
  // allocations that land on the first device in `--device`, like the output
  // head below. Charging them against every card cost a two-card plan more
  // than twice what they are worth: a 2 GiB projector took 50k tokens off the
  // ceiling of a 12 GiB pair against 23k off one card.
  const mainCardOnly =
    (mmprojOnGpu ? (shape.mmprojBytes ?? 0) : 0) +
    mtpExtraBytes +
    (mtpExtraBytes > 0 && !options.noKvOffload
      ? (options.mtpDraftKvPerToken ?? 0) * kvBpe * ceiling
      : 0)
  const swaActive =
    options.swaFull !== true &&
    shape.layers.some((l) => (l.swaWindow ?? 0) > 0)
  const kvForLayer = (layer: (typeof shape.layers)[number]) => {
    if (!layer.hasKvCache) return rsPerLayer
    const window = swaActive ? (layer.swaWindow ?? 0) : 0
    const length = window > 0 ? Math.min(ceiling, window + options.ubatch) : ceiling
    return kvBpe * shape.kvElementsPerLayerToken * length
  }
  const rsPerLayer = recurrentStatePerLayer(shape, options)

  const cpuLayers = shape.layerCount - gpuLayers
  // A dense model gives its last layers to the GPU; a MoE keeps every layer
  // and gives back only expert tensors, from the front.
  const resident = shape.isMoe
    ? shape.layers.map((_, i) => i)
    : shape.layers.map((_, i) => i).slice(cpuLayers)

  const counts = indices.map(() => 0)
  let slot = 0
  // The output head is not named by any override rule, so llama.cpp puts it on
  // the first device in `--device`. Leaving it uncharged overfilled that card
  // by the size of the head and failed the load in `graph_reserve`, after
  // every weight had already been placed.
  let used = shape.outputBytes + mainCardOnly
  for (const index of resident) {
    const layer = shape.layers[index]
    const stripped = shape.isMoe && index < cpuLayers
    const bytes =
      layer.total -
      (stripped ? layer.experts : 0) +
      (options.noKvOffload && layer.hasKvCache ? 0 : kvForLayer(layer))
    const capacity = (free[indices[slot]] ?? 0) - fixed
    if (used + bytes > capacity && slot < indices.length - 1) {
      slot++
      used = 0
    }
    counts[slot]++
    used += bytes
  }
  const lastCapacity = (free[indices[indices.length - 1]] ?? 0) - fixed
  return { counts, overflow: Math.max(0, used - lastCapacity) }
}

/**
 * The GPUs worth using for this model.
 *
 * Adding a card is worth it exactly when it takes weights off the CPU: on this
 * machine a second GPU took a 35B MoE from 3.9 to 10.3 tokens/s and a dense
 * 16B from 4.2 to 7.2, because in both cases the alternative was system RAM.
 * A model that already fits on the fastest card gains nothing and pays the
 * cross-device sync, so cards are added only until the weights fit.
 */
export function recommendedGpuIndices(
  shape: ModelShape,
  machine: MachineMemory,
  options: PlannerOptions
): number[] {
  const free = freeVramByGpu(machine)
  const ordered = gpusByPriority(machine)
  if (ordered.length <= 1) return ordered

  // An integrated device reports system RAM as its own memory. Adding one to
  // a plan that already holds a discrete card budgets that RAM twice, so a
  // mixed machine plans against its discrete cards alone. An all-integrated
  // machine keeps them all (and reads as unified above).
  const integrated = new Set(
    ordered.filter((index) =>
      isIntegratedGpu(
        machine.gpus[index]?.kind,
        machine.gpus[index]?.totalBytes ?? 0,
        machine.ramBytes
      )
    )
  )
  const pool =
    integrated.size > 0 && integrated.size < ordered.length
      ? ordered.filter((index) => !integrated.has(index))
      : ordered

  const mmprojOnGpu = options.mmprojOffload !== false
  const mtpExtraBytes =
    options.mtp === true && (options.mtpDraftBytes ?? 0) > 0
      ? options.mtpDraftBytes!
      : 0
  const weights =
    shape.layers.reduce((sum, layer) => sum + layer.total, 0) +
    shape.outputBytes +
    (mmprojOnGpu ? (shape.mmprojBytes ?? 0) : 0) +
    mtpExtraBytes
  const perDevice = estimateComputeBufferBytes(
    shape,
    options.ubatch,
    options.computeFloorBytes,
    options.mtp === true
  )
  const kvFloor =
    kvBytesPerToken(shape, options.kvCacheType) * MIN_USABLE_CONTEXT +
    (mtpExtraBytes > 0
      ? (options.mtpDraftKvPerToken ?? 0) *
        kvElementBytes(options.kvCacheType) *
        MIN_USABLE_CONTEXT
      : 0)

  const chosen: number[] = []
  let budget = 0
  for (const index of pool) {
    chosen.push(index)
    budget += free[index] ?? 0
    if (budget >= weights + kvFloor + perDevice * chosen.length) break
  }
  return chosen
}

/**
 * A regex alternation matching `count` layer indices from `start`. Built by
 * enumeration rather than by numeric-range regex tricks, which are unreadable
 * and easy to get wrong at the tens boundary.
 */
function layerRange(start: number, count: number): string {
  return Array.from({ length: count }, (_, i) => String(start + i)).join('|')
}

/**
 * Builds the planner's view of the machine from llama.cpp's device list.
 *
 * The device list is the only reading that matches what a load can actually
 * allocate: it comes from the same CUDA/Vulkan/Metal context llama.cpp will
 * use, so the driver's own overhead is already deducted. `hardware` supplies
 * only the system RAM and whether the memory is unified.
 */
export function machineMemoryFrom(
  devices: DeviceList[],
  hardware: HardwareData,
  perDevice: Record<string, { usedMiB?: number; reserveMiB?: number }> = {},
  /**
   * System RAM in use at plan time, in MiB. Omitted when the reading is stale
   * or when the model being planned is itself resident — its own footprint is
   * part of that reading, and counting it would shrink the bands of a session
   * that is running fine.
   */
  ramUsedMiB?: number
): MachineMemory {
  const MIB = 1024 * 1024
  const capabilities = new Map(
    hardware.gpus.map((gpu) => [
      gpu.name,
      Number(gpu.nvidia_info?.compute_capability),
    ])
  )
  // The device list carries no kind field, but the hardware enumeration one
  // level up does: `vulkan_info.device_type` is the Debug form of
  // VkPhysicalDeviceType ("IntegratedGpu" / "DiscreteGpu"), and a reported
  // NVIDIA compute capability means a discrete card. Matched by name, like
  // the capability map above — the two enumerations do not agree on order.
  // Unclassified devices keep the size heuristic in `isIntegratedGpu`.
  const kinds = new Map(
    (hardware.gpus ?? []).map((gpu) => {
      const rawCapability = gpu.nvidia_info?.compute_capability
      const capability =
        typeof rawCapability === 'string' && rawCapability.trim() !== ''
          ? Number(rawCapability)
          : NaN
      if (Number.isFinite(capability)) return [gpu.name, 'discrete'] as const
      const deviceType = gpu.vulkan_info?.device_type ?? ''
      if (/integrated/i.test(deviceType)) return [gpu.name, 'integrated'] as const
      if (/discrete/i.test(deviceType)) return [gpu.name, 'discrete'] as const
      return [gpu.name, undefined] as const
    })
  )
  const ramBytes = (hardware.total_memory || 0) * MIB
  const gpus = devices.map((device) => {
    const capability = capabilities.get(device.name)
    return {
      id: device.id,
      name: device.name,
      capability: Number.isFinite(capability) ? capability : undefined,
      // Name match first, then the backend the id names. Both can miss, and
      // then the size heuristic decides.
      kind: kinds.get(device.name) ?? deviceKindFromId(device.id),
      totalBytes: device.mem * MIB,
      // The system's reading of what is already on the card. llama.cpp's own
      // free figure is not usable here: it is 566 MiB pessimistic on an idle
      // card and optimistic on one driving a display, where Windows counts
      // desktop memory it is willing to evict.
      usedBytes:
        (perDevice[device.id]?.usedMiB ??
          Math.max(0, device.mem - device.free)) * MIB,
      reserveBytes: (perDevice[device.id]?.reserveMiB ?? 0) * MIB,
    }
  })
  // Apple silicon reports no discrete GPU, and its VRAM is the system RAM.
  // So is an all-integrated machine elsewhere: an Intel/AMD iGPU reports
  // system RAM as its own memory, and budgeting it as VRAM and again as RAM
  // spends the same bytes twice. The kind resolves it; the 60% size rule is
  // only the fallback for devices the enumeration could not classify
  // (a 24 GB discrete card on 32 GB of RAM must not read integrated).
  const allIntegrated =
    gpus.length > 0 &&
    gpus.every((gpu) => isIntegratedGpu(gpu.kind, gpu.totalBytes, ramBytes))
  return {
    gpus,
    ramBytes,
    ramUsedBytes: ramUsedMiB !== undefined ? ramUsedMiB * MIB : undefined,
    // Apple silicon reports no discrete GPU, and its VRAM is the system RAM.
    unified:
      (hardware.os_type === 'macos' && hardware.cpu.arch === 'aarch64') ||
      allIntegrated,
  }
}

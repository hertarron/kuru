import { useCallback, useEffect, useMemo, useState } from 'react'
import type { GgufMetadata } from '@janhq/tauri-plugin-llamacpp-api'

import {
  DISPLAY_GPU_USED_BYTES,
  gpuMemoryUsage,
  resolveGpuReserveMiB,
  useHardware,
} from '@/hooks/useHardware'
import { useServiceHub } from '@/hooks/useServiceHub'
import { useModelProvider } from '@/hooks/useModelProvider'
import { readModelShape, draftKvElementsPerToken, type ModelShape } from '@/lib/gguf'
import {
  calibrationForAxes,
  calibrationKeyFor,
  type CalibrationMeasurement,
  computeFloorForPlacement,
  DEFAULT_SEQUENCES,
  effectiveBackend,
  effectiveRopeFactor,
  freeVramByGpu,
  machineMemoryFrom,
  planContextBands,
  plannedGpuIndices,
  recommendedGpuIndices,
  type ContextBand,
  type MachineMemory,
  type PlannerOptions,
  usableRamBytes,
} from '@/lib/contextPlanner'
import { useModelCalibration } from '@/hooks/useModelCalibration'
import { useLoadBaseline } from '@/hooks/useLoadBaseline'

/**
 * A model's GGUF header never changes, and reading it costs a file read, so it
 * is kept for the life of the session.
 */
const shapeCache = new Map<string, Promise<ModelShape | undefined>>()

function loadShape(
  modelId: string,
  read: (id: string) => Promise<GgufMetadata | undefined>
): Promise<ModelShape | undefined> {
  const cached = shapeCache.get(modelId)
  if (cached) return cached

  const pending = read(modelId).then((header) =>
    header ? readModelShape(header) : undefined
  )
  shapeCache.set(modelId, pending)
  return pending
}

/** Below this, a change is background noise rather than a freed application. */
const DRIFT_FLOOR_BYTES = 512 * 1024 * 1024

const roundToFloor = (bytes: number) =>
  Math.round(bytes / DRIFT_FLOOR_BYTES) * DRIFT_FLOOR_BYTES

/**
 * Free VRAM across the machine as the system reports it. Too optimistic to
 * plan against — it cannot see the driver's own overhead — but cheap to poll,
 * and the error cancels when it is compared against itself, so it is what
 * tells the UI that memory has been freed since the plan was made.
 */
function reportedFreeVram(usage: { gpus: SystemUsage['gpus'] }): number {
  const MIB = 1024 * 1024
  return usage.gpus.reduce(
    (sum, gpu) => sum + Math.max(0, gpu.total_memory - gpu.used_memory) * MIB,
    0
  )
}

type SystemUsage = ReturnType<typeof useHardware.getState>['systemUsage']

export interface PlanGpu {
  index: number
  /** llama.cpp's name, such as `CUDA0`. */
  id: string
  name: string
  freeBytes: number
  selected: boolean
  /**
   * VRAM held at rest by something other than this app, which on a desktop
   * almost always means the card is driving a display. Work placed there
   * competes with the desktop, which no benchmark of this app can see.
   */
  busyAtRest: boolean
}

export interface ModelPlan {
  shape?: ModelShape
  /** Fastest placement first. Empty when the model does not fit at all. */
  bands: ContextBand[]
  loading: boolean
  /** The machine as planned against, for `placementArgs`. */
  machine?: MachineMemory
  gpus: PlanGpu[]
  /** Free VRAM across the selected GPUs when the plan was made. */
  plannedFreeVram: number
  /** Budgetable system RAM when the plan was made, for the host-only line. */
  plannedFreeRam: number
  /** True when the selection is the planner's own rather than the user's. */
  autoSelected: boolean
  /**
   * Signed change in free VRAM since the plan was made, once it is large
   * enough to matter. Positive means something closed and more context is
   * available than the plan offers. Zero while the reading is steady.
   */
  vramDrift: number
  /** Re-measure memory and re-plan against what is free now. */
  remeasure: () => void
  /**
   * The compute-buffer floor the plan used, when it came from a calibration
   * probe rather than the conservative fallback. `placementArgs` has to be
   * given the same one or its placement will not match the bands.
   */
  computeFloorBytes?: number
  /** True once this model has been measured. */
  calibrated: boolean
  /**
   * The measurement the floor came from, which is the entry for this axis or,
   * when MTP was switched since it ran, its MTP sibling.
   */
  calibration?: CalibrationMeasurement
  /** True while the model being planned is itself loaded. Its own footprint
   * sits inside the memory readings, but the next load happens after it
   * unloads — so the plan budgets freed memory, not what is free right now
   * (RAM already did this; VRAM did not, which collapsed the bands of any
   * session that was running fine). */
  resident: boolean
  /**
   * Axis key of the calibration entry the floor was selected with, for
   * reading the same entry (and filing probe results under it) elsewhere.
   */
  calibrationKey?: string
  /**
   * The full planner options the bands were computed with — the inputs above
   * plus the engine-resolved extras (projector offload, MTP state, backend,
   * RoPE factor). Pass this to `placementArgs` so the placement matches the
   * bands.
   */
  plannerOptions?: PlannerOptions
}

/**
 * Context bands for a model on this machine.
 *
 * The memory reading is taken once, when the shape loads, and held. Free VRAM
 * moves constantly, and re-planning on every change would shift the slider's
 * snap points under the user mid-drag. The cost is that a reading taken while
 * a game was open stays pessimistic, so `remeasure` re-reads on demand.
 *
 * The reading comes from llama.cpp's own device list rather than from NVML,
 * because only that figure matches what a load can allocate.
 */
export function useModelPlan(
  modelId: string | undefined,
  options: PlannerOptions & { devices?: string; flashAttn?: string }
): ModelPlan {
  const serviceHub = useServiceHub()
  const [shape, setShape] = useState<ModelShape | undefined>()
  const [measured, setMeasured] = useState<MachineMemory | undefined>()
  const [extras, setExtras] = useState<
    {
      mmprojBytes: number
      mtp: boolean
      mtpDraftBytes: number
      mtpDraftKvPerToken: number
    } | undefined
  >()
  const [loading, setLoading] = useState(false)
  const [measuredAt, setMeasuredAt] = useState(0)
  const [reportedAtPlan, setReportedAtPlan] = useState<number | undefined>()
  const [resident, setResident] = useState(false)

  // Global-only planner inputs: the backend build (calibration axis), the
  // RoPE factor (trained-context cap), and `--swa-full` (window sizing).
  // Returned as stored, never copied: a selector that builds a new array on
  // every call makes zustand's snapshot differ each render, which React ends
  // with "Maximum update depth exceeded" as soon as the sheet opens.
  const globalSettings = useModelProvider(
    (s) => s.getProviderByName('llamacpp')?.settings
  )
  const backend = useMemo(() => {
    const find = (key: string) =>
      globalSettings?.find((x) => x.key === key)?.controller_props?.value
    return effectiveBackend(
      find('version_backend'),
      find('llamacpp_version'),
      find('llamacpp_backend')
    )
  }, [globalSettings])
  const ropeFactor = useMemo(() => {
    const find = (key: string) =>
      globalSettings?.find((x) => x.key === key)?.controller_props?.value
    return effectiveRopeFactor(
      find('rope_scaling'),
      find('rope_scale'),
      find('rope_freq_scale')
    )
  }, [globalSettings])
  const swaFull = useMemo(() => {
    const find = (key: string) =>
      globalSettings?.find((x) => x.key === key)?.controller_props?.value
    return find('swa_full') === true
  }, [globalSettings])

  useEffect(() => {
    if (!modelId) {
      setShape(undefined)
      setMeasured(undefined)
      setExtras(undefined)
      setResident(false)
      return
    }

    let cancelled = false
    setLoading(true)
    setResident(false)
    const models = serviceHub.models()
    Promise.all([
      loadShape(modelId, (id) => serviceHub.models().readModelGguf(id)),
      serviceHub.hardware().getLlamacppDevices(),
      // Fresh RAM readings: the store copy is only polled while the system
      // monitor is open, and the plan is only as honest as its snapshot.
      serviceHub.hardware().getSystemUsage().catch(() => null),
      serviceHub.models().getActiveModels().catch(() => [] as string[]),
      // Sizes model.yml folds into `size_bytes` that the GGUF header does
      // not cover: the vision projector and the MTP draft model. Missing on
      // older engines; then every model plans as text-only without MTP.
      (typeof models.getModelExtraSizes === 'function'
        ? models.getModelExtraSizes(modelId).catch(() => undefined)
        : Promise.resolve(undefined)) as Promise<
        | {
            mmprojBytes: number
            mtp: boolean
            mtpDraftBytes: number
            mtpDraftHeader?: GgufMetadata
          }
        | undefined
      >,
    ])
      .then(([loaded, devices, freshUsage, activeModels, extra]) => {
        if (cancelled) return
        const { hardwareData, systemUsage: storedUsage, gpuReserveMiB } =
          useHardware.getState()
        const systemUsage = freshUsage ?? storedUsage
        if (freshUsage) useHardware.getState().updateSystemUsage(freshUsage)
        const active = devices.filter((device) => device.activated !== false)
        setShape(
          loaded
            ? { ...loaded, mmprojBytes: extra?.mmprojBytes ?? 0 }
            : undefined
        )
        setExtras(
          extra
            ? {
                mmprojBytes: extra.mmprojBytes ?? 0,
                mtp: extra.mtp === true,
                mtpDraftBytes: extra.mtpDraftBytes ?? 0,
                mtpDraftKvPerToken: draftKvElementsPerToken(
                  extra.mtpDraftHeader
                ),
              }
            : undefined
        )
        // Budget RAM from what is actually free — unless this model is
        // itself resident, in which case its own footprint sits inside that
        // reading and counting it would collapse the bands of a session that
        // is running fine. An unreadable snapshot falls back to the fixed
        // reserve, as before.
        const resident = !!modelId && (activeModels ?? []).includes(modelId)
        setResident(resident)
        // A resident model's own footprint is inside every live reading, and
        // the next load happens after it unloads. The load path measured the
        // machine without it, so that measurement is what to plan against —
        // it counts what other applications hold, which the live reading
        // cannot separate out and a blanket zero throws away.
        const baseline = resident
          ? useLoadBaseline.getState().byModel[modelId!]
          : undefined
        const usedMiB = resident
          ? baseline?.ramUsedMiB
          : systemUsage?.used_memory
        const ramUsedMiB =
          typeof usedMiB === 'number' && usedMiB > 0 ? usedMiB : undefined
        setReportedAtPlan(resident ? undefined : reportedFreeVram(systemUsage))
        // A resident model's own VRAM sits inside this reading, and the next
        // load happens after it unloads — so it plans against unloaded cards
        // (stored reserves only). Counting it collapsed the bands of any
        // session running fine, and the loaded-state usage additionally marked
        // every card as driving a display, withholding 512 MiB per card that
        // was never the desktop's. RAM already plans this way above.
        const perDevice = Object.fromEntries(
          active.map((device) => [
            device.id,
            {
              usedMiB: resident
                ? // No baseline means this model was loaded by a path that
                  // does not fit (hand-tuned placement, or Kuru Fit off), so
                  // fall back to treating the cards as empty rather than
                  // charging the model for its own memory.
                  (baseline?.perDeviceUsedMiB[device.id] ?? 0)
                : gpuMemoryUsage(systemUsage, hardwareData, device.name)?.used,
              reserveMiB: resolveGpuReserveMiB(
                systemUsage,
                hardwareData,
                device,
                gpuReserveMiB
              ),
            },
          ])
        )
        // TEMPORARY: diagnosing a resident model that plans as "does not fit".
        console.debug('[kuru plan]', {
          modelId,
          activeModels,
          resident,
          hasBaseline: !!baseline,
          ramUsedMiB,
          devices: active.map((d) => ({
            id: d.id,
            memMiB: d.mem,
            freeMiB: d.free,
            usedMiB: perDevice[d.id]?.usedMiB,
            reserveMiB: perDevice[d.id]?.reserveMiB,
          })),
        })
        setMeasured(
          machineMemoryFrom(active, hardwareData, perDevice, ramUsedMiB)
        )
      })
      .catch(() => {
        if (!cancelled) setMeasured(undefined)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [modelId, serviceHub, measuredAt])

  const remeasure = useCallback(() => setMeasuredAt(Date.now()), [])

  const {
    kvCacheType,
    ubatch,
    devices,
    sequences = DEFAULT_SEQUENCES,
    flashAttn = 'auto',
    mmprojOffload,
    noKvOffload,
  } = options

  // A stored device list is the user's own choice and wins. Without one the
  // planner picks, which is the case for every model nobody has touched.
  const chosenIndices = useMemo(() => {
    if (!measured || !shape) return undefined
    const stored = (devices ?? '')
      .split(',')
      .map((id) => id.trim())
      .filter(Boolean)
      .map((id) => measured.gpus.findIndex((gpu) => gpu.id === id))
      .filter((index) => index >= 0)
    if (stored.length) return stored
    // Deliberately the conservative floor: this decides which cards to plan
    // against, and the measured floor is only valid once that is decided.
    return recommendedGpuIndices(shape, measured, {
      kvCacheType,
      ubatch,
      mmprojOffload,
      mtp: extras?.mtp,
      mtpDraftBytes: extras?.mtpDraftBytes,
      mtpDraftKvPerToken: extras?.mtpDraftKvPerToken,
    })
  }, [
    measured,
    shape,
    devices,
    kvCacheType,
    ubatch,
    mmprojOffload,
    extras,
  ])

  const machine = useMemo(
    () =>
      measured && chosenIndices
        ? { ...measured, gpuIndices: chosenIndices }
        : undefined,
    [measured, chosenIndices]
  )

  // The axis a measurement of this placement belongs to (card set, parallel
  // slots, flash-attn, backend build, MTP state). A probe files its result
  // under this key; the floor may come from its MTP sibling.
  const calibrationAxes = useMemo(() => {
    if (!machine) return undefined
    return {
      deviceIds: plannedGpuIndices(machine).map(
        (index) => machine.gpus[index]?.id ?? ''
      ),
      parallel: sequences,
      flashAttn,
      backend,
      mtp: extras?.mtp === true,
    }
  }, [machine, sequences, flashAttn, backend, extras])
  const calibrationKey = useMemo(
    () => (calibrationAxes ? calibrationKeyFor(calibrationAxes) : undefined),
    [calibrationAxes]
  )

  const stored = useModelCalibration((state) =>
    modelId ? state.byModel[modelId] : undefined
  )
  const { measurement: calibration, mtpWhenMeasured } = useMemo(
    () =>
      calibrationAxes
        ? calibrationForAxes(stored, calibrationAxes)
        : { measurement: undefined, mtpWhenMeasured: false },
    [stored, calibrationAxes]
  )

  const computeFloorBytes = useMemo(() => {
    if (!shape || !machine) return undefined
    return computeFloorForPlacement(shape, machine, calibration, mtpWhenMeasured)
  }, [shape, machine, calibration, mtpWhenMeasured])

  const plannerOptions = useMemo<PlannerOptions | undefined>(() => {
    if (!shape || !machine) return undefined
    return {
      kvCacheType,
      ubatch,
      computeFloorBytes,
      sequences,
      noKvOffload,
      mmprojOffload,
      mtp: extras?.mtp,
      mtpDraftBytes: extras?.mtpDraftBytes,
      mtpDraftKvPerToken: extras?.mtpDraftKvPerToken,
      swaFull,
      ropeFactor,
    }
  }, [
    shape,
    machine,
    kvCacheType,
    ubatch,
    computeFloorBytes,
    sequences,
    noKvOffload,
    mmprojOffload,
    extras,
    swaFull,
    ropeFactor,
  ])

  const bands = useMemo(() => {
    if (!shape || !machine || !plannerOptions) return []
    return planContextBands({ shape, machine, options: plannerOptions })
  }, [shape, machine, plannerOptions])

  const gpus = useMemo<PlanGpu[]>(() => {
    if (!machine) return []
    const free = freeVramByGpu(machine)
    const selected = new Set(plannedGpuIndices(machine))
    return machine.gpus.map((gpu, index) => ({
      index,
      id: gpu.id,
      name: gpu.name,
      freeBytes: free[index] ?? 0,
      selected: selected.has(index),
      // Read from the planned machine rather than from live usage, so a
      // resident model's own VRAM never makes every card look busy.
      busyAtRest: gpu.usedBytes > DISPLAY_GPU_USED_BYTES,
    }))
  }, [machine])

  const plannedFreeVram = machine
    ? plannedGpuIndices(machine).reduce(
        (sum, index) => sum + (freeVramByGpu(machine)[index] ?? 0),
        0
      )
    : 0
  const plannedFreeRam = machine ? usableRamBytes(machine) : 0
  // Rounded so a steady reading gives one stable value instead of a new one on
  // every poll, which would re-render the sheet continuously.
  const liveFree = useHardware((state) =>
    roundToFloor(reportedFreeVram(state.systemUsage))
  )
  const drift =
    reportedAtPlan === undefined ? 0 : liveFree - roundToFloor(reportedAtPlan)

  return {
    shape,
    bands,
    loading,
    machine,
    gpus,
    plannedFreeVram,
    plannedFreeRam,
    autoSelected: !(devices ?? '').trim(),
    vramDrift: Math.abs(drift) >= DRIFT_FLOOR_BYTES ? drift : 0,
    remeasure,
    computeFloorBytes,
    calibrated: computeFloorBytes !== undefined,
    calibration,
    resident,
    calibrationKey,
    plannerOptions,
  }
}

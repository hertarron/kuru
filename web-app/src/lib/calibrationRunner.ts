import {
  isCalibrationCancelled,
  useModelCalibration,
  type CalibrationReport,
} from '@/hooks/useModelCalibration'
import { useDownloadStore } from '@/hooks/useDownloadStore'
import {
  gpuMemoryUsage,
  resolveGpuReserveMiB,
  useHardware,
} from '@/hooks/useHardware'
import { useModelProvider } from '@/hooks/useModelProvider'
import { hasActiveLlamacppRequest } from '@/containers/dialogs/llamacppRouterError'
import type { GgufMetadata } from '@janhq/tauri-plugin-llamacpp-api'
import type { DeviceList, SystemUsage } from '@/services/hardware/types'
import {
  autofitPlacement,
  calibrationForAxes,
  calibrationKeyFor,
  computeFloorForPlacement,
  effectiveBackend,
  effectiveCacheType,
  effectiveFlashAttn,
  effectiveMtp,
  effectiveParallel,
  effectiveRopeFactor,
  effectiveUbatch,
  machineMemoryFrom,
} from './contextPlanner'
import { readModelShape, draftKvElementsPerToken } from './gguf'
import { applyPlannerPatch } from './plannerPatch'

/**
 * Shared fit-test probe runner. The settings sheet's manual button and the
 * background scheduler run the same probe; this keeps the run bookkeeping
 * (start/record/end) and the error classification in one place.
 *
 * A cancelled probe records nothing and resolves `cancelled`, so a caller
 * whose run was preempted (popover cancel, chat send) simply drops it.
 */
export type CalibrationProbeOutcome =
  | { status: 'measured'; report: CalibrationReport }
  | { status: 'unsupported' }
  | { status: 'cancelled' }
  | { status: 'failed'; error: string }

interface ProbeModelsService {
  calibrateModel(modelId: string): Promise<CalibrationReport | undefined>
}

export async function runCalibrationProbe(
  models: ProbeModelsService,
  modelId: string,
  axisKey: string
): Promise<CalibrationProbeOutcome> {
  useModelCalibration.getState().startRun(modelId)
  try {
    const report = await models.calibrateModel(modelId)
    if (!report) return { status: 'unsupported' }
    useModelCalibration.getState().record(modelId, axisKey, report)
    return { status: 'measured', report }
  } catch (error) {
    if (isCalibrationCancelled(error)) return { status: 'cancelled' }
    return { status: 'failed', error: String(error) }
  } finally {
    useModelCalibration.getState().endRun(modelId)
  }
}

export interface CalibrationSchedulerHub {
  models(): {
    getActiveModels(provider?: string): Promise<string[]>
    readModelGguf(modelId: string): Promise<GgufMetadata | undefined>
    calibrateModel(modelId: string): Promise<CalibrationReport | undefined>
    getModelExtraSizes?(modelId: string): Promise<
      | {
          mmprojBytes: number
          mtp: boolean
          mtpModelPath?: string
          mtpDraftBytes: number
          mtpDraftHeader?: GgufMetadata
        }
      | undefined
    >
    updateModelSettings?(
      modelId: string,
      patch: Record<string, string | number | boolean>
    ): Promise<void>
  }
  hardware(): {
    getLlamacppDevices(): Promise<DeviceList[]>
    getSystemUsage(): Promise<SystemUsage | null>
  }
}

type SettingsBag =
  | Record<string, { controller_props?: { value?: unknown } }>
  | undefined

function readSetting(settings: SettingsBag, key: string): unknown {
  return settings?.[key]?.controller_props?.value
}

/**
 * A global llama.cpp setting by key. Provider-level settings are a list,
 * unlike the per-model map.
 */
export function globalLlamacppSetting(key: string): unknown {
  const settings = useModelProvider.getState().getProviderByName('llamacpp')
    ?.settings as
    | Array<{ key?: string; controller_props?: { value?: unknown } }>
    | undefined
  return settings?.find((s) => s?.key === key)?.controller_props?.value
}

/**
 * True when a background probe may run: no probe already going, no chat in
 * flight, and no model loaded on either local engine. Fail-closed: when the
 * state cannot be read, no probe starts. (The send path is the mirror image
 * and fails open.)
 */
export async function isProbeIdle(
  hub: CalibrationSchedulerHub
): Promise<boolean> {
  if (Object.keys(useModelCalibration.getState().running).length > 0) {
    return false
  }
  if (hasActiveLlamacppRequest()) return false
  try {
    const models = hub.models()
    const [llama, mlx] = await Promise.all([
      models.getActiveModels('llamacpp').catch(() => [] as string[]),
      models.getActiveModels('mlx').catch(() => [] as string[]),
    ])
    return llama.length === 0 && mlx.length === 0
  } catch {
    return false
  }
}

/**
 * Measures the model in the background when it is useful and free to do so.
 * Silent by design: every outcome only lands in the calibration store, where
 * the download popover and the settings sheet already reflect it.
 *
 * Skips — never queues — when a probe is already running, the machine is not
 * idle, the model is hand-tuned / an embedding / still downloading, or a
 * valid measurement (or an attributable failure) already covers the current
 * card set.
 *
 * A model with no card selection yet is fitted here rather than skipped. The
 * probe measures the settings as stored, and an empty selection makes
 * llama.cpp spread over every card while the planner recommends one — a
 * measurement that could never cover the plan. This used to be a skip,
 * waiting for the first load's autofit to write the selection, which meant a
 * newly imported model was never measured until it had been loaded once and
 * unloaded again: its first load, often its only one, ran on the
 * conservative compute-buffer fallback. On a two-card plan that fallback
 * costs about a gigabyte of KV cache. So the placement the first load would
 * have written is written here instead, and the probe runs against it.
 */
export async function ensureCalibration(
  hub: CalibrationSchedulerHub,
  modelId: string
): Promise<void> {
  const calib = useModelCalibration.getState()
  if (calib.running[modelId] !== undefined) return
  if (Object.keys(calib.running).length > 0) return
  if (!(await isProbeIdle(hub))) return

  const provider = useModelProvider.getState().getProviderByName('llamacpp')
  const model = provider?.models?.find((m) => m.id === modelId) as
    | { embedding?: boolean; settings?: SettingsBag }
    | undefined
  if (!model || model.embedding === true) return
  if (readSetting(model.settings, 'plan_manual') === true) return
  // Legacy Jan fitting owns placement; Kuru Fit is opt-out, so installs
  // predating the key behave as fitted.
  if (globalLlamacppSetting('kuru_fit') === false) return

  // A partial file measures nothing; the manual button stays for that case.
  const downloads = useDownloadStore.getState()
  if (downloads.localDownloadingModels.has(modelId)) return
  if (Object.values(downloads.downloads).some((d) => d.name === modelId)) {
    return
  }

  let devices = (readSetting(model.settings, 'device') as string) || ''

  let header: GgufMetadata | undefined
  try {
    header = await hub.models().readModelGguf(modelId)
  } catch {
    return
  }
  const shape = header ? readModelShape(header) : undefined
  if (!shape) return

  let devicesList: DeviceList[]
  try {
    devicesList = await hub.hardware().getLlamacppDevices()
  } catch {
    return
  }
  const { hardwareData, systemUsage: storedUsage, gpuReserveMiB } =
    useHardware.getState()
  if (!hardwareData || !hardwareData.total_memory) return
  let systemUsage = storedUsage
  try {
    const fresh = await hub.hardware().getSystemUsage()
    if (fresh) {
      systemUsage = fresh
      useHardware.getState().updateSystemUsage(fresh)
    }
  } catch {
    // A stale store copy still plans; the reserve floor covers its age.
  }

  const active = devicesList.filter((d) => d.activated !== false)

  // Nothing is loaded (idle above), so measured use is safe to count.
  const usedMiB = systemUsage?.used_memory
  const perDevice = Object.fromEntries(
    active.map((device) => [
      device.id,
      {
        usedMiB: gpuMemoryUsage(systemUsage, hardwareData, device.name)?.used,
        reserveMiB: resolveGpuReserveMiB(
          systemUsage,
          hardwareData,
          device,
          gpuReserveMiB
        ),
      },
    ])
  )
  const machine = machineMemoryFrom(
    active,
    hardwareData,
    perDevice,
    typeof usedMiB === 'number' && usedMiB > 0 ? usedMiB : undefined
  )
  // The floor — and therefore the key — follows the effective parallel,
  // flash-attn, backend-build and MTP state, not just the card set. A
  // measurement from before any of them moved is simply for another axis,
  // never stale data for this one.
  const parallel = effectiveParallel(
    readSetting(model.settings, 'parallel'),
    globalLlamacppSetting('parallel')
  )
  const flashAttn = effectiveFlashAttn(
    readSetting(model.settings, 'flash_attn'),
    globalLlamacppSetting('flash_attn')
  )
  const backend = effectiveBackend(
    globalLlamacppSetting('version_backend'),
    globalLlamacppSetting('llamacpp_version'),
    globalLlamacppSetting('llamacpp_backend')
  )
  const ropeFactor = effectiveRopeFactor(
    globalLlamacppSetting('rope_scaling'),
    globalLlamacppSetting('rope_scale'),
    globalLlamacppSetting('rope_freq_scale')
  )
  const swaFull = globalLlamacppSetting('swa_full') === true
  // Sizes model.yml folds into `size_bytes` that the header does not cover.
  // Missing on older engines; then the model plans as text-only without MTP.
  const extra = await hub
    .models()
    .getModelExtraSizes?.(modelId)
    .catch(() => undefined)
  if (extra?.mmprojBytes) shape.mmprojBytes = extra.mmprojBytes
  const mtp = effectiveMtp(extra?.mtp)
  const mtpDraftBytes = extra?.mtpDraftBytes ?? 0
  const mtpDraftKvPerToken = draftKvElementsPerToken(extra?.mtpDraftHeader)
  const mmprojOffload = readSetting(model.settings, 'offload_mmproj') !== false

  if (!devices.trim()) {
    const fitted = autofitPlacement({
      shape,
      machine,
      kvCacheType: effectiveCacheType(
        readSetting(model.settings, 'cache_type_k'),
        globalLlamacppSetting('cache_type_k')
      ),
      ubatch: effectiveUbatch(
        readSetting(model.settings, 'ubatch_size'),
        globalLlamacppSetting('ubatch_size')
      ),
      sequences: parallel,
      flashAttn,
      backend,
      mtp,
      mtpDraftBytes,
      mtpDraftKvPerToken,
      mmprojOffload,
      swaFull,
      ropeFactor,
      noKvOffload: readSetting(model.settings, 'no_kv_offload') === true,
      devices: '',
      storedCtxLen: Number(readSetting(model.settings, 'ctx_len')) || 0,
      ctxAuto: readSetting(model.settings, 'ctx_auto') !== false,
      calibrations: useModelCalibration.getState().byModel[modelId],
      currentValue: (key) => readSetting(model.settings, key),
    })
    // Nothing fits at a usable context, so there is no placement to measure.
    if (!fitted) return
    const chosen = fitted.patch.device
    if (typeof chosen !== 'string' || !chosen.trim()) return
    await applyPlannerPatch(
      modelId,
      fitted.patch,
      hub.models().updateModelSettings?.bind(hub.models())
    )
    devices = chosen
  }

  const gpuIndices = devices
    .split(',')
    .map((id) => id.trim())
    .filter(Boolean)
    .map((id) => active.findIndex((gpu) => gpu.id === id))
    .filter((index) => index >= 0)
  // A selection naming cards that are gone plans nothing.
  if (gpuIndices.length === 0) return
  const machine1 = { ...machine, gpuIndices }
  const axisKey = calibrationKeyFor({
    deviceIds: gpuIndices.map((i) => machine.gpus[i]?.id ?? ''),
    parallel,
    flashAttn,
    backend,
    mtp,
  })
  const stored = useModelCalibration.getState().byModel[modelId]
  const { measurement, mtpWhenMeasured } = calibrationForAxes(stored, {
    deviceIds: gpuIndices.map((i) => machine.gpus[i]?.id ?? ''),
    parallel,
    flashAttn,
    backend,
    mtp,
  })
  if (
    computeFloorForPlacement(shape, machine1, measurement, mtpWhenMeasured) !==
    undefined
  ) {
    return
  }
  const calibration = stored?.[axisKey]

  // Negative cache: a failure on this axis is terminal until the user
  // re-measures by hand. Anything else means this axis was never tried.
  if (calibration?.failure) return

  await runCalibrationProbe(hub.models(), modelId, axisKey)
}

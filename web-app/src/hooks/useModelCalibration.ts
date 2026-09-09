import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { localStorageKey } from '@/constants/localStorage'
import { backendStorage } from '@/lib/backendStorage'
import {
  calibrationKeyFor,
  DEFAULT_SEQUENCES,
} from '@/lib/contextPlanner'

/** VRAM one device held after a probe load, in bytes. */
export interface DeviceAllocation {
  /** llama.cpp's device name, such as `CUDA0`. */
  device: string
  modelBytes: number
  kvBytes: number
  recurrentBytes: number
  computeBytes: number
}

export interface CalibrationReport {
  devices: DeviceAllocation[]
  /** Set when the probe hit an allocation failure instead of loading. */
  failure?: string
  /** The settings the probe ran. The figures are only valid for these. */
  ubatch: number
  context: number
  measuredAt: number
  /**
   * `version/backend` at the time. llama.cpp changes its graph between
   * builds, so a measurement does not carry across one unchallenged.
   */
  backend: string
}

/**
 * True when `error` is the backend's response to cancelling a fit test rather
 * than a real failure. The Rust probe returns `INVALID_ARGUMENT` with a
 * "cancelled" marker in the details; the value can arrive as the serialized
 * object, a string, or an Error wrapping either.
 */
export function isCalibrationCancelled(error: unknown): boolean {
  if (typeof error === 'string') return /calibration cancelled/i.test(error)
  if (error instanceof Error) {
    if (/calibration cancelled/i.test(error.message)) return true
    const start = error.message.indexOf('{')
    const end = error.message.lastIndexOf('}')
    if (start !== -1 && end > start) {
      try {
        return isCalibrationCancelled(JSON.parse(error.message.slice(start, end + 1)))
      } catch {
        return false
      }
    }
    return false
  }
  if (typeof error === 'object' && error !== null) {
    const record = error as Record<string, unknown>
    return (
      (typeof record.details === 'string' &&
        /calibration cancelled/i.test(record.details)) ||
      (typeof record.message === 'string' &&
        /calibration cancelled/i.test(record.message))
    )
  }
  return false
}

interface CalibrationState {
  /** Model id to axis key (`calibrationKeyFor`) to probe report. */
  byModel: Record<string, Record<string, CalibrationReport>>
  /**
   * Model id to the time its probe started. A probe takes about a minute and
   * survives leaving the settings page, so the download popover reads this to
   * show that something is still running.
   */
  running: Record<string, number>
  record: (
    modelId: string,
    axisKey: string,
    calibration: CalibrationReport
  ) => void
  forget: (modelId: string, axisKey?: string) => void
  startRun: (modelId: string) => void
  endRun: (modelId: string) => void
}

export const useModelCalibration = create<CalibrationState>()(
  persist(
    (set) => ({
      byModel: {},
      running: {},
      record: (modelId, axisKey, calibration) =>
        set((state) => ({
          byModel: {
            ...state.byModel,
            [modelId]: {
              ...(state.byModel[modelId] ?? {}),
              [axisKey]: calibration,
            },
          },
        })),
      forget: (modelId, axisKey) =>
        set((state) => {
          /* eslint-disable @typescript-eslint/no-unused-vars */
          if (axisKey === undefined) {
            const { [modelId]: _removed, ...rest } = state.byModel
            return { byModel: rest }
          }
          const { [axisKey]: _dropped, ...rest } = state.byModel[
            modelId
          ] ?? {}
          /* eslint-enable @typescript-eslint/no-unused-vars */
          return {
            byModel: { ...state.byModel, [modelId]: rest },
          }
        }),
      startRun: (modelId) =>
        set((state) => ({
          running: { ...state.running, [modelId]: Date.now() },
        })),
      endRun: (modelId) =>
        set((state) => {
          /* eslint-disable @typescript-eslint/no-unused-vars */
          const { [modelId]: _removed, ...rest } = state.running
          /* eslint-enable @typescript-eslint/no-unused-vars */
          return { running: rest }
        }),
    }),
    {
      name: localStorageKey.modelCalibration,
      storage: createJSONStorage(() => backendStorage),
      skipHydration: true,
      version: 1,
      migrate: (persistedState) => {
        const persisted = (persistedState ?? {}) as {
          byModel?: Record<string, unknown>
        }
        const next: Record<string, Record<string, CalibrationReport>> = {}
        for (const [modelId, value] of Object.entries(
          persisted.byModel ?? {}
        )) {
          if (
            value &&
            typeof value === 'object' &&
            Array.isArray(
              (value as { devices?: unknown }).devices
            )
          ) {
            // v0 filed one report per model. Re-file it under its card set
            // with the defaults it was almost certainly measured under (auto
            // parallel, auto flash-attn, MTP off) and the backend it records;
            // anything else re-probes silently.
            const report = value as CalibrationReport
            next[modelId] = {
              [calibrationKeyFor({
                deviceIds: report.devices.map((d) => d.device),
                parallel: DEFAULT_SEQUENCES,
                flashAttn: 'auto',
                backend:
                  typeof report.backend === 'string' ? report.backend : '',
                mtp: false,
              })]: report,
            }
          } else if (value && typeof value === 'object') {
            next[modelId] = value as Record<string, CalibrationReport>
          }
        }
        return { ...persisted, byModel: next } as CalibrationState
      },
      // `running` is this session's own state. Persisting it would leave a
      // probe showing as in progress forever after a crash.
      partialize: (state) => ({ byModel: state.byModel }),
    }
  )
)

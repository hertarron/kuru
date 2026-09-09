import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { localStorageKey } from '@/constants/localStorage'
import { backendStorage } from '@/lib/backendStorage'

// Hardware data types
export interface CPU {
  arch: string
  core_count: number
  extensions: string[]
  name: string
  usage: number
  instructions?: string[] // Cortex migration: ensure instructions data ready
}

export interface GPUAdditionalInfo {
  compute_cap: string
  driver_version: string
}

export interface GPU {
  name: string
  total_memory: number
  vendor: string
  uuid: string
  driver_version: string
  activated?: boolean
  nvidia_info: {
    index: number
    compute_capability: string
  }
  vulkan_info: {
    index: number
    device_id: number
    device_type: string
    api_version: string
  }
}

export interface OS {
  name: string
  version: string
}

export interface RAM {
  available: number
  total: number
}

export interface HardwareData {
  cpu: CPU
  gpus: GPU[]
  os_type: string
  os_name: string
  total_memory: number
  os?: OS
  ram?: RAM
}

export interface SystemUsage {
  cpu: number
  used_memory: number
  total_memory: number
  gpus: {
    uuid: string
    used_memory: number
    total_memory: number
  }[]
}

// Default values
const defaultHardwareData: HardwareData = {
  cpu: {
    arch: '',
    core_count: 0,
    extensions: [],
    name: '',
    usage: 0,
  },
  gpus: [],
  os_type: '',
  os_name: '',
  total_memory: 0,
}

const defaultSystemUsage: SystemUsage = {
  cpu: 0,
  used_memory: 0,
  total_memory: 0,
  gpus: [],
}

/**
 * VRAM held at rest by something other than this app. On a desktop that almost
 * always means the card is driving a display.
 *
 * This labels a card in the UI. It does not move the budget: a busy card's
 * memory is already subtracted, because `usedBytes` is a live reading of what
 * the desktop and everything else hold right now. Reserving on top of that
 * charged the desktop twice and cost real context. What a reserve actually
 * buys is room for the desktop to *grow* after the model loads, and by
 * measurement that growth does not fail a load — Windows evicts desktop
 * memory and the desktop stutters. That is a comfort preference, so it is the
 * user's own per-card figure and nothing is held back by default.
 */
export const DISPLAY_GPU_USED_BYTES = 512 * 1024 * 1024

/**
 * Matched by name and then by UUID rather than by position: llama.cpp's device
 * list and the system's GPU list are two separate enumerations, and they do not
 * agree on order. Indexing one with the other reported the owner's idle RTX
 * 3070 as the display card because the GTX 1080 came first in the other list.
 */
/**
 * Live VRAM for one llama.cpp device, in MiB, or undefined when the polled
 * list does not carry it. This is the system's own reading, which is larger
 * than the figure llama.cpp can allocate: it cannot see the CUDA context and
 * driver overhead, which is about 1 GB on Windows.
 */
export function gpuMemoryUsage(
  usage: SystemUsage,
  hardware: HardwareData,
  deviceName: string
): { used: number; total: number } | undefined {
  const uuid = hardware.gpus.find((gpu) => gpu.name === deviceName)?.uuid
  if (!uuid) return undefined
  const found = usage.gpus.find((gpu) => gpu.uuid === uuid)
  if (!found || !found.total_memory) return undefined
  return { used: found.used_memory, total: found.total_memory }
}

/**
 * MiB to hold back on one card, on top of the planner's own margin. The
 * user's figure or nothing — see `DISPLAY_GPU_USED_BYTES` for why a card
 * driving a display no longer gets one automatically.
 */
export function resolveGpuReserveMiB(
  _usage: SystemUsage,
  _hardware: HardwareData,
  device: { id: string; name: string },
  stored: Record<string, number>
): number {
  return stored[device.id] ?? 0
}

export function isDisplayGpu(
  usage: SystemUsage,
  hardware: HardwareData,
  deviceName: string
): boolean {
  const MIB = 1024 * 1024
  const found = gpuMemoryUsage(usage, hardware, deviceName)
  return (found?.used ?? 0) * MIB > DISPLAY_GPU_USED_BYTES
}

interface HardwareStore {
  // Hardware data
  hardwareData: HardwareData
  systemUsage: SystemUsage

  /**
   * MiB withheld from the context planner per llama.cpp device id, on top of
   * the planner's own margin. Keyed by id rather than by index because the
   * order of the device list is not stable across driver versions.
   */
  gpuReserveMiB: Record<string, number>
  setGpuReserve: (deviceId: string, mib: number) => void

  // Update functions
  setCPU: (cpu: CPU) => void
  setGPUs: (gpus: GPU[]) => void
  setOS: (os: OS) => void
  setRAM: (ram: RAM) => void

  // Update entire hardware data at once
  setHardwareData: (data: HardwareData) => void

  // Update individual GPU
  updateGPU: (index: number, gpu: GPU) => void

  // Update RAM available
  updateSystemUsage: (usage: SystemUsage) => void

  // GPU loading state
  gpuLoading: { [index: number]: boolean }
  setGpuLoading: (index: number, loading: boolean) => void

  // Polling control
  pollingPaused: boolean
  pausePolling: () => void
  resumePolling: () => void
}

export const useHardware = create<HardwareStore>()(
  persist(
    (set) => ({
      hardwareData: defaultHardwareData,
      systemUsage: defaultSystemUsage,
      gpuReserveMiB: {},
      setGpuReserve: (deviceId, mib) =>
        set((state) => ({
          gpuReserveMiB: {
            ...state.gpuReserveMiB,
            [deviceId]: Math.max(0, Math.round(mib)),
          },
        })),
      gpuLoading: {},
      pollingPaused: false,
      setGpuLoading: (index, loading) =>
        set((state) => ({
          gpuLoading: {
            ...state.gpuLoading,
            [state.hardwareData.gpus[index].uuid]: loading,
          },
        })),
      pausePolling: () => set({ pollingPaused: true }),
      resumePolling: () => set({ pollingPaused: false }),

      setCPU: (cpu) =>
        set((state) => ({
          hardwareData: {
            ...state.hardwareData,
            cpu,
          },
        })),

      setGPUs: (gpus) =>
        set((state) => ({
          hardwareData: {
            ...state.hardwareData,
            gpus,
          },
        })),

      setOS: (os) =>
        set((state) => ({
          hardwareData: {
            ...state.hardwareData,
            os,
          },
        })),

      setRAM: (ram) =>
        set((state) => ({
          hardwareData: {
            ...state.hardwareData,
            ram,
          },
        })),

      setHardwareData: (data) =>
        set({
          hardwareData: {
            ...data,
            cpu: {
              ...data.cpu,
              // Cortex migration - ensure instructions data ready
              instructions: [],
            },
            ram: {
              available: 0,
              total: 0,
            },
            gpus: data.gpus.map((gpu) => ({
              ...gpu,
              activated: gpu.activated ?? false,
            })),
          },
        }),

      updateGPU: (index, gpu) =>
        set((state) => {
          const newGPUs = [...state.hardwareData.gpus]
          if (index >= 0 && index < newGPUs.length) {
            newGPUs[index] = gpu
          }
          return {
            hardwareData: {
              ...state.hardwareData,
              gpus: newGPUs,
            },
          }
        }),

      updateSystemUsage: (systemUsage) =>
        set(() => ({
          systemUsage,
        })),
    }),
    {
      name: localStorageKey.settingHardware,
      storage: createJSONStorage(() => backendStorage),
      skipHydration: true,
    }
  )
)

/**
 * Parses the allocation lines a calibration probe collects from llama.cpp.
 *
 * The planner computes weights and KV cache from GGUF metadata exactly, but
 * the compute buffer comes out of llama.cpp's graph builder and no metadata
 * predicts it. These are the engine's own figures for the placement the model
 * is actually configured with.
 */

/** VRAM one device holds after a load, in bytes. */
export interface DeviceAllocation {
  /** llama.cpp's device name, such as `CUDA0`. */
  device: string
  modelBytes: number
  kvBytes: number
  /** Recurrent/SSM state, on a hybrid or recurrent model. Zero otherwise. */
  recurrentBytes: number
  computeBytes: number
}

export interface CalibrationReport {
  devices: DeviceAllocation[]
  /** Set when the probe hit an allocation failure instead of loading. */
  failure?: string
  /** The micro-batch the probe ran, which the compute figure scales with. */
  ubatch: number
  /** The context the probe ran, which the KV and compute figures scale with. */
  context: number
  /** `version/backend`. llama.cpp's graph changes between builds. */
  backend: string
  measuredAt: number
}

const MIB = 1024 * 1024

/**
 * llama-server in router mode writes its child's output twice: once as the
 * child logs it, and again forwarded by the parent behind a `[port]` tag. The
 * forwarded copies are dropped rather than deduplicated by content, because a
 * model with two caches of the same size prints two identical lines that both
 * count.
 */
const FORWARDED = /^\[\d+\]/

const LINE =
  /(\S+)\s+(model|KV|RS|compute|output)\s+buffer size\s*=\s*([\d.]+)\s*MiB/

/**
 * Host-side buffers, which llama.cpp reports in the same shape as the device
 * ones. `CUDA_Host` is pinned system RAM, not VRAM on the card it is named
 * after, so counting it would inflate every figure on `CUDA0`.
 */
function isDeviceBuffer(name: string): boolean {
  return name !== 'CPU' && !name.includes('_Host') && !name.startsWith('CPU_')
}

function isFailureLine(line: string): boolean {
  const lower = line.toLowerCase()
  return (
    lower.includes('cudamalloc') ||
    lower.includes('failed to allocate') ||
    lower.includes('error loading model') ||
    lower.includes('failed to load model')
  )
}

export function parseCalibration(
  lines: string[],
  ran: { ubatch: number; context: number; backend: string }
): CalibrationReport {
  const byDevice = new Map<string, DeviceAllocation>()
  let failure: string | undefined

  for (const line of lines) {
    if (FORWARDED.test(line.trim())) continue
    if (!failure && isFailureLine(line)) failure = line.trim()

    const match = LINE.exec(line)
    if (!match) continue
    const [, device, kind, size] = match
    if (!isDeviceBuffer(device)) continue

    const bytes = Number(size) * MIB
    if (!Number.isFinite(bytes)) continue

    const entry = byDevice.get(device) ?? {
      device,
      modelBytes: 0,
      kvBytes: 0,
      recurrentBytes: 0,
      computeBytes: 0,
    }
    switch (kind) {
      case 'model':
      case 'output':
        entry.modelBytes += bytes
        break
      // A model with sliding-window attention reports one line per cache, so
      // these accumulate. The compute buffer is a single reservation that can
      // be printed more than once, so it does not.
      case 'KV':
        entry.kvBytes += bytes
        break
      case 'RS':
        entry.recurrentBytes += bytes
        break
      case 'compute':
        entry.computeBytes = Math.max(entry.computeBytes, bytes)
        break
    }
    byDevice.set(device, entry)
  }

  return {
    devices: [...byDevice.values()],
    failure,
    ubatch: ran.ubatch,
    context: ran.context,
    backend: ran.backend,
    measuredAt: Date.now(),
  }
}

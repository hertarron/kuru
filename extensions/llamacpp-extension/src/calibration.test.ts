import { describe, expect, it } from 'vitest'
import { parseCalibration } from './calibration'

const ran = { ubatch: 512, context: 4096, backend: 'b9967/win-cuda-12' }

// Verbatim from a probe run: two GPUs, a sliding-window model that reports two
// KV caches a card, host buffers named after the device they are pinned for,
// and the parent's forwarded copy of every line the child already wrote.
const LOG = `
0.00.899.520 I load_tensors:   CPU_Mapped model buffer size =   306.00 MiB
0.00.899.521 I load_tensors:        CUDA0 model buffer size =   243.76 MiB
0.00.899.522 I load_tensors:        CUDA1 model buffer size =   518.78 MiB
0.01.218.357 I llama_context:  CUDA_Host  output buffer size =     4.00 MiB
0.01.218.542 I llama_kv_cache:      CUDA0 KV buffer size =     8.00 MiB
0.01.219.844 I llama_kv_cache:      CUDA0 KV buffer size =    30.00 MiB
0.01.231.094 I sched_reserve:      CUDA0 compute buffer size =    82.07 MiB
0.01.231.098 I sched_reserve:      CUDA1 compute buffer size =    82.07 MiB
0.01.231.099 I sched_reserve:  CUDA_Host compute buffer size =    30.58 MiB
[12714] 0.01.218.542 I llama_kv_cache:      CUDA0 KV buffer size =     8.00 MiB
[12714] 0.01.219.844 I llama_kv_cache:      CUDA0 KV buffer size =    30.00 MiB
[12714] 0.01.231.094 I sched_reserve:      CUDA0 compute buffer size =    82.07 MiB
`.trim().split('\n')

const MIB = 1024 * 1024

describe('parseCalibration', () => {
  it('reads per-device VRAM and ignores host buffers and forwarded copies', () => {
    const report = parseCalibration(LOG, ran)
    const cuda0 = report.devices.find((d) => d.device === 'CUDA0')

    expect(report.devices.map((d) => d.device)).toEqual(['CUDA0', 'CUDA1'])
    expect(cuda0?.computeBytes).toBeCloseTo(82.07 * MIB)
    // Both of the card's own caches, neither of the parent's echoes.
    expect(cuda0?.kvBytes).toBeCloseTo(38 * MIB)
    expect(report.failure).toBeUndefined()
  })

  it('reports an allocation failure instead of a silent short reading', () => {
    const report = parseCalibration(
      ['ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory'],
      ran
    )
    expect(report.failure).toContain('cudaMalloc failed')
  })
})

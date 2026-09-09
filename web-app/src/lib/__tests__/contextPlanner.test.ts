import { describe, expect, it } from 'vitest'
import type { ModelShape } from '../gguf'
import {
  autofitPlacement,
  bandForContext,
  calibrationKeyFor,
  computeFloorForPlacement,
  defaultMainGpuIndex,
  effectiveBackend,
  effectiveCacheType,
  effectiveFlashAttn,
  effectiveMtp,
  effectiveParallel,
  effectiveRopeFactor,
  effectiveUbatch,
  estimateComputeBufferBytes,
  contextScratchBytesPerToken,
  autoBand,
  deviceKindFromId,
  isIntegratedDevice,
  isIntegratedGpu,
  machineMemoryFrom,
  recommendedGpuIndices,
  freeVramByGpu,
  placementArgs,
  planContextBands,
  plannedGpuIndices,
  resolveContext,
} from '../contextPlanner'

const GiB = 1024 ** 3

function denseShape(overrides: Partial<ModelShape> = {}): ModelShape {
  return {
    arch: 'llama',
    layerCount: 32,
    embeddingLength: 4096,
    trainedContext: 131072,
    kvElementsPerToken: 32 * 8 * (128 + 128),
    kvElementsPerLayerToken: 8 * (128 + 128),
    layers: Array.from({ length: 32 }, () => ({
      total: 0.14 * GiB,
      experts: 0,
      hasKvCache: true,
    })),
    nonLayerBytes: 0.6 * GiB,
    outputBytes: 0.35 * GiB,
    isMoe: false,
    expertActiveRatio: 1,
    recurrentStateBytesPerSeq: 0,
    recurrentLayerCount: 0,
    ...overrides,
  }
}

function moeShape(overrides: Partial<ModelShape> = {}): ModelShape {
  return {
    arch: 'qwen3moe',
    layerCount: 48,
    embeddingLength: 2048,
    trainedContext: 40960,
    kvElementsPerToken: 48 * 4 * (128 + 128),
    kvElementsPerLayerToken: 4 * (128 + 128),
    layers: Array.from({ length: 48 }, () => ({
      total: 0.376 * GiB,
      experts: 0.364 * GiB,
      hasKvCache: true,
    })),
    nonLayerBytes: 0.4 * GiB,
    outputBytes: 0.2 * GiB,
    isMoe: true,
    expertActiveRatio: 8 / 128,
    recurrentStateBytesPerSeq: 0,
    recurrentLayerCount: 0,
    ...overrides,
  }
}

const options = { kvCacheType: 'q8_0', ubatch: 1024 }
/** A card with nothing else on it, so free VRAM is the full amount less the margin. */
function card(totalGiB: number, usedGiB = 0, index = 0) {
  return {
    id: `CUDA${index}`,
    name: `Card ${index}`,
    totalBytes: totalGiB * GiB,
    usedBytes: usedGiB * GiB,
  }
}

const smallCard = {
  gpus: [card(8)],
  ramBytes: 32 * GiB,
  unified: false,
}

describe('planContextBands', () => {
  it('orders bands fastest first, with context strictly increasing', () => {
    const bands = planContextBands({
      shape: denseShape(),
      machine: smallCard,
      options,
    })

    expect(bands.length).toBeGreaterThan(1)
    for (let i = 1; i < bands.length; i++) {
      expect(bands[i].gpuLayers).toBeLessThan(bands[i - 1].gpuLayers)
      expect(bands[i].maxContext).toBeGreaterThan(bands[i - 1].maxContext)
    }
  })

  it('never offers more context than the model was trained for', () => {
    const bands = planContextBands({
      shape: denseShape({ trainedContext: 8192 }),
      machine: smallCard,
      options,
    })

    for (const band of bands) expect(band.maxContext).toBeLessThanOrEqual(8192)
  })

  it('stops offloading once the trained context is reached', () => {
    // Past that point another layer on the CPU buys nothing and costs speed.
    const bands = planContextBands({
      shape: denseShape(),
      machine: smallCard,
      options,
    })
    const last = bands[bands.length - 1]

    expect(last.maxContext).toBe(131072)
    expect(last.gpuLayers).toBeGreaterThan(0)
  })

  it('charges dense KV to RAM for layers that are not on the GPU', () => {
    // With every layer on the GPU all the KV is VRAM-bound. Halving the RAM
    // cannot change that band, but it must lower the bands that offload.
    const shape = denseShape()
    const roomy = planContextBands({ shape, machine: smallCard, options })
    const tight = planContextBands({
      shape,
      machine: { ...smallCard, ramBytes: 6 * GiB },
      options,
    })

    expect(tight[0].maxContext).toBe(roomy[0].maxContext)
    expect(tight[tight.length - 1].maxContext).toBeLessThan(
      roomy[roomy.length - 1].maxContext
    )
  })

  it('gives a MoE model far more context per offloaded layer than a dense one', () => {
    // The point of the expert-only placement: the freed bytes are large and
    // the bytes read per token barely move.
    const bands = planContextBands({
      shape: moeShape(),
      machine: smallCard,
      options,
    })
    const step = bands[1].maxContext - bands[0].maxContext

    expect(step).toBeGreaterThan(5000)
    expect(bands[1].cpuActiveBytes - bands[0].cpuActiveBytes).toBeLessThan(
      0.05 * GiB
    )
  })

  it('does not count a unified memory pool twice', () => {
    const shape = moeShape()
    const machine = { gpus: [card(16)], ramBytes: 16 * GiB, unified: true }
    const bands = planContextBands({ shape, machine, options })
    const split = planContextBands({
      shape,
      machine: { ...machine, unified: false },
      options,
    })

    // Treated as split memory the same numbers look like 32 GiB of capacity,
    // so the split plan can offload where the unified plan cannot.
    expect(bands.length).toBeLessThanOrEqual(split.length)
  })

  it('returns nothing when the model cannot fit at a usable context', () => {
    const bands = planContextBands({
      shape: moeShape(),
      machine: { gpus: [card(2)], ramBytes: 4 * GiB, unified: false },
      options,
    })

    expect(bands).toEqual([])
  })
})

describe('two cards of different sizes', () => {
  // 8 GiB idle beside 8 GiB with 3 GiB already spoken for. Spending VRAM as
  // one pool passes a split no placement can produce: layers are indivisible,
  // so the first card stops short of its own budget and every leftover byte
  // lands on the last one.
  const lopsided = {
    gpus: [
      { ...card(8), capability: 8.6 },
      { ...card(8, 3, 1), capability: 6.1 },
    ],
    ramBytes: 32 * GiB,
    unified: false,
    gpuIndices: [0, 1],
  }

  it('sizes every band so the last card is not overcommitted', () => {
    const shape = denseShape()
    const bands = planContextBands({ shape, machine: lopsided, options })
    expect(bands.length).toBeGreaterThan(0)

    const free = freeVramByGpu(lopsided)
    for (const band of bands) {
      const args = placementArgs(shape, band, lopsided, options)
      const counts = (args.tensorSplit ?? '1')
        .split(',')
        .map((f) => Math.round(Number(f) * band.gpuLayers))
      const fixed =
        estimateComputeBufferBytes(shape, options.ubatch) +
        contextScratchBytesPerToken(shape, options.ubatch) * band.maxContext
      const perLayer =
        shape.layers[0].total +
        (34 / 32) * shape.kvElementsPerLayerToken * band.maxContext
      counts.forEach((layers, i) => {
        const head = i === 0 ? shape.outputBytes : 0
        expect(layers * perLayer + head + fixed).toBeLessThanOrEqual(free[i])
      })
    }
  })

  it('fills the faster card first', () => {
    const shape = denseShape()
    const bands = planContextBands({ shape, machine: lopsided, options })
    const args = placementArgs(shape, bands[0], lopsided, options)
    const [first, second] = (args.tensorSplit ?? '').split(',').map(Number)
    expect(args.device).toBe('CUDA0,CUDA1')
    expect(first).toBeGreaterThan(second)
  })
})

describe('placementArgs', () => {
  it('places a dense model with ngl alone', () => {
    const bands = planContextBands({
      shape: denseShape(),
      machine: smallCard,
      options,
    })
    const args = placementArgs(denseShape(), bands[2], smallCard, options)

    // `-ngl` counts the output head, so N layers on the GPU is N + 1.
    expect(args.ngl).toBe(bands[2].gpuLayers + 1)
    expect(args.overrideTensorBufferType).toBeUndefined()
    expect(args.tensorSplit).toBeUndefined()
  })

  it('always names the devices it budgeted', () => {
    // Left unset, llama.cpp spreads the model over every card it can see and
    // spends memory the plan never counted.
    const machine = {
      gpus: [card(8), card(8, 0, 1)],
      ramBytes: 64 * GiB,
      unified: false,
      gpuIndices: [1],
    }
    const bands = planContextBands({ shape: denseShape(), machine, options })

    expect(placementArgs(denseShape(), bands[0], machine, options).device).toBe('CUDA1')
  })

  it('keeps every MoE layer on the GPU and moves only expert tensors', () => {
    const shape = moeShape()
    const args = placementArgs(
      shape,
      { gpuLayers: 3, maxContext: 16384, gpuActiveBytes: 0, cpuActiveBytes: 0 },
      smallCard,
      options
    )

    expect(args.ngl).toBe(shape.layerCount + 1)
    // 3 layers keep their experts, so the other 45 give theirs back. Only one
    // card is in play here, so nothing is pinned to a device.
    const given = Array.from({ length: shape.layerCount - 3 }, (_, i) => i)
    expect(args.overrideTensorBufferType).toBe(
      'token_embd.weight=CPU,' +
        `blk\\.(${given.join('|')})\\.ffn_.*_exps.*=CPU`
    )
  })

  it('omits the CUDA rule when no expert layer stays on the GPU', () => {
    const shape = moeShape()
    const args = placementArgs(
      shape,
      { gpuLayers: 0, maxContext: 40960, gpuActiveBytes: 0, cpuActiveBytes: 0 },
      smallCard,
      options
    )

    // Every layer's experts move, so every layer is named.
    expect(args.overrideTensorBufferType).toContain('|47)')
    expect(args.overrideTensorBufferType).not.toContain('CUDA0')
  })

  it('gives each card whole layers, sized by bytes and not by count', () => {
    // A layer that kept its experts is several times the size of one that gave
    // them back, so dealing by count fills one card and OOMs the other. And a
    // layer split across two cards aborts the load outright.
    const machine = {
      gpus: [card(3), card(3, 0, 1)],
      ramBytes: 64 * GiB,
      unified: false,
      gpuIndices: [0, 1],
    }
    const args = placementArgs(
      moeShape(),
      { gpuLayers: 9, maxContext: 16384, gpuActiveBytes: 0, cpuActiveBytes: 0 },
      machine,
      options
    )

    const rules = args.overrideTensorBufferType!.split(',')
    const perCard = rules.filter((rule) => rule.includes('CUDA'))
    const named = perCard.flatMap((rule) =>
      rule.slice(rule.indexOf('(') + 1, rule.indexOf(')')).split('|')
    )
    expect(named).toEqual(
      Array.from({ length: moeShape().layerCount }, (_, i) => String(i))
    )
    expect(perCard[0]).toContain('=CUDA0')
    expect(perCard[1]).toContain('=CUDA1')
  })

})

describe('recommendedGpuIndices', () => {
  const twoCards = {
    gpus: [card(12), card(6, 0, 1)],
    ramBytes: 64 * GiB,
    unified: false,
  }

  it('uses one card when the model fits on it', () => {
    // Measured: splitting a model that already fits halved throughput.
    const small = denseShape({
      layers: Array.from({ length: 32 }, () => ({
        total: 0.05 * GiB,
        experts: 0,
        hasKvCache: true,
      })),
    })

    expect(recommendedGpuIndices(small, twoCards, options)).toEqual([0])
  })

  it('adds a card when the weights would otherwise spill to the CPU', () => {
    expect(recommendedGpuIndices(moeShape(), twoCards, options)).toEqual([0, 1])
  })
})

describe('bandForContext', () => {
  it('picks the fastest band that reaches the request', () => {
    const bands = planContextBands({
      shape: denseShape(),
      machine: smallCard,
      options,
    })
    // One token past the first ceiling must land on the second band, never on
    // a slower one that also happens to reach it.
    const target = bands[0].maxContext + 1
    const band = bandForContext(bands, target)!

    expect(band).toBe(bands[1])
    expect(bands[0].maxContext).toBeLessThan(target)
  })
})


describe('VRAM accounting', () => {
  it('reads free VRAM per card instead of assuming a fixed reserve', () => {
    // Monitors on a secondary card: the compute card is nearly all free, and a
    // flat 1 GiB reserve would throw away context that is actually available.
    const machine = {
      gpus: [card(24, 0.5), card(8, 2, 1)],
      ramBytes: 64 * GiB,
      unified: false,
    }
    const [compute, display] = freeVramByGpu(machine)

    expect(compute).toBeCloseTo(22.75 * GiB, -6)
    expect(display).toBeCloseTo(5.25 * GiB, -6)
    expect(defaultMainGpuIndex(machine)).toBe(0)
  })

  it('plans against the card with the most free VRAM, not the largest card', () => {
    const shape = moeShape()
    const bands = planContextBands({
      shape,
      // The bigger card is almost entirely occupied by something else.
      machine: {
        gpus: [card(24, 22), card(12, 0.5, 1)],
        ramBytes: 64 * GiB,
        unified: false,
      },
      options,
    })
    const onSmallCardAlone = planContextBands({
      shape,
      machine: { gpus: [card(12, 0.5)], ramBytes: 64 * GiB, unified: false },
      options,
    })

    expect(bands).toEqual(onSmallCardAlone)
  })

  it('does not pool VRAM across cards, since split placement is not built', () => {
    const shape = moeShape()
    const twoCards = planContextBands({
      shape,
      machine: { gpus: [card(8), card(8, 0, 1)], ramBytes: 64 * GiB, unified: false },
      options,
    })
    const oneCard = planContextBands({
      shape,
      machine: { gpus: [card(8)], ramBytes: 64 * GiB, unified: false },
      options,
    })

    expect(twoCards).toEqual(oneCard)
  })
})

describe('RAM accounting', () => {
  // 16 GiB of RAM behind an 8 GiB card: roomy enough that the VRAM side
  // binds the fast bands, tight enough that RAM binds the offloaded tail
  // below the trained cap.
  const ramBound = {
    gpus: [card(8)],
    ramBytes: 16 * GiB,
    unified: false,
  }

  it('budgets RAM from measured use instead of the fixed reserve alone', () => {
    // 10 GiB already in use leaves 6, not the 13 the reserve alone assumes.
    // Idle the tail reaches the trained cap; busy it binds on RAM below it.
    const shape = denseShape()
    const idle = planContextBands({ shape, machine: ramBound, options })
    const busy = planContextBands({
      shape,
      machine: { ...ramBound, ramUsedBytes: 10 * GiB },
      options,
    })

    const idleLast = idle[idle.length - 1].maxContext
    const busyLast = busy[busy.length - 1].maxContext
    expect(idleLast).toBe(131072)
    expect(busyLast).toBeLessThan(idleLast)
    expect(busyLast).toBeLessThan(131072)
    expect(busy.length).toBeLessThanOrEqual(idle.length)
    // The all-GPU head band touches no RAM, so it is identical.
    expect(busy[0]).toEqual(idle[0])
  })

  it('keeps the fixed reserve as a floor below measured use', () => {
    // 1 GiB in use on an idle-looking box changes nothing: the 3 GiB reserve
    // still binds.
    const shape = denseShape()
    const idle = planContextBands({ shape, machine: ramBound, options })
    const quiet = planContextBands({
      shape,
      machine: { ...ramBound, ramUsedBytes: 1 * GiB },
      options,
    })

    expect(quiet).toEqual(idle)
  })

  it('charges host compute and scratch to RAM by CPU share', () => {
    // RAM binds the tail below the trained cap here, so the tail ceiling
    // moves with the host terms, which scale with the micro-batch. (The head
    // bands move with ubatch too, but through the VRAM-side scratch — a
    // different mechanism, already covered.)
    const machine = { ...ramBound, ramUsedBytes: 10 * GiB }
    const shape = denseShape()
    const light = planContextBands({
      shape,
      machine,
      options: { ...options, ubatch: 512 },
    })
    const fast = planContextBands({
      shape,
      machine,
      options: { ...options, ubatch: 2048 },
    })

    expect(light[light.length - 1].maxContext).toBeLessThan(131072)
    expect(fast[fast.length - 1].maxContext).toBeLessThan(
      light[light.length - 1].maxContext
    )
  })

  it('leaves a purely recurrent model unbounded by RAM', () => {
    // No KV cache, no per-token host scratch: context is free up to training.
    const shape = denseShape({
      kvElementsPerToken: 0,
      kvElementsPerLayerToken: 0,
      layers: Array.from({ length: 32 }, () => ({
        total: 0.14 * GiB,
        experts: 0,
        hasKvCache: false,
      })),
      recurrentStateBytesPerSeq: 1024,
      recurrentLayerCount: 32,
    })
    const bands = planContextBands({ shape, machine: ramBound, options })

    expect(bands.length).toBeGreaterThan(0)
    expect(bands[bands.length - 1].maxContext).toBe(131072)
  })

  it('carries a RAM-use reading through machineMemoryFrom', () => {
    const hardware = {
      cpu: { arch: '', core_count: 0, extensions: [], name: '', usage: 0 },
      gpus: [],
      os_type: '',
      os_name: '',
      total_memory: 32768,
    }
    const devices = [
      { id: 'CUDA0', name: 'Card 0', mem: 8192, free: 8192, activated: true },
    ]

    expect(
      machineMemoryFrom(devices, hardware, {}, 10240).ramUsedBytes
    ).toBe(10240 * 1024 * 1024)
    expect(
      machineMemoryFrom(devices, hardware, {}).ramUsedBytes
    ).toBeUndefined()
  })
})

describe('resident-model planning', () => {
  // Rocinante-XL-16B geometry: 54 dense Mistral-Nemo layers (155.6 MiB each),
  // embedding 5120, 8 KV heads. Regression test for a session running this
  // model fine on 2x8 GiB cards (5.8 + 6.2 GiB in use, 22.7 of 32 GiB RAM)
  // while the settings sheet insisted it fits nowhere, not even on the CPU.
  function rocinanteShape(): ModelShape {
    return {
      arch: 'llama',
      layerCount: 54,
      embeddingLength: 5120,
      trainedContext: 131072,
      kvElementsPerToken: 54 * 8 * (160 + 160),
      kvElementsPerLayerToken: 8 * (160 + 160),
      layers: Array.from({ length: 54 }, () => ({
        total: 155.6 * 1024 * 1024,
        experts: 0,
        hasKvCache: true,
      })),
      nonLayerBytes: 0.86 * GiB,
      outputBytes: 0.5 * GiB,
      isMoe: false,
      expertActiveRatio: 1,
      recurrentStateBytesPerSeq: 0,
      recurrentLayerCount: 0,
    }
  }
  const MIB = 1024 * 1024
  const fastQ8 = { kvCacheType: 'q8_0', ubatch: 2048 }

  it('finds no band when the snapshot counts the running model as unavailable', () => {
    // Loaded-state readings: the model's own VRAM/RAM counted against it,
    // plus a display reserve on each card (loaded usage trips the display
    // heuristic). Nothing fits, not even fully offloaded.
    const loaded = {
      gpus: [
        { ...card(8, 5940 / 1024), reserveBytes: 512 * MIB },
        { ...card(8, 6413 / 1024, 1), reserveBytes: 512 * MIB },
      ],
      ramBytes: 32 * GiB,
      ramUsedBytes: 22.7 * GiB,
      unified: false,
      gpuIndices: [0, 1],
    }

    expect(
      planContextBands({ shape: rocinanteShape(), machine: loaded, options: fastQ8 })
    ).toEqual([])
  })

  it('plans bands once the resident footprint is budgeted as freed', () => {
    // What the sheet must plan against for a resident model: the next load
    // happens after it unloads, so its footprint is free and only the
    // user-set reserve applies.
    const unloaded = {
      gpus: [card(8, 0), { ...card(8, 0, 1), reserveBytes: 512 * MIB }],
      ramBytes: 32 * GiB,
      unified: false,
      gpuIndices: [0, 1],
    }

    const bands = planContextBands({
      shape: rocinanteShape(),
      machine: unloaded,
      options: fastQ8,
    })

    expect(bands.length).toBeGreaterThan(0)
    expect(bands[0].gpuLayers).toBe(54)
  })
})

describe('computeFloorForPlacement', () => {
  const machine = {
    gpus: [card(12)],
    ramBytes: 64 * GiB,
    unified: false,
  }

  it('inverts a matching measurement with margin', () => {
    // 600 MiB measured at ubatch 512, ctx 8192 on the planned card:
    // floor = 600MiB − 512·4096·20 − 512·2·8192, times 1.1.
    const floor = computeFloorForPlacement(denseShape(), machine, {
      devices: [{ device: 'CUDA0', computeBytes: 600 * 1024 * 1024 }],
      ubatch: 512,
      context: 8192,
    })!

    expect(floor).toBeCloseTo(
      (600 * 1024 * 1024 - 512 * 4096 * 20 - 512 * 2 * 8192) * 1.1,
      0
    )
  })

  it('ignores a measurement taken on a different card set', () => {
    const measured = {
      devices: [{ device: 'CUDA1', computeBytes: 600 * 1024 * 1024 }],
      ubatch: 512,
      context: 8192,
    }

    expect(computeFloorForPlacement(denseShape(), machine, measured)).toBeUndefined()
  })

  it('ignores failed and empty measurements', () => {
    expect(
      computeFloorForPlacement(denseShape(), machine, {
        devices: [{ device: 'CUDA0', computeBytes: 600 * 1024 * 1024 }],
        failure: 'cudaMalloc failed',
        ubatch: 512,
        context: 8192,
      })
    ).toBeUndefined()
    expect(
      computeFloorForPlacement(denseShape(), machine, {
        devices: [],
        ubatch: 512,
        context: 8192,
      })
    ).toBeUndefined()
    expect(computeFloorForPlacement(denseShape(), machine, undefined)).toBeUndefined()
  })
})

describe('autofitPlacement', () => {
  const machine = {
    gpus: [card(8)],
    ramBytes: 32 * GiB,
    unified: false,
  }
  const current = (values: Record<string, unknown>) => (key: string) =>
    values[key]

  it('fits a stored request to the fastest reaching band', () => {
    const shape = denseShape()
    const first = planContextBands({ shape, machine, options })[0]
    const result = autofitPlacement({
      shape,
      machine,
      kvCacheType: 'q8_0',
      ubatch: 1024,
      sequences: 4,
      flashAttn: 'auto',
      devices: '',
      storedCtxLen: first.maxContext,
      ctxAuto: false,
      currentValue: current({}),
    })!

    expect(result.band).toEqual(first)
    expect(result.requested).toBe(first.maxContext)
    expect(result.shortfall).toBeUndefined()
    expect(result.patch.ngl).toBe(first.gpuLayers + 1)
    expect(result.patch.device).toBe('CUDA0')
    // A stored request is never rewritten.
    expect(result.patch.ctx_len).toBeUndefined()
  })

  it('reports an empty patch when already fitted', () => {
    const shape = denseShape()
    const first = planContextBands({ shape, machine, options })[0]
    const fitted = autofitPlacement({
      shape,
      machine,
      kvCacheType: 'q8_0',
      ubatch: 1024,
      sequences: 4,
      flashAttn: 'auto',
      devices: '',
      storedCtxLen: first.maxContext,
      ctxAuto: false,
      currentValue: current({}),
    })!
    const again = autofitPlacement({
      shape,
      machine,
      kvCacheType: 'q8_0',
      ubatch: 1024,
      sequences: 4,
      flashAttn: 'auto',
      devices: '',
      storedCtxLen: first.maxContext,
      ctxAuto: false,
      currentValue: current(fitted.patch),
    })!

    expect(again.patch).toEqual({})
    expect(again.shortfall).toBeUndefined()
  })

  it('loads best-effort with a shortfall instead of clamping the request', () => {
    const shape = denseShape()
    const bands = planContextBands({ shape, machine, options })
    const result = autofitPlacement({
      shape,
      machine,
      kvCacheType: 'q8_0',
      ubatch: 1024,
      sequences: 4,
      flashAttn: 'auto',
      devices: '',
      storedCtxLen: 1_000_000,
      ctxAuto: false,
      currentValue: current({}),
    })!

    expect(result.band).toEqual(bands[bands.length - 1])
    expect(result.shortfall).toEqual({
      requested: 1_000_000,
      fits: bands[bands.length - 1].maxContext,
    })
    expect(result.patch.ctx_len).toBeUndefined()
  })

  it('ignores the seeded request while the context is automatic', () => {
    const shape = denseShape()
    const first = planContextBands({ shape, machine, options })[0]
    // 8192 is what every model is seeded with at import, so on auto it says
    // nothing about what the user wants and the top band wins.
    const result = autofitPlacement({
      shape,
      machine,
      kvCacheType: 'q8_0',
      ubatch: 1024,
      sequences: 4,
      flashAttn: 'auto',
      devices: '',
      storedCtxLen: 8192,
      ctxAuto: true,
      currentValue: current({}),
    })!

    expect(result.requested).toBe(first.maxContext)
    expect(result.patch.ctx_len).toBe(first.maxContext)
    expect(result.shortfall).toBeUndefined()
  })

  it('records a default context only when none usable was stored', () => {
    const shape = denseShape()
    const first = planContextBands({ shape, machine, options })[0]
    const result = autofitPlacement({
      shape,
      machine,
      kvCacheType: 'q8_0',
      ubatch: 1024,
      sequences: 4,
      flashAttn: 'auto',
      devices: '',
      storedCtxLen: 0,
      ctxAuto: true,
      currentValue: current({}),
    })!

    expect(result.patch.ctx_len).toBe(first.maxContext)
    expect(result.shortfall).toBeUndefined()
  })

  it('returns nothing when nothing fits at a usable context', () => {
    expect(
      autofitPlacement({
        shape: moeShape(),
        machine: { gpus: [card(2)], ramBytes: 4 * GiB, unified: false },
        kvCacheType: 'q8_0',
        ubatch: 1024,
        sequences: 4,
        flashAttn: 'auto',
        devices: '',
        storedCtxLen: 8192,
        ctxAuto: true,
        currentValue: current({}),
      })
    ).toBeUndefined()
  })

  it('treats a mismatched calibration like no calibration', () => {
    const shape = denseShape()
    const base = {
      shape,
      machine,
      kvCacheType: 'q8_0',
      ubatch: 1024,
      sequences: 4,
      flashAttn: 'auto',
      devices: '',
      storedCtxLen: 65536,
      ctxAuto: false,
      currentValue: current({}),
    }
    const mismatched = autofitPlacement({
      ...base,
      calibrations: {
        'CUDA9|par4|faauto|be|mtpoff': {
          devices: [{ device: 'CUDA9', computeBytes: 100 * 1024 * 1024 }],
          ubatch: 1024,
          context: 8192,
        },
      },
    })!
    const uncalibrated = autofitPlacement({ ...base })!

    expect(mismatched.band).toEqual(uncalibrated.band)
    expect(mismatched.patch).toEqual(uncalibrated.patch)
  })

  it('uses a matching measurement to keep more layers on the GPU', () => {
    const shape = denseShape()
    const base = {
      shape,
      machine,
      kvCacheType: 'q8_0',
      ubatch: 1024,
      sequences: 4,
      flashAttn: 'auto',
      devices: '',
      storedCtxLen: 65536,
      ctxAuto: false,
      currentValue: current({}),
    }
    const plain = autofitPlacement({ ...base })!
    const measured = autofitPlacement({
      ...base,
      calibrations: {
        'CUDA0|par4|faauto|be|mtpoff': {
          devices: [{ device: 'CUDA0', computeBytes: 100 * 1024 * 1024 }],
          ubatch: 1024,
          context: 8192,
        },
      },
    })!

    // The measured floor is far below the fallback, so the same request
    // reaches a faster band — without changing the request itself.
    expect(measured.band.gpuLayers).toBeGreaterThan(plain.band.gpuLayers)
    expect(measured.shortfall).toBeUndefined()
  })
})

describe('calibration axes', () => {
  it('keys measurements by card set, parallel, flash-attn, backend, and MTP', () => {
    expect(
      calibrationKeyFor({
        deviceIds: ['CUDA1', 'CUDA0'],
        parallel: 4,
        flashAttn: 'auto',
        backend: 'b9967/cuda',
        mtp: false,
      })
    ).toBe('CUDA0+CUDA1|par4|faauto|beb9967/cuda|mtpoff')
    expect(
      calibrationKeyFor({
        deviceIds: [],
        parallel: 1,
        flashAttn: 'off',
        backend: '',
        mtp: true,
      })
    ).toBe('|par1|faoff|be|mtpon')
  })

  it('resolves effective parallel from override to global to auto', () => {
    expect(effectiveParallel(8, 2)).toBe(8)
    expect(effectiveParallel(undefined, 2)).toBe(2)
    expect(effectiveParallel(undefined, undefined)).toBe(4)
    expect(effectiveParallel(0, -1)).toBe(4)
    // Text inputs persist strings; the preset's own normalizer coerces them.
    expect(effectiveParallel('8', 2)).toBe(8)
    expect(effectiveParallel('', 2)).toBe(2)
  })

  it('resolves effective flash-attn from override to global to auto', () => {
    expect(effectiveFlashAttn('off', 'on')).toBe('off')
    expect(effectiveFlashAttn(undefined, 'on')).toBe('on')
    expect(effectiveFlashAttn(undefined, undefined)).toBe('auto')
    expect(effectiveFlashAttn('yes', 'on')).toBe('on')
  })

  it('resolves effective ubatch from override to global to default', () => {
    expect(effectiveUbatch(2048, 512)).toBe(2048)
    expect(effectiveUbatch('2048', 512)).toBe(2048)
    expect(effectiveUbatch(undefined, 256)).toBe(256)
    expect(effectiveUbatch(undefined, undefined)).toBe(512)
    expect(effectiveUbatch(0, 256)).toBe(256)
    expect(effectiveUbatch('', 256)).toBe(256)
  })

  it('resolves effective cache type from override to global to default', () => {
    expect(effectiveCacheType('f16', 'q8_0')).toBe('f16')
    expect(effectiveCacheType(undefined, 'q5_1')).toBe('q5_1')
    expect(effectiveCacheType(undefined, undefined)).toBe('q8_0')
    expect(effectiveCacheType('', 'q8_0')).toBe('q8_0')
  })
})

describe('resolveContext', () => {
  it('passes a typed value through without rounding it to the ceiling', () => {
    // Sitting just under a ceiling is how a user keeps the faster placement.
    const bands = planContextBands({
      shape: moeShape(),
      machine: smallCard,
      options,
    })
    const ceiling = bands[2].maxContext
    const resolved = resolveContext(bands, ceiling - 480)!

    expect(resolved.contextLength).toBe(ceiling - 480)
    expect(resolved.band).toBe(bands[2])
    expect(resolved.freeHeadroom).toBe(480)
  })

  it('reports no headroom when the request is exactly a ceiling', () => {
    const bands = planContextBands({
      shape: moeShape(),
      machine: smallCard,
      options,
    })
    const resolved = resolveContext(bands, bands[1].maxContext)!

    expect(resolved.band).toBe(bands[1])
    expect(resolved.freeHeadroom).toBe(0)
  })

  it('moves to the next band when the request passes a ceiling by one token', () => {
    const bands = planContextBands({
      shape: moeShape(),
      machine: smallCard,
      options,
    })
    const resolved = resolveContext(bands, bands[0].maxContext + 1)!

    expect(resolved.band).toBe(bands[1])
    expect(resolved.band.gpuLayers).toBeLessThan(bands[0].gpuLayers)
  })

  it('clamps a request below the usable floor', () => {
    const bands = planContextBands({
      shape: moeShape(),
      machine: smallCard,
      options,
    })

    expect(resolveContext(bands, 512)!.contextLength).toBe(4096)
  })

  it('returns nothing when no band reaches the request', () => {
    const bands = planContextBands({
      shape: moeShape(),
      machine: smallCard,
      options,
    })

    expect(resolveContext(bands, 1_000_000)).toBeUndefined()
  })
})

describe('host-only planning', () => {
  // A 7B-class dense model on a 32 GiB machine with no GPU at all.
  const noGpu = { gpus: [], ramBytes: 32 * GiB, unified: false }

  it('selects no cards instead of falling back to card zero', () => {
    expect(plannedGpuIndices(noGpu)).toEqual([])
  })

  it('plans bands on RAM alone instead of reporting no fit', () => {
    const bands = planContextBands({
      shape: denseShape(),
      machine: noGpu,
      options,
    })

    expect(bands.length).toBeGreaterThan(0)
    // Only the fully-hosted placement exists.
    expect(bands.every((b) => b.gpuLayers === 0)).toBe(true)
  })

  it('emits ngl 0 with no device line', () => {
    const bands = planContextBands({
      shape: denseShape(),
      machine: noGpu,
      options,
    })
    const args = placementArgs(
      denseShape(),
      bands[bands.length - 1],
      noGpu,
      options
    )

    expect(args.ngl).toBe(0)
    expect(args.device).toBe('')
  })
})

describe('single allocations on the main card', () => {
  const wideShape = () =>
    denseShape({
      layerCount: 40,
      embeddingLength: 5120,
      trainedContext: 1000000,
      kvElementsPerToken: 40 * 8 * 256,
      kvElementsPerLayerToken: 8 * 256,
      layers: Array.from({ length: 40 }, () => ({
        total: 0.2 * GiB,
        experts: 0,
        hasKvCache: true,
      })),
    })
  const wide = { ...options, computeFloorBytes: 200 * 1024 * 1024 }
  const top = (
    machine: Parameters<typeof planContextBands>[0]['machine'],
    mmprojBytes = 0
  ) =>
    planContextBands({
      shape: { ...wideShape(), mmprojBytes },
      machine,
      options: wide,
    }).slice(-1)[0]!.maxContext

  it('charges a projector once, not once per card', () => {
    // The projector is one allocation on the first device in `--device`, like
    // the output head. Charged against every card, a 2 GiB projector cost a
    // two-card plan more than twice what it cost a single card.
    const two = {
      gpus: [card(12, 0, 0), card(12, 0, 1)],
      ramBytes: 32 * GiB,
      ramUsedBytes: 6 * GiB,
      unified: false,
      gpuIndices: [0, 1],
    }
    const one = { ...two, gpus: [card(12, 0, 0)], gpuIndices: [0] }
    const twoCardCost = top(two) - top(two, 2 * GiB)
    const oneCardCost = top(one) - top(one, 2 * GiB)

    expect(twoCardCost).toBeGreaterThan(0)
    expect(twoCardCost).toBeLessThanOrEqual(oneCardCost * 1.1)
  })
})

describe('autoBand', () => {
  const band = (gpuLayers: number, maxContext: number, cpuActiveBytes: number) => ({
    gpuLayers,
    maxContext,
    gpuActiveBytes: 0,
    cpuActiveBytes,
  })

  it('takes the roomiest band whose bus traffic stays close to the fastest', () => {
    const bands = [
      band(27, 13116, 1015),
      band(26, 27635, 1063),
      band(25, 40960, 1111),
      band(24, 60000, 2000),
    ]

    expect(autoBand(bands)!.maxContext).toBe(40960)
  })

  it('never moves the first layer off a fully resident model', () => {
    // Zero bus bytes is the one step that turns no PCIe traffic into some,
    // which no ratio against zero can measure.
    const bands = [band(32, 8192, 0), band(31, 60000, 500)]

    expect(autoBand(bands)!.gpuLayers).toBe(32)
  })

  it('is the fastest band when there is only one', () => {
    expect(autoBand([band(20, 9000, 400)])!.maxContext).toBe(9000)
    expect(autoBand([])).toBeUndefined()
  })
})

describe('integrated GPUs', () => {
  const hardware = (totalMiB: number) => ({
    cpu: { arch: 'x86_64', core_count: 8, extensions: [], name: '', usage: 0 },
    gpus: [],
    os_type: 'windows',
    os_name: '',
    total_memory: totalMiB,
  })
  const hwGpu = (
    name: string,
    opts: { computeCapability?: string; deviceType?: string } = {}
  ) => ({
    name,
    total_memory: 0,
    vendor: '',
    uuid: '',
    driver_version: '',
    nvidia_info: { index: 0, compute_capability: opts.computeCapability ?? '' },
    vulkan_info: {
      index: 0,
      device_id: 0,
      device_type: opts.deviceType ?? '',
      api_version: '1.3',
    },
  })
  const hardwareWith = (totalMiB: number, gpus: ReturnType<typeof hwGpu>[]) => ({
    ...hardware(totalMiB),
    gpus,
  })

  it('treats a device over 60% of system RAM as integrated', () => {
    expect(isIntegratedDevice(30 * GiB, 32 * GiB)).toBe(true)
    expect(isIntegratedDevice(8 * GiB, 32 * GiB)).toBe(false)
    expect(isIntegratedDevice(8 * GiB, 0)).toBe(false)
  })

  it('lets a resolved kind win over the size heuristic', () => {
    expect(isIntegratedGpu('integrated', 8 * GiB, 32 * GiB)).toBe(true)
    expect(isIntegratedGpu('discrete', 30 * GiB, 32 * GiB)).toBe(false)
    expect(isIntegratedGpu(undefined, 30 * GiB, 32 * GiB)).toBe(true)
    expect(isIntegratedGpu(undefined, 8 * GiB, 32 * GiB)).toBe(false)
  })

  it('falls back to the backend the device id names', () => {
    // The kind map matches on GPU name, and the two enumerations do not
    // always spell a card the same way. CUDA never enumerates an iGPU, so
    // the id alone keeps a 24 GB card on 32 GB of RAM out of the size rule.
    expect(deviceKindFromId('CUDA0')).toBe('discrete')
    expect(deviceKindFromId('ROCm1')).toBe('discrete')
    expect(deviceKindFromId('Vulkan0')).toBeUndefined()
    expect(deviceKindFromId('SYCL0')).toBeUndefined()

    const machine = machineMemoryFrom(
      [{ id: 'CUDA0', name: 'RTX 4090', mem: 24576, free: 24000, activated: true }],
      hardwareWith(32768, [hwGpu('NVIDIA GeForce RTX 4090 (different name)')]),
      {}
    )

    expect(machine.gpus[0].kind).toBe('discrete')
    expect(machine.unified).toBe(false)
  })

  it('reads an all-integrated machine as unified', () => {
    const machine = machineMemoryFrom(
      [
        {
          id: 'VULKAN0',
          name: 'AMD Raphael',
          mem: 30000,
          free: 28000,
          activated: true,
        },
      ],
      hardware(32768),
      {}
    )

    expect(machine.unified).toBe(true)
  })

  it('keeps a discrete card discrete', () => {
    const machine = machineMemoryFrom(
      [
        {
          id: 'CUDA0',
          name: 'RTX 3070',
          mem: 8192,
          free: 7000,
          activated: true,
        },
      ],
      hardware(32768),
      {}
    )

    expect(machine.unified).toBe(false)
  })

  it('trusts a DiscreteGpu device_type over the size heuristic', () => {
    // Regression: a 24 GB discrete card on 32 GB of RAM is 75% of system
    // RAM, so the bare heuristic read it integrated, zeroed usable RAM,
    // and planned no bands at all.
    const machine = machineMemoryFrom(
      [
        {
          id: 'CUDA0',
          name: 'RTX 4090',
          mem: 24576,
          free: 23000,
          activated: true,
        },
      ],
      hardwareWith(32768, [hwGpu('RTX 4090', { deviceType: 'DiscreteGpu' })]),
      {}
    )

    expect(machine.gpus[0].kind).toBe('discrete')
    expect(machine.unified).toBe(false)
    expect(
      planContextBands({ shape: denseShape(), machine, options }).length
    ).toBeGreaterThan(0)
  })

  it('trusts an IntegratedGpu device_type even for a small carve-out', () => {
    // The heuristic alone reads a 512 MB carve-out as discrete.
    const machine = machineMemoryFrom(
      [
        {
          id: 'VULKAN0',
          name: 'Radeon 780M',
          mem: 512,
          free: 400,
          activated: true,
        },
      ],
      hardwareWith(32768, [hwGpu('Radeon 780M', { deviceType: 'IntegratedGpu' })]),
      {}
    )

    expect(machine.gpus[0].kind).toBe('integrated')
    expect(machine.unified).toBe(true)
  })

  it('treats a reported NVIDIA compute capability as discrete', () => {
    const machine = machineMemoryFrom(
      [
        {
          id: 'CUDA0',
          name: 'RTX 4090',
          mem: 24576,
          free: 23000,
          activated: true,
        },
      ],
      hardwareWith(32768, [hwGpu('RTX 4090', { computeCapability: '8.9' })]),
      {}
    )

    expect(machine.gpus[0].kind).toBe('discrete')
    expect(machine.unified).toBe(false)
  })

  it('plans a mixed machine against the discrete card alone', () => {
    const machine = {
      gpus: [
        card(8),
        {
          id: 'VULKAN0',
          name: 'iGPU',
          totalBytes: 30 * GiB,
          usedBytes: 2 * GiB,
        },
      ],
      ramBytes: 32 * GiB,
      unified: false,
    }

    expect(recommendedGpuIndices(denseShape(), machine, options)).toEqual([0])
  })

  it('plans a kind-classified mixed machine against the discrete card alone', () => {
    // The 8 GB card is under the heuristic line; the iGPU is classified by
    // kind, so a heap of shared memory never joins the pool.
    const machine = machineMemoryFrom(
      [
        { id: 'CUDA0', name: 'RTX 3070', mem: 8192, free: 7000, activated: true },
        { id: 'VULKAN0', name: 'Radeon 780M', mem: 512, free: 400, activated: true },
      ],
      hardwareWith(32768, [
        hwGpu('RTX 3070', { deviceType: 'DiscreteGpu' }),
        hwGpu('Radeon 780M', { deviceType: 'IntegratedGpu' }),
      ]),
      {}
    )

    expect(machine.unified).toBe(false)
    expect(recommendedGpuIndices(denseShape(), machine, options)).toEqual([0])
  })

  it('never budgets more than physical RAM on integrated-only', () => {
    const machine = machineMemoryFrom(
      [
        {
          id: 'VULKAN0',
          name: 'AMD Raphael',
          mem: 30000,
          free: 28000,
          activated: true,
        },
      ],
      hardware(32768),
      {}
    )
    const free = freeVramByGpu(machine).reduce((sum, b) => sum + b, 0)
    const ram = machine.unified
      ? 0
      : Math.max(0, machine.ramBytes - 3 * GiB)

    expect(free + ram).toBeLessThanOrEqual(machine.ramBytes)
  })
})

describe('vision projector', () => {
  const mmproj = 1.2 * GiB
  const machine = { gpus: [card(8)], ramBytes: 32 * GiB, unified: false }
  // The all-CPU band is RAM-bound either way; the projector moves the
  // VRAM-bound fastest band, so that is what the comparison reads.
  const fastest = (bands: { maxContext: number }[]) => bands[0].maxContext

  it('plans a lower ceiling with the projector on the GPU than on the host', () => {
    const onGpu = planContextBands({
      shape: denseShape({ mmprojBytes: mmproj }),
      machine,
      options: { ...options, mmprojOffload: true },
    })
    const onHost = planContextBands({
      shape: denseShape({ mmprojBytes: mmproj }),
      machine,
      options: { ...options, mmprojOffload: false },
    })

    expect(onGpu.length).toBeGreaterThan(0)
    expect(onHost.length).toBeGreaterThan(0)
    expect(fastest(onGpu)).toBeLessThan(fastest(onHost))
  })

  it('leaves a text-only model exactly as before', () => {
    const plain = planContextBands({ shape: denseShape(), machine, options })
    const flagged = planContextBands({
      shape: denseShape(),
      machine,
      options: { ...options, mmprojOffload: false },
    })

    expect(flagged).toEqual(plain)
  })
})

describe('sliding-window KV', () => {
  // Gemma-3-1B geometry: 26 layers, window 512, one global layer in six.
  function gemmaLike(): ModelShape {
    const perLayer = 8 * (128 + 128)
    return {
      arch: 'gemma3',
      layerCount: 26,
      embeddingLength: 2048,
      trainedContext: 1048576,
      kvElementsPerToken: 26 * perLayer,
      kvElementsPerLayerToken: perLayer,
      layers: Array.from({ length: 26 }, (_, i) => ({
        total: 0.05 * GiB,
        experts: 0,
        hasKvCache: true,
        swaWindow: i % 6 === 5 ? 0 : 512,
      })),
      nonLayerBytes: 0.3 * GiB,
      outputBytes: 0.2 * GiB,
      isMoe: false,
      expertActiveRatio: 1,
      recurrentStateBytesPerSeq: 0,
      recurrentLayerCount: 0,
    }
  }
  const machine = { gpus: [card(8)], ramBytes: 32 * GiB, unified: false }
  // The all-CPU band is RAM-bound either way; the window sizing moves the
  // VRAM-bound fastest band, so that is what the comparison reads.
  const fastest = (bands: { maxContext: number }[]) => bands[0].maxContext

  it('plans several times the context of full-cache charging', () => {
    const windowed = planContextBands({ shape: gemmaLike(), machine, options })
    const full = planContextBands({
      shape: {
        ...gemmaLike(),
        layers: gemmaLike().layers.map((l) => ({ ...l, swaWindow: 0 })),
      },
      machine,
      options,
    })

    expect(windowed.length).toBeGreaterThan(0)
    expect(fastest(windowed)).toBeGreaterThan(fastest(full) * 3)
  })

  it('switches the window sizing off under --swa-full', () => {
    const forced = planContextBands({
      shape: gemmaLike(),
      machine,
      options: { ...options, swaFull: true },
    })
    const full = planContextBands({
      shape: {
        ...gemmaLike(),
        layers: gemmaLike().layers.map((l) => ({ ...l, swaWindow: 0 })),
      },
      machine,
      options,
    })

    expect(forced).toEqual(full)
  })
})

describe('speculative decoding', () => {
  const machine = { gpus: [card(8)], ramBytes: 32 * GiB, unified: false }
  // The all-CPU band is RAM-bound either way; the draft weights move the
  // VRAM-bound fastest band, so that is what the comparison reads.
  const fastest = (bands: { maxContext: number }[]) => bands[0].maxContext

  it('lowers band ceilings when MTP brings a draft model', () => {
    const plain = planContextBands({ shape: denseShape(), machine, options })
    const withDraft = planContextBands({
      shape: denseShape(),
      machine,
      options: { ...options, mtp: true, mtpDraftBytes: 2 * GiB },
    })

    expect(withDraft.length).toBeGreaterThan(0)
    expect(fastest(withDraft)).toBeLessThan(fastest(plain))
  })

  it('charges the second graph of an embedded MTP block', () => {
    // No draft file, so no weights and no cache of its own — but the head
    // still runs over every micro-batch, which is what OOMed a 27B whose
    // plan charged embedded MTP nothing at all.
    const plain = planContextBands({ shape: denseShape(), machine, options })
    const embedded = planContextBands({
      shape: denseShape(),
      machine,
      options: { ...options, mtp: true },
    })

    expect(embedded.length).toBeGreaterThan(0)
    expect(fastest(embedded)).toBeLessThan(fastest(plain))
  })

  it('charges the draft context beside its weights', () => {
    const weightsOnly = planContextBands({
      shape: denseShape(),
      machine,
      options: { ...options, mtp: true, mtpDraftBytes: 2 * GiB },
    })
    const withKv = planContextBands({
      shape: denseShape(),
      machine,
      options: {
        ...options,
        mtp: true,
        mtpDraftBytes: 2 * GiB,
        mtpDraftKvPerToken: 4096,
      },
    })
    // Same placement on both sides: the draft context only ever lowers a
    // ceiling (here it also drops the marginal fastest band under the
    // usable floor, so bands[0] itself is not comparable).
    const at = (
      bands: { gpuLayers: number; maxContext: number }[],
      layers: number
    ) => bands.find((b) => b.gpuLayers === layers)!.maxContext

    expect(withKv.length).toBeGreaterThan(0)
    expect(at(withKv, 28)).toBeLessThan(at(weightsOnly, 28))
  })
})

describe('rope scaling', () => {
  // Memory-rich: the trained-context cap binds, not the budget.
  const machine = { gpus: [card(24)], ramBytes: 64 * GiB, unified: false }

  it('doubles the ceiling at factor 2 and restores it when off', () => {
    const capped = planContextBands({ shape: denseShape(), machine, options })
    const scaled = planContextBands({
      shape: denseShape(),
      machine,
      options: { ...options, ropeFactor: 2 },
    })

    expect(capped[capped.length - 1].maxContext).toBe(131072)
    expect(scaled[scaled.length - 1].maxContext).toBe(262144)
  })

  it('resolves the factor from rope_scale, else 1 / rope_freq_scale', () => {
    expect(effectiveRopeFactor('linear', 2, 1)).toBe(2)
    expect(effectiveRopeFactor('yarn', 1, 0.5)).toBe(2)
    expect(effectiveRopeFactor('none', 2, 1)).toBe(1)
    expect(effectiveRopeFactor(undefined, 2, 1)).toBe(1)
    expect(effectiveRopeFactor('linear', undefined, undefined)).toBe(1)
  })
})

describe('calibration axes additions', () => {
  const machine = { gpus: [card(8)], ramBytes: 32 * GiB, unified: false }
  const base = {
    shape: denseShape(),
    machine,
    kvCacheType: 'q8_0',
    ubatch: 1024,
    sequences: 4,
    flashAttn: 'auto',
    devices: '',
    storedCtxLen: 65536,
    ctxAuto: false,
    currentValue: () => undefined,
  }
  const tinyFloor = (backend: string, mtp: boolean) => ({
    [calibrationKeyFor({
      deviceIds: ['CUDA0'],
      parallel: 4,
      flashAttn: 'auto',
      backend,
      mtp,
    })]: {
      devices: [{ device: 'CUDA0', computeBytes: 100 * 1024 * 1024 }],
      ubatch: 1024,
      context: 8192,
    },
  })

  it('ignores a floor measured on another backend build', () => {
    const measured = autofitPlacement({
      ...base,
      backend: 'b9967/cuda',
      calibrations: tinyFloor('b9967/cuda', false),
    })!
    const otherBuild = autofitPlacement({
      ...base,
      backend: 'b9999/cuda',
      calibrations: tinyFloor('b9967/cuda', false),
    })!
    const uncalibrated = autofitPlacement({ ...base })!

    expect(measured.band.gpuLayers).toBeGreaterThan(
      uncalibrated.band.gpuLayers
    )
    expect(otherBuild.band).toEqual(uncalibrated.band)
  })

  it('reuses a floor measured with MTP in the other state', () => {
    // The MTP term is explicit in the estimate, so the floor under it is the
    // same measurement either way — discarding it the moment MTP was
    // switched on dropped the plan back to the conservative fallback.
    const measured = autofitPlacement({
      ...base,
      calibrations: tinyFloor('', true),
      mtp: true,
    })!
    const otherState = autofitPlacement({
      ...base,
      calibrations: tinyFloor('', true),
      mtp: false,
    })!
    const uncalibrated = autofitPlacement({ ...base })!

    expect(measured.band.gpuLayers).toBeGreaterThan(
      uncalibrated.band.gpuLayers
    )
    expect(otherState.band.gpuLayers).toBeGreaterThan(
      uncalibrated.band.gpuLayers
    )
  })

  it('resolves the backend id and the MTP state', () => {
    expect(effectiveBackend('b9967/cuda', 'b1', 'x')).toBe('b9967/cuda')
    expect(effectiveBackend('', 'b9967', 'cuda')).toBe('b9967/cuda')
    expect(effectiveBackend(undefined, undefined, undefined)).toBe('')
    expect(effectiveMtp(true)).toBe(true)
    expect(effectiveMtp(false)).toBe(false)
    expect(effectiveMtp(undefined)).toBe(false)
  })
})

import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  ensureCalibration,
  isProbeIdle,
  runCalibrationProbe,
} from '../calibrationRunner'
import { useModelCalibration } from '@/hooks/useModelCalibration'
import { useModelProvider } from '@/hooks/useModelProvider'
import { useHardware } from '@/hooks/useHardware'
import { useDownloadStore } from '@/hooks/useDownloadStore'
import { useAppState } from '@/hooks/useAppState'

const EMPTY_SHAPE = {
  embeddingLength: 4096,
  layerCount: 0,
  layers: [],
  kvElementsPerToken: 0,
  kvElementsPerLayerToken: 0,
  recurrentStateBytesPerSeq: 0,
  recurrentLayerCount: 0,
}

/** A model small enough that the 12 GiB card in `fakeHub` plans bands for it. */
const FITTABLE_SHAPE = {
  arch: 'llama',
  embeddingLength: 4096,
  layerCount: 4,
  layers: Array.from({ length: 4 }, () => ({
    total: 256 * 1024 ** 2,
    experts: 0,
    hasKvCache: true,
  })),
  kvElementsPerToken: 4096,
  kvElementsPerLayerToken: 1024,
  nonLayerBytes: 128 * 1024 ** 2,
  outputBytes: 64 * 1024 ** 2,
  isMoe: false,
  expertActiveRatio: 1,
  trainedContext: 32768,
  recurrentStateBytesPerSeq: 0,
  recurrentLayerCount: 0,
}

let mockShape: unknown = EMPTY_SHAPE

vi.mock('../gguf', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../gguf')>()),
  readModelShape: () => mockShape,
}))

const GiB = 1024 ** 3

function setting(value: unknown) {
  return { controller_props: { value } }
}

function seedProviderModel(
  modelId: string,
  values: Record<string, unknown> = {}
) {
  useModelProvider.setState({
    providers: [
      {
        provider: 'llamacpp',
        models: [
          {
            id: modelId,
            settings: Object.fromEntries(
              Object.entries(values).map(([k, v]) => [k, setting(v)])
            ),
          },
        ],
      },
    ],
  } as never)
}

function seedHardware() {
  useHardware.setState({
    hardwareData: {
      ...useHardware.getState().hardwareData,
      total_memory: 32768,
    },
    systemUsage: {
      ...useHardware.getState().systemUsage,
      used_memory: 4096,
    },
  })
}

function fakeHub(overrides: Record<string, unknown> = {}) {
  return {
    models: () => ({
      getActiveModels: vi.fn().mockResolvedValue([]),
      readModelGguf: vi.fn().mockResolvedValue({ kind: 'mock' }),
      calibrateModel: vi.fn().mockResolvedValue(undefined),
      ...((overrides.models ?? {}) as object),
    }),
    hardware: () => ({
      getLlamacppDevices: vi.fn().mockResolvedValue([
        {
          id: 'CUDA0',
          name: 'Card 0',
          mem: 12288,
          free: 12288,
          activated: true,
        },
      ]),
      getSystemUsage: vi.fn().mockResolvedValue({
        cpu: 0,
        used_memory: 4096,
        total_memory: 32768,
        gpus: [],
      }),
      ...((overrides.hardware ?? {}) as object),
    }),
  } as never
}

beforeEach(() => {
  vi.clearAllMocks()
  useModelCalibration.setState({ running: {}, byModel: {} })
  useModelProvider.setState({ providers: [], selectedModel: null })
  useDownloadStore.setState({
    downloads: {},
    localDownloadingModels: new Set(),
  })
  useAppState.setState({
    currentStreamThreadId: undefined,
    abortControllers: {},
    busyThreads: {},
    loadingModels: {},
    streamingContents: {},
  })
  seedHardware()
  mockShape = EMPTY_SHAPE
})

describe('runCalibrationProbe', () => {
  const report = {
    devices: [],
    ubatch: 512,
    context: 8192,
    measuredAt: Date.now(),
    backend: 'b1/cuda',
  }

  it('records a measurement under its axis key and clears the run', async () => {
    const models = {
      calibrateModel: vi.fn().mockResolvedValue(report),
    }

    const outcome = await runCalibrationProbe(
      models,
      'm1',
      'CUDA0|par4|faauto|be|mtpoff'
    )

    expect(outcome).toEqual({ status: 'measured', report })
    expect(useModelCalibration.getState().byModel.m1).toEqual({
      'CUDA0|par4|faauto|be|mtpoff': report,
    })
    expect(useModelCalibration.getState().running).toEqual({})
  })

  it('reports unsupported when the engine cannot measure', async () => {
    const models = { calibrateModel: vi.fn().mockResolvedValue(undefined) }

    expect(await runCalibrationProbe(models, 'm1', 'CUDA0|par4|faauto|be|mtpoff')).toEqual(
      {
        status: 'unsupported',
      }
    )
    expect(useModelCalibration.getState().byModel.m1).toBeUndefined()
    expect(useModelCalibration.getState().running).toEqual({})
  })

  it('resolves cancelled without recording', async () => {
    const models = {
      calibrateModel: vi
        .fn()
        .mockRejectedValue({ details: 'Calibration cancelled by user.' }),
    }

    expect(
      await runCalibrationProbe(models, 'm1', 'CUDA0|par4|faauto|be|mtpoff')
    ).toEqual({
      status: 'cancelled',
    })
    expect(useModelCalibration.getState().byModel.m1).toBeUndefined()
    expect(useModelCalibration.getState().running).toEqual({})
  })

  it('reports failures with the message', async () => {
    const models = {
      calibrateModel: vi.fn().mockRejectedValue(new Error('boom')),
    }

    const outcome = await runCalibrationProbe(models, 'm1', 'CUDA0|par4|faauto|be|mtpoff')

    expect(outcome.status).toBe('failed')
    expect(useModelCalibration.getState().running).toEqual({})
  })
})

describe('isProbeIdle', () => {
  it('is idle when nothing runs, streams, or sits loaded', async () => {
    await expect(isProbeIdle(fakeHub())).resolves.toBe(true)
  })

  it('is busy while a probe runs', async () => {
    useModelCalibration.setState({ running: { m1: Date.now() } })
    await expect(isProbeIdle(fakeHub())).resolves.toBe(false)
  })

  it('is busy while a chat streams', async () => {
    useAppState.setState({ currentStreamThreadId: 't1' })
    await expect(isProbeIdle(fakeHub())).resolves.toBe(false)
  })

  it('is busy while a model sits loaded', async () => {
    const hub = fakeHub({
      models: { getActiveModels: vi.fn().mockResolvedValue(['m1']) },
    })
    await expect(isProbeIdle(hub)).resolves.toBe(false)
  })
})

describe('ensureCalibration', () => {
  const values = { device: 'CUDA0', ctx_len: 8192 }

  it('probes an uncovered model when idle', async () => {
    seedProviderModel('m1', values)
    const calibrateModel = vi.fn().mockResolvedValue({
      devices: [{ device: 'CUDA0', computeBytes: 100 }],
      ubatch: 512,
      context: 8192,
      measuredAt: Date.now(),
      backend: 'b1/cuda',
    })
    const hub = fakeHub({ models: { calibrateModel } })

    await ensureCalibration(hub, 'm1')

    expect(calibrateModel).toHaveBeenCalledWith('m1')
    expect(
      useModelCalibration.getState().byModel.m1['CUDA0|par4|faauto|be|mtpoff']
        .devices[0].device
    ).toBe('CUDA0')
    expect(useModelCalibration.getState().running).toEqual({})
  })

  it('skips an axis the stored measurement already covers', async () => {
    seedProviderModel('m1', values)
    useModelCalibration.setState({
      byModel: {
        m1: {
          'CUDA0|par4|faauto|be|mtpoff': {
            devices: [
              {
                device: 'CUDA0',
                modelBytes: 1,
                kvBytes: 0,
                recurrentBytes: 0,
                computeBytes: 600 * 1024 * 1024,
              },
            ],
            ubatch: 512,
            context: 8192,
            measuredAt: Date.now(),
            backend: 'b1/cuda',
          },
        },
      },
    })
    const calibrateModel = vi.fn()
    const hub = fakeHub({ models: { calibrateModel } })

    await ensureCalibration(hub, 'm1')

    expect(calibrateModel).not.toHaveBeenCalled()
  })

  it('skips hand-tuned and downloading models', async () => {
    const calibrateModel = vi.fn()
    const hub = fakeHub({ models: { calibrateModel } })

    seedProviderModel('m1', { ...values, plan_manual: true })
    await ensureCalibration(hub, 'm1')

    seedProviderModel('m3', values)
    useDownloadStore.getState().addLocalDownloadingModel('m3')
    await ensureCalibration(hub, 'm3')

    expect(calibrateModel).not.toHaveBeenCalled()
  })

  it('fits a model with no card selection instead of skipping it', async () => {
    // Nothing writes `device` until the first load, so skipping here left a
    // freshly imported model unmeasured through the one load it may ever get,
    // running on the conservative compute-buffer fallback.
    mockShape = FITTABLE_SHAPE
    const calibrateModel = vi.fn().mockResolvedValue(undefined)
    const updateModelSettings = vi.fn().mockResolvedValue(undefined)
    const hub = fakeHub({ models: { calibrateModel, updateModelSettings } })

    seedProviderModel('m2', { ctx_len: 8192 })
    await ensureCalibration(hub, 'm2')

    expect(calibrateModel).toHaveBeenCalledWith('m2')
    // The placement the first load would have written is written here, so the
    // probe measures the cards the plan actually uses.
    expect(updateModelSettings).toHaveBeenCalledWith(
      'm2',
      expect.objectContaining({ device: 'CUDA0' })
    )
    const stored = useModelProvider
      .getState()
      .getProviderByName('llamacpp')
      ?.models?.find((m) => m.id === 'm2')?.settings?.device?.controller_props
      ?.value
    expect(stored).toBe('CUDA0')
  })

  it('still skips a model with no card selection that cannot fit', async () => {
    const calibrateModel = vi.fn()
    const hub = fakeHub({ models: { calibrateModel } })

    seedProviderModel('m2', { ctx_len: 8192 })
    await ensureCalibration(hub, 'm2')

    expect(calibrateModel).not.toHaveBeenCalled()
  })

  it('never retries a failure on the same axis', async () => {
    seedProviderModel('m1', values)
    useModelCalibration.setState({
      byModel: {
        m1: {
          'CUDA0|par4|faauto|be|mtpoff': {
            devices: [
              {
                device: 'CUDA0',
                modelBytes: 1,
                kvBytes: 0,
                recurrentBytes: 0,
                computeBytes: 100,
              },
            ],
            failure: 'cudaMalloc failed',
            ubatch: 512,
            context: 8192,
            measuredAt: Date.now(),
            backend: 'b1/cuda',
          },
        },
      },
    })
    const calibrateModel = vi.fn()
    const hub = fakeHub({ models: { calibrateModel } })

    await ensureCalibration(hub, 'm1')

    expect(calibrateModel).not.toHaveBeenCalled()
  })

  it('tries an axis the stored failure was not measured on', async () => {
    // The failure is filed under CUDA1; the current selection is CUDA0.
    seedProviderModel('m1', values)
    useModelCalibration.setState({
      byModel: {
        m1: {
          'CUDA1|par4|faauto|be|mtpoff': {
            devices: [
              {
                device: 'CUDA1',
                modelBytes: 1,
                kvBytes: 0,
                recurrentBytes: 0,
                computeBytes: 100,
              },
            ],
            failure: 'cudaMalloc failed',
            ubatch: 512,
            context: 8192,
            measuredAt: Date.now(),
            backend: 'b1/cuda',
          },
        },
      },
    })
    const calibrateModel = vi.fn().mockResolvedValue({
      devices: [],
      ubatch: 512,
      context: 8192,
      measuredAt: Date.now(),
      backend: 'b1/cuda',
    })
    const hub = fakeHub({ models: { calibrateModel } })

    await ensureCalibration(hub, 'm1')

    expect(calibrateModel).toHaveBeenCalledWith('m1')
  })
})

/**
 * Model shape for the context planner, derived from GGUF metadata.
 *
 * The bytes come from `readGgufMetadata` in the llamacpp plugin, which reads
 * only the header — of a file on disk or of a URL, over range requests — so a
 * model can be planned before it is downloaded. Nothing here knows a model
 * family by name: a new architecture is new metadata, not a new branch.
 */

import type { GgufMetadata, GgufTensorInfo } from '@janhq/tauri-plugin-llamacpp-api'

/**
 * ggml type id -> [block size in elements, bytes per block]. Quantized formats
 * store a whole block with its scales, so the figure is per block rather than
 * per element. The ids are fixed by ggml and only ever appended to.
 */
const GGML_TYPES: Record<number, [number, number]> = {
  0: [1, 4], // F32
  1: [1, 2], // F16
  2: [32, 18], // Q4_0
  3: [32, 20], // Q4_1
  6: [32, 22], // Q5_0
  7: [32, 24], // Q5_1
  8: [32, 34], // Q8_0
  9: [32, 36], // Q8_1
  10: [256, 84], // Q2_K
  11: [256, 110], // Q3_K
  12: [256, 144], // Q4_K
  13: [256, 176], // Q5_K
  14: [256, 210], // Q6_K
  15: [256, 292], // Q8_K
  16: [256, 66], // IQ2_XXS
  17: [256, 74], // IQ2_XS
  18: [256, 98], // IQ3_XXS
  19: [256, 50], // IQ1_S
  20: [32, 18], // IQ4_NL
  21: [256, 110], // IQ3_S
  22: [256, 82], // IQ2_S
  23: [256, 136], // IQ4_XS
  24: [1, 1], // I8
  25: [1, 2], // I16
  26: [1, 4], // I32
  27: [1, 8], // I64
  28: [1, 8], // F64
  29: [256, 56], // IQ1_M
  30: [1, 2], // BF16
}

function tensorBytes(tensor: GgufTensorInfo): number {
  const elements = tensor.dims.reduce((a, b) => a * b, 1)
  const spec = GGML_TYPES[tensor.ggml_type]
  // An unknown id is a ggml type newer than this table. f16 keeps the estimate
  // in the right order of magnitude instead of reporting zero bytes.
  if (!spec) return elements * 2
  const [blockSize, blockBytes] = spec
  return (elements / blockSize) * blockBytes
}

export interface LayerBytes {
  /** Every `blk.N.*` tensor in this layer. */
  total: number
  /** The MoE expert tensors within it. Zero for a dense model. */
  experts: number
  /**
   * Whether this layer holds a cache that grows with the context. A recurrent
   * layer (`ssm_*`) keeps a fixed-size state instead, so it costs nothing per
   * token. Hybrid models interleave the two, and charging KV for every layer
   * overestimates the cache by the interleave ratio.
   */
  hasKvCache: boolean
  /**
   * Sliding-window size for this layer, or 0/undefined on a global (full
   * context) layer. Window layers keep only `min(context, window + ubatch)`
   * of cache (see `set_swa_pattern` in llama.cpp `src/llama-model.cpp` for
   * the interleaved patterns); without this every window layer is charged
   * the full context, which overstates Gemma-3's cache ~5.6x at 32k.
   */
  swaWindow?: number
}

export interface ModelShape {
  arch: string
  layerCount: number
  /** Hidden size. Drives the compute-buffer estimate in the planner. */
  embeddingLength: number
  /** Context the model was trained for. The planner never offers more. */
  trainedContext: number
  /**
   * K and V elements held per token across all layers. Multiply by the bytes
   * per element of the cache type for the KV cache cost of one token.
   */
  kvElementsPerToken: number
  /** K/V elements per token for one attention layer. */
  kvElementsPerLayerToken: number
  layers: LayerBytes[]
  /** Embeddings and output head together. */
  nonLayerBytes: number
  /**
   * The output head and final norm alone. llama.cpp offloads these as soon as
   * `-ngl` is at least 1 and leaves the token embedding on the CPU, so this is
   * the part of `nonLayerBytes` that lands in VRAM.
   */
  outputBytes: number
  isMoe: boolean
  /**
   * Share of expert weights read per token; 1 for a dense model. Everything
   * that is not an expert is read in full on every token.
   */
  expertActiveRatio: number
  /**
   * State one recurrent layer holds for one sequence, in bytes. A recurrent
   * layer keeps this instead of a KV cache: it does not grow with the context,
   * but it is charged once per sequence, and llama-server runs four sequences
   * unless told otherwise.
   *
   * This is llama.cpp's own `n_embd_r` and `n_embd_s`, both in f32. Checked
   * against the engine's reported `RS buffer size` on two models: 2.0938 MiB a
   * layer, 30 layers x 4 sequences = 251.25 MiB on Qwen3.6-35B-A3B and 24
   * layers x 1 = 50.25 MiB on Qwen3.5-9B.
   */
  recurrentStateBytesPerSeq: number
  /** Layers holding that state, counted from their `ssm_` tensors. */
  recurrentLayerCount: number
  /**
   * Vision projector weights in bytes (the `mmproj` file named by
   * `mmproj_path` in model.yml). Zero/undefined on a text-only model.
   * Whether these land in VRAM or RAM is decided by `offload_mmproj`, which
   * the planner reads beside the shape.
   */
  mmprojBytes?: number
}

/** Metadata values arrive as strings, including the numeric ones. */
function num(kv: Record<string, string>, key: string): number | undefined {
  const v = Number(kv[key])
  return Number.isFinite(v) ? v : undefined
}

/**
 * Global-layer period per architecture, copied from the `set_swa_pattern`
 * defaults in llama.cpp (`src/models/*.cpp`): with period P, layer `il` is
 * global exactly when `il % P === P - 1`, and every other layer keeps only
 * the sliding window. Newer GGUFs may carry an explicit
 * `<arch>.attention.sliding_window_pattern` instead, which wins when present.
 */
const SWA_PERIOD_BY_ARCH: Record<string, number> = {
  gemma2: 2,
  gemma3: 6,
  gemma3n: 5,
  cohere2: 4,
  cohere2moe: 4,
  'gpt-oss': 2,
  llama4: 4,
}

/**
 * Sliding-window size for one layer, or 0 for a global layer. Returns 0 for
 * every layer when no usable window is declared.
 */
function swaWindowForLayer(
  kv: Record<string, string>,
  arch: string,
  layerIndex: number,
  layerCount: number,
  period: number | undefined
): number {
  const window = num(kv, `${arch}.attention.sliding_window`)
  if (!window || window <= 0 || layerCount <= 0) return 0
  const raw = kv[`${arch}.attention.sliding_window_pattern`]
  if (raw !== undefined) {
    const text = raw.trim().replace(/^\[|\]$/g, '')
    if (/^\d+$/.test(text)) {
      const p = Number(text)
      if (p > 0) return layerIndex % p === p - 1 ? 0 : window
      return window
    }
    const flags = text
      .split(/[,;\s]+/)
      .filter(Boolean)
      .map((token) => /^(1|true|swa|local)$/i.test(token))
    if (flags.length === layerCount) {
      return flags[layerIndex] ? window : 0
    }
    if (flags.length === 1) return flags[0] ? window : 0
    // Unparseable pattern: fall through to the architecture default, if any.
  }
  if (period === undefined || period <= 0) return 0
  return layerIndex % period === period - 1 ? 0 : window
}

export function readModelShape(header: GgufMetadata): ModelShape {
  const kv = header.metadata
  const arch = kv['general.architecture'] ?? 'unknown'

  const layerCount = num(kv, `${arch}.block_count`) ?? 0
  const embedding = num(kv, `${arch}.embedding_length`) ?? 0
  const headCount = num(kv, `${arch}.attention.head_count`) ?? 0
  const kvHeads = num(kv, `${arch}.attention.head_count_kv`) ?? headCount
  // key_length and value_length are optional. Without them a head is the
  // embedding split evenly across the attention heads.
  const headDim = headCount > 0 ? embedding / headCount : 0
  const keyLength = num(kv, `${arch}.attention.key_length`) ?? headDim
  const valueLength = num(kv, `${arch}.attention.value_length`) ?? headDim

  const expertCount = num(kv, `${arch}.expert_count`) ?? 0
  const expertUsed = num(kv, `${arch}.expert_used_count`) ?? 0
  const isMoe = expertCount > 0 && expertUsed > 0

  const swaPeriod = SWA_PERIOD_BY_ARCH[arch]

  const layers: LayerBytes[] = Array.from({ length: layerCount }, (_, i) => ({
    total: 0,
    experts: 0,
    // Assumed until an `ssm_` tensor proves otherwise: an unknown architecture
    // is far likelier to be plain attention, and over-charging is the safe way
    // to be wrong.
    hasKvCache: true,
    swaWindow: swaWindowForLayer(kv, arch, i, layerCount, swaPeriod),
  }))
  let nonLayerBytes = 0
  let outputBytes = 0
  // Blocks carrying `ssm_` tensors, counted separately from `hasKvCache:
  // false`: an embedded MTP (nextn) block also holds no KV cache but keeps
  // no recurrent state either, so it must not attract a state charge.
  const ssmLayers = new Set<number>()
  for (const t of header.tensors) {
    const bytes = tensorBytes(t)
    const match = /^blk\.(\d+)\./.exec(t.name)
    if (!match) {
      nonLayerBytes += bytes
      if (!t.name.startsWith('token_embd')) outputBytes += bytes
      continue
    }
    const layer = layers[Number(match[1])]
    if (!layer) continue
    layer.total += bytes
    if (/ffn_.*_exps/.test(t.name)) layer.experts += bytes
    if (/\.ssm_/.test(t.name)) {
      layer.hasKvCache = false
      ssmLayers.add(Number(match[1]))
    }
    // An embedded MTP (nextn) block carries prediction tensors but no KV
    // cache of its own. `block_count` includes it, so without this it is
    // charged a full layer of cache (Qwen3.8-27B counted 65 KV layers for 64).
    if (/\.nextn\./.test(t.name)) layer.hasKvCache = false
  }

  const kvLayerCount = layers.filter((l) => l.hasKvCache).length
  // Multi-head latent attention (DeepSeek-V3, R1 distills, Kimi) stores a
  // compressed latent of `kv_lora_rank` plus the decoupled RoPE dimensions
  // per token per layer — not one head of keys and values per KV head.
  // Charging `head_count_kv * (key_length + value_length)` overstates the
  // cache ~70x (40,960 elements vs 576 on DeepSeek-V3) and declares fittable
  // hardware unfittable.
  const kvLoraRank = num(kv, `${arch}.attention.kv_lora_rank`)
  const kvElementsPerLayerToken =
    kvLoraRank !== undefined && kvLoraRank > 0
      ? kvLoraRank + (num(kv, `${arch}.rope.dimension_count`) ?? 0)
      : kvHeads * (keyLength + valueLength)

  const convKernel = num(kv, `${arch}.ssm.conv_kernel`) ?? 0
  const stateSize = num(kv, `${arch}.ssm.state_size`) ?? 0
  const innerSize = num(kv, `${arch}.ssm.inner_size`) ?? 0
  const groupCount = num(kv, `${arch}.ssm.group_count`) ?? 0
  const recurrentLayerCount = ssmLayers.size
  const recurrentStateBytesPerSeq =
    recurrentLayerCount > 0
      ? ((convKernel - 1) * (innerSize + 2 * groupCount * stateSize) +
          stateSize * innerSize) *
        4
      : 0

  return {
    arch,
    layerCount,
    embeddingLength: embedding,
    trainedContext: num(kv, `${arch}.context_length`) ?? 0,
    kvElementsPerToken: kvLayerCount * kvElementsPerLayerToken,
    kvElementsPerLayerToken,
    layers,
    nonLayerBytes,
    outputBytes,
    isMoe,
    expertActiveRatio: isMoe ? expertUsed / expertCount : 1,
    recurrentStateBytesPerSeq,
    recurrentLayerCount,
  }
}

/**
 * Draft KV elements per token for an MTP draft header, sized with the same
 * shape math as the main model. Zero without a header: the draft weights
 * alone are still charged, and the second context goes uncounted rather
 * than blocking the plan.
 */
export function draftKvElementsPerToken(
  header: GgufMetadata | undefined
): number {
  return header ? readModelShape(header).kvElementsPerToken : 0
}

/**
 * Bytes per cache element for each `--cache-type-k/v` value. The quantized
 * types are block formats, so the figure is fractional: Q8_0 packs 32 values
 * into 34 bytes.
 */
export const KV_CACHE_TYPE_BYTES: Record<string, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  q8_0: 34 / 32,
  q5_1: 24 / 32,
  q5_0: 22 / 32,
  q4_1: 20 / 32,
  q4_0: 18 / 32,
  iq4_nl: 18 / 32,
}

export function kvCacheBytes(
  shape: ModelShape,
  contextLength: number,
  cacheType: string
): number {
  const perElement = KV_CACHE_TYPE_BYTES[cacheType] ?? KV_CACHE_TYPE_BYTES.f16
  return shape.kvElementsPerToken * contextLength * perElement
}

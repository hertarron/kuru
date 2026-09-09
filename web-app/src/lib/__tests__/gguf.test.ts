import { describe, expect, it } from 'vitest'
import type { GgufMetadata } from '@janhq/tauri-plugin-llamacpp-api'
import { readModelShape, draftKvElementsPerToken } from '../gguf'

/**
 * A hybrid model in the shape of Qwen3.6: one attention layer every fourth
 * block, the rest linear-attention blocks carrying `ssm_` state.
 */
function hybridHeader(interval: number, blocks: number): GgufMetadata {
  const tensors = []
  for (let i = 0; i < blocks; i++) {
    const attention = (i + 1) % interval === 0
    tensors.push({ name: `blk.${i}.attn_norm.weight`, dims: [2048], ggml_type: 0 })
    tensors.push(
      attention
        ? { name: `blk.${i}.attn_k.weight`, dims: [2048, 512], ggml_type: 0 }
        : { name: `blk.${i}.ssm_conv1d.weight`, dims: [4, 4096], ggml_type: 0 }
    )
  }
  return {
    version: 3,
    tensor_count: tensors.length,
    metadata: {
      'general.architecture': 'hybrid',
      'hybrid.block_count': String(blocks),
      'hybrid.embedding_length': '2048',
      'hybrid.attention.head_count': '16',
      'hybrid.attention.head_count_kv': '2',
      'hybrid.attention.key_length': '256',
      'hybrid.attention.value_length': '256',
      'hybrid.context_length': '262144',
    },
    tensors,
  }
}

describe('readModelShape', () => {
  it('charges KV cache only to layers that hold one', () => {
    // Charging every block would overstate the cache by the interleave ratio,
    // which cost a Qwen3.6 model three quarters of its context.
    const shape = readModelShape(hybridHeader(4, 40))

    expect(shape.layers.filter((l) => l.hasKvCache)).toHaveLength(10)
    expect(shape.kvElementsPerToken).toBe(10 * 2 * (256 + 256))
  })

  it('treats a plain attention model as having a cache in every layer', () => {
    const shape = readModelShape(hybridHeader(1, 40))

    expect(shape.kvElementsPerToken).toBe(40 * 2 * (256 + 256))
  })

  it('sizes an MLA cache from the latent rank plus rope dims', () => {
    // DeepSeek-V3 geometry: 128 KV heads of 192+128 would charge 40,960
    // elements per token per layer; the stored latent is 512 + 64.
    const header = hybridHeader(1, 3)
    header.metadata['general.architecture'] = 'deepseek'
    header.metadata['deepseek.block_count'] = '3'
    header.metadata['deepseek.embedding_length'] = '2048'
    header.metadata['deepseek.attention.head_count'] = '16'
    header.metadata['deepseek.attention.head_count_kv'] = '128'
    header.metadata['deepseek.attention.key_length'] = '192'
    header.metadata['deepseek.attention.value_length'] = '128'
    header.metadata['deepseek.attention.kv_lora_rank'] = '512'
    header.metadata['deepseek.rope.dimension_count'] = '64'
    const shape = readModelShape(header)

    expect(shape.kvElementsPerLayerToken).toBe(512 + 64)
    expect(shape.kvElementsPerToken).toBe(3 * (512 + 64))
  })

  it('marks one layer in six global on Gemma-3, from the llama.cpp default', () => {
    const header = hybridHeader(1, 12)
    header.metadata['general.architecture'] = 'gemma3'
    header.metadata['gemma3.block_count'] = '12'
    header.metadata['gemma3.embedding_length'] = '2048'
    header.metadata['gemma3.attention.head_count'] = '16'
    header.metadata['gemma3.attention.head_count_kv'] = '2'
    header.metadata['gemma3.attention.key_length'] = '256'
    header.metadata['gemma3.attention.value_length'] = '256'
    header.metadata['gemma3.attention.sliding_window'] = '512'
    const shape = readModelShape(header)

    // Period 6: global exactly where il % 6 === 5.
    expect(shape.layers.map((l) => l.swaWindow ?? 0)).toEqual([
      512, 512, 512, 512, 512, 0, 512, 512, 512, 512, 512, 0,
    ])
    // Window layers still hold a cache; the per-token figure is unchanged.
    expect(shape.kvElementsPerToken).toBe(12 * 2 * (256 + 256))
  })

  it('prefers an explicit sliding_window_pattern from the GGUF', () => {
    const header = hybridHeader(1, 4)
    header.metadata['general.architecture'] = 'gemma3'
    header.metadata['gemma3.block_count'] = '4'
    header.metadata['gemma3.embedding_length'] = '2048'
    header.metadata['gemma3.attention.head_count'] = '16'
    header.metadata['gemma3.attention.head_count_kv'] = '2'
    header.metadata['gemma3.attention.key_length'] = '256'
    header.metadata['gemma3.attention.value_length'] = '256'
    header.metadata['gemma3.attention.sliding_window'] = '512'
    header.metadata['gemma3.attention.sliding_window_pattern'] = '2'
    const shape = readModelShape(header)

    expect(shape.layers.map((l) => l.swaWindow ?? 0)).toEqual([512, 0, 512, 0])
  })

  it('charges no KV cache to an embedded MTP block', () => {
    // Qwen3.8-27B: block_count includes the nextn block, which carries
    // prediction tensors but no cache of its own.
    const header = hybridHeader(1, 3)
    header.tensors.push({
      name: 'blk.2.nextn.eh_proj.weight',
      dims: [1024, 1024],
      ggml_type: 0,
    })
    const shape = readModelShape(header)

    expect(shape.layers[2].hasKvCache).toBe(false)
    expect(shape.layers.filter((l) => l.hasKvCache)).toHaveLength(2)
    expect(shape.kvElementsPerToken).toBe(
      2 * shape.kvElementsPerLayerToken
    )
  })

  it('sizes a draft header with the same shape math, or zero without one', () => {
    const header = hybridHeader(1, 4)

    expect(draftKvElementsPerToken(header)).toBe(4 * 2 * (256 + 256))
    expect(draftKvElementsPerToken(undefined)).toBe(0)
  })
})

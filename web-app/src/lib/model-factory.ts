/**
 * Model Factory
 *
 * This factory provides a unified interface for creating language models from various providers.
 * It handles the complexity of initializing different AI SDK providers with their specific
 * configurations and returns a standard LanguageModel interface.
 *
 * Supported Providers:
 * - llamacpp: Local models via llama.cpp (requires running session)
 * - mlx: Local models via MLX-Swift on Apple Silicon (requires running session)
 * - anthropic: Claude models via Anthropic API (@ai-sdk/anthropic v2.0)
 * - google/gemini: Gemini models via Google Generative AI API (@ai-sdk/google v2.0)
 * - openai: OpenAI models via OpenAI API (@ai-sdk/openai)
 * - OpenAI-compatible: Azure, Groq, Together, Fireworks, DeepSeek, Mistral, Cohere, etc.
 *
 * Usage:
 * ```typescript
 * const model = await ModelFactory.createModel(modelId, provider, parameters)
 * ```
 *
 * The factory automatically:
 * - Handles provider-specific authentication and headers
 * - Manages llamacpp session discovery and connection
 * - Configures custom headers for each provider
 * - Returns a unified LanguageModel interface compatible with Vercel AI SDK
 */

/**
 * Inference parameters for customizing model behavior
 */
export interface ModelParameters {
  temperature?: number
  top_k?: number
  top_p?: number
  repeat_penalty?: number
  max_output_tokens?: number
  max_context_tokens?: number
  auto_compact?: boolean
  presence_penalty?: number
  frequency_penalty?: number
  /** Newline-separated in the UI; sent as the wire `stop` array. */
  stop?: string
}

import {
  extractReasoningMiddleware,
  wrapLanguageModel,
  type LanguageModel,
} from 'ai'
import { createOpenAI } from '@ai-sdk/openai'
import {
  createOpenAICompatible,
  MetadataExtractor,
  OpenAICompatibleChatLanguageModel,
} from '@ai-sdk/openai-compatible'
import { createAnthropic } from '@ai-sdk/anthropic'
import { createXai } from '@ai-sdk/xai'
import { createMistral } from '@ai-sdk/mistral'
import { createGoogleGenerativeAI } from '@ai-sdk/google'
import { invoke } from '@tauri-apps/api/core'
import { SessionInfo } from '@janhq/core'
import { fetch as httpFetch } from '@tauri-apps/plugin-http'
import { hasAudioSentinel, splitAudioSentinels } from './audio-sentinel'
import { hasVideoSentinel, splitVideoSentinels } from './video-sentinel'
import { filterDefaultSseEvents } from './sseEventTypeFilter'
import { isPlatformTauri } from '@/lib/platform/utils'
import { providerRemoteApiKeyChain } from '@/lib/provider-api-keys'
import {
  LLAMACPP_ONLY_PARAM_KEYS,
  paramsSettings,
} from '@/lib/predefinedParams'
import {
  resolveProviderCaps,
  isModelLevelRejected,
  getMutualExclusionDrops,
  getProviderApiType,
} from '@/lib/providerCaps'
import { describeEngineError } from '@/lib/engineError'
import { i18n } from '@/i18n/react-i18next-compat'
import { useAppState } from '@/hooks/useAppState'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import { ensureAnthropicHeaders } from '@/lib/remoteModelCatalog'
import {
  autofitPlacement,
  effectiveBackend,
  effectiveCacheType,
  effectiveFlashAttn,
  effectiveParallel,
  effectiveRopeFactor,
  effectiveUbatch,
  machineMemoryFrom,
} from './contextPlanner'
import { readModelShape, draftKvElementsPerToken, type ModelShape } from './gguf'
import type { GgufMetadata } from '@janhq/tauri-plugin-llamacpp-api'
import type { DeviceList, SystemUsage } from '@/services/hardware/types'

/**
 * Llama.cpp timings structure from the response
 */
interface LlamaCppTimings {
  prompt_n?: number
  predicted_n?: number
  predicted_per_second?: number
  prompt_per_second?: number
  cache_n?: number
}

// prompt_n only counts tokens freshly processed this turn; tokens served from
// the KV cache (the rest of the conversation) are reported separately in
// cache_n, so total prompt/context usage is the sum of both.
const totalPromptTokens = (timings: LlamaCppTimings): number =>
  (timings.prompt_n ?? 0) + (timings.cache_n ?? 0)

interface LlamaCppPromptProgress {
  total?: number
  cache?: number
  processed?: number
  time_ms?: number
}

interface LlamaCppChunk {
  timings?: LlamaCppTimings
  prompt_progress?: LlamaCppPromptProgress
}

/**
 * Custom metadata extractor for MLX that extracts timing information
 * and converts it to token usage format. MLX uses the same timing structure
 * as llama.cpp.
 */
const providerMetadataExtractor: MetadataExtractor = {
  extractMetadata: async ({ parsedBody }: { parsedBody: unknown }) => {
    const body = parsedBody as LlamaCppChunk
    if (body?.timings) {
      return {
        providerMetadata: {
          promptTokens: totalPromptTokens(body.timings),
          completionTokens: body.timings.predicted_n ?? null,
          tokensPerSecond: body.timings.predicted_per_second ?? null,
          promptPerSecond: body.timings.prompt_per_second ?? null,
        },
      }
    }
    return undefined
  },
  createStreamExtractor: () => {
    let lastTimings: LlamaCppTimings | undefined

    return {
      processChunk: (parsedChunk: unknown) => {
        const chunk = parsedChunk as LlamaCppChunk
        const state = useAppState.getState()
        const streamThreadId = state.currentStreamThreadId
        if (state.loadingModel) {
          state.updateLoadingModel(false)
        }
        if (streamThreadId) {
          state.updateThreadLoadingModel(streamThreadId, false)
        }
        if (chunk?.timings) {
          lastTimings = chunk.timings
          const liveStats = {
            promptTokens: totalPromptTokens(lastTimings),
            completionTokens: lastTimings.predicted_n ?? 0,
            tokensPerSecond: lastTimings.predicted_per_second ?? null,
            promptPerSecond: lastTimings.prompt_per_second ?? null,
          }
          state.updateLiveTokenStats(liveStats)
          if (streamThreadId) {
            state.updateThreadLiveTokenStats(streamThreadId, liveStats)
          }
        }
        const pp = chunk?.prompt_progress
        if (
          pp &&
          typeof pp.total === 'number' &&
          typeof pp.processed === 'number'
        ) {
          const progress = {
            total: pp.total,
            processed: pp.processed,
            cache: pp.cache ?? 0,
            time_ms: pp.time_ms ?? 0,
          }
          state.updatePromptProgress(progress)
          if (streamThreadId) {
            state.updateThreadPromptProgress(streamThreadId, progress)
          }
        }
      },
      buildMetadata: () => {
        if (lastTimings) {
          return {
            providerMetadata: {
              promptTokens: totalPromptTokens(lastTimings),
              completionTokens: lastTimings.predicted_n ?? null,
              tokensPerSecond: lastTimings.predicted_per_second ?? null,
              promptPerSecond: lastTimings.prompt_per_second ?? null,
            },
          }
        }
        return undefined
      },
    }
  },
}

/**
 * Keys from inference parameters that are client-side only and must not
 * be forwarded in the HTTP body to remote APIs.
 */
const CLIENT_SIDE_PARAM_KEYS: ReadonlySet<string> = new Set([
  'ctx_len',
  'max_context_tokens',
  'auto_compact',
])

function filterParameters(
  parameters: Record<string, unknown>,
  keepLlamacppOnly: boolean
): Record<string, unknown> {
  if (keepLlamacppOnly) return parameters
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(parameters)) {
    if (LLAMACPP_ONLY_PARAM_KEYS.has(k)) continue
    out[k] = v
  }
  return out
}

/**
 * Strip sampler params the active provider doesn't accept. Built-in providers
 * with strict capability tables (e.g. real OpenAI rejecting `top_k`) drop the
 * unsupported keys here so the wire request never carries them. Custom/permissive
 * providers keep everything — the user opted into an unknown OAI-compat endpoint.
 *
 * Unknown keys (not in paramsSettings) pass through untouched so non-sampler
 * fields like reasoning controls aren't affected.
 */
function stripUnsupportedSamplers(
  parameters: Record<string, unknown>,
  provider: ProviderObject,
  modelId: string
): Record<string, unknown> {
  const caps = resolveProviderCaps(provider)
  const exclusionDrops = getMutualExclusionDrops(
    parameters,
    provider.provider,
    getProviderApiType(provider)
  )
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(parameters)) {
    if (exclusionDrops.has(k)) continue
    if (isModelLevelRejected(k, provider.provider, modelId)) continue
    const def = paramsSettings[k]
    if (!def) {
      out[k] = v
      continue
    }
    if (def.capability === 'client_only') {
      out[k] = v
      continue
    }
    if (def.capability === 'core') {
      out[k] = v
      continue
    }
    if (caps.supported.has(def.capability) || caps.maybe.has(def.capability)) {
      out[k] = v
    }
  }
  return out
}

/**
 * Strip Jinja-stack-trace noise and other diagnostic prelude from upstream
 * error messages so the UI shows the human-readable cause. Examples:
 *
 *   "\n------------\nWhile executing CallExpression at line 1, column 226 in
 *    source:\n...op.index0 % 2 == 0) %}{{ raise_exception('Conversation roles
 *    must alternate user...\n      ^\nError: Jinja Exception: Conversation
 *    roles must alternate user/assistant/user/assistant/..."
 *
 * becomes:
 *
 *   "Jinja Exception: Conversation roles must alternate user/assistant/..."
 *
 * Pure / no I/O so it can be exercised from tests.
 */
/**
 * Heuristic: does this upstream error look like "this parameter is not
 * accepted"? Catches OpenAI's "Unsupported parameter", Anthropic's mutual
 * exclusion, Mistral/Cohere "unknown/disallowed field" shapes, etc.
 *
 * Used to gate a one-shot retry with all our injected sampling params
 * stripped — cheap recovery without parsing per-provider error grammars.
 */
export function isSamplingParamRejection(message: string): boolean {
  if (typeof message !== 'string') return false
  return (
    /unsupported parameter/i.test(message) ||
    /is not supported with this model/i.test(message) ||
    /cannot both be specified|use only one/i.test(message) ||
    /unknown field/i.test(message) ||
    /property\s+['"`][^'"`]+['"`]\s+is unsupported/i.test(message) ||
    /field\s+['"`][^'"`]+['"`]\s+is not allowed/i.test(message) ||
    /unrecognized arguments?/i.test(message)
  )
}

export function cleanUpstreamErrorMessage(raw: string): string {
  if (typeof raw !== 'string') return raw
  let msg = raw.replace(/\r\n/g, '\n').trim()

  // Prefer the final `Error: <…>` line if present — llama.cpp's Jinja runtime
  // emits "Error: Jinja Exception: <root cause>" after a multi-line trace.
  const errorLineMatch = msg.match(/(?:^|\n)\s*Error:\s*(.+?)(?:\n|$)/)
  if (errorLineMatch && errorLineMatch[1]) {
    return errorLineMatch[1].trim()
  }

  // Otherwise drop a leading `------------` rule and "While executing …"
  // prelude that adds noise without explaining the failure.
  msg = msg.replace(/^-{3,}\s*\n?/, '').trim()
  msg = msg.replace(/^While executing[\s\S]*?\n\s*\^\s*\n?/, '').trim()
  return msg
}

/**
 * Map a transport-level fetch failure (no HTTP response — DNS, connect, TLS,
 * timeout, dropped connection) to an actionable message. `@tauri-apps/plugin-http`
 * rethrows reqwest's raw "error sending request for url …" string, which isn't
 * useful to users. Returns null when `err` is not a recognised transport error.
 */
export function describeTransportError(err: unknown): string | null {
  const raw = err instanceof Error ? err.message : String(err)
  if (typeof raw !== 'string' || !raw) return null
  const m = raw.toLowerCase()

  const isTransport =
    /error sending request/.test(m) ||
    /error trying to connect/.test(m) ||
    /failed to fetch|networkerror|load failed/.test(m) ||
    /connection (refused|reset|closed|aborted)/.test(m) ||
    /connection error|broken pipe/.test(m) ||
    /dns error|failed to lookup|name resolution/.test(m) ||
    /(operation|request|connection) timed out|timeout/.test(m) ||
    /tls|certificate|ssl|handshake/.test(m) ||
    /unreachable/.test(m)
  if (!isTransport) return null

  if (/dns error|failed to lookup|name resolution|unreachable/.test(m)) {
    return "Couldn't reach the provider — the address could not be resolved. Check the provider's Base URL and your internet connection."
  }
  if (/timed out|timeout/.test(m)) {
    return 'The provider took too long to respond and the request timed out. It may be overloaded or slow to start — try again.'
  }
  if (/tls|certificate|ssl|handshake/.test(m)) {
    return "Couldn't establish a secure connection to the provider (TLS/certificate error). Verify the endpoint URL."
  }
  return "Couldn't reach the provider — the connection failed. Check the provider's Base URL and your internet connection, then try again."
}

function requestUrlOf(input: RequestInfo | URL): string {
  if (typeof input === 'string') return input
  if (input instanceof URL) return input.toString()
  return (input as Request).url ?? ''
}

/**
 * Create a custom fetch function that injects additional parameters into the
 * request body, normalising key names for OpenAI-compatible APIs:
 * - `max_output_tokens` is remapped to `max_tokens`
 * - client-side-only keys (e.g. `ctx_len`) are stripped
 *
 * Also rewrites OpenAI-shape error bodies on non-OK responses so the inner
 * `error.message` is cleaned of stack-trace noise before the AI SDK surfaces
 * it to the UI.
 */
export function createCustomFetch(
  baseFetch: typeof globalThis.fetch,
  parameters: Record<string, unknown>,
  keepLlamacppOnly = false,
  onLlamacppServerError?: () => void,
  filterNamedSseEvents = false
): typeof globalThis.fetch {
  // Params entered via a text input arrive as strings (see InputControl).
  // Providers reject a string where a number is required, so coerce back to a
  // number when the param's predefined default is numeric.
  const coerceNumericParam = (key: string, value: unknown): unknown => {
    if (typeof value !== 'string') return value
    const def = paramsSettings[key]
    if (!def || typeof def.value !== 'number' || value.trim() === '') return value
    const n = Number(value)
    return Number.isNaN(n) ? value : n
  }

  // Internal param keys that don't match the llama-server wire field name.
  const WIRE_KEY_REMAP: Record<string, string> = {
    max_output_tokens: 'max_tokens',
    dynatemp_exp: 'dynatemp_exponent',
  }

  // Server expects an array of sampler names; the UI stores a comma/
  // semicolon-separated string for easy editing.
  // Stop strings are newline-separated in the UI because a stop string may
  // itself contain a comma. Empty yields undefined, so an untouched field
  // sends nothing rather than an empty array some servers reject.
  const coerceStop = (value: unknown): unknown => {
    if (typeof value !== 'string') return value
    const stops = value
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean)
    return stops.length > 0 ? stops : undefined
  }

  const coerceSamplers = (value: unknown): unknown => {
    if (typeof value !== 'string') return value
    const names = value
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter(Boolean)
    return names.length > 0 ? names : undefined
  }

  const buildBody = (
    rawBody: Record<string, unknown>,
    includeOurParams: boolean
  ): Record<string, unknown> => {
    if (!includeOurParams) {
      decodeAudioSentinelsInBody(rawBody)
      decodeVideoSentinelsInBody(rawBody)
      return rawBody
    }
    const normalised: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(parameters)) {
      if (CLIENT_SIDE_PARAM_KEYS.has(key)) continue
      if (!keepLlamacppOnly && LLAMACPP_ONLY_PARAM_KEYS.has(key)) continue
      const targetKey = WIRE_KEY_REMAP[key] ?? key
      const coerced =
        key === 'samplers'
          ? coerceSamplers(value)
          : key === 'stop'
            ? coerceStop(value)
            : coerceNumericParam(key, value)
      if (coerced === undefined) continue
      normalised[targetKey] = coerced
    }
    const merged = { ...rawBody, ...normalised }
    if (keepLlamacppOnly) {
      // Assert the server default explicitly so a preset/CLI override can't
      // silently disable prompt-prefix KV reuse across turns.
      merged.cache_prompt = true
    }
    if (keepLlamacppOnly && merged.stream === true) {
      merged.return_progress = true
      // Requests per-chunk timings so the token counter can update live
      // during generation instead of only once the stream finishes.
      merged.timings_per_token = true
    }
    // llama-server convention: max_tokens = -1 means "unlimited". Users who
    // set max_output_tokens = 0 in assistant params mean "no cap", not
    // "produce zero tokens" — coerce here, gated to llamacpp only because
    // OpenAI/Anthropic reject negative values.
    if (keepLlamacppOnly && merged.max_tokens === 0) {
      merged.max_tokens = -1
    }
    decodeAudioSentinelsInBody(merged)
    decodeVideoSentinelsInBody(merged)
    return merged
  }

  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    let rawBody: Record<string, unknown> | null = null
    if (init?.method === 'POST' || !init?.method) {
      try {
        rawBody = init?.body ? JSON.parse(init.body as string) : {}
      } catch (e) {
        throw new Error(
          `Failed to parse request body as JSON: ${e instanceof Error ? e.message : String(e)}`
        )
      }
      init = { ...init, body: JSON.stringify(buildBody(rawBody!, true)) }
    }

    let res: Response
    try {
      res = await baseFetch(input, init)
    } catch (err) {
      const friendly = describeTransportError(err)
      if (!friendly) throw err
      throw new Error(`${friendly} (${requestUrlOf(input)})`)
    }
    if (res.ok) {
      // OpenAI-compatible servers may interleave custom named SSE events (e.g.
      // tool-progress) with chat.completion.chunk data; the AI SDK validates
      // every data line against the chunk schema, so strip non-default events.
      // Opt-in only: Anthropic and the OpenAI Responses API use named SSE
      // events as their protocol, so filtering there blanks the whole stream.
      const contentType = res.headers.get('content-type') || ''
      if (
        filterNamedSseEvents &&
        res.body &&
        contentType.includes('text/event-stream')
      ) {
        return new Response(filterDefaultSseEvents(res.body), {
          status: res.status,
          statusText: res.statusText,
          headers: res.headers,
        })
      }
      return res
    }

    const isLlamacpp500 = keepLlamacppOnly && res.status === 500

    let parsed: { error?: { message?: unknown; [k: string]: unknown } } | null =
      null
    const contentType = res.headers.get('content-type') || ''
    if (contentType.includes('json')) {
      try {
        parsed = JSON.parse(await res.clone().text())
      } catch {
        parsed = null
      }
    }

    const innerMessage =
      typeof parsed?.error?.message === 'string'
        ? (parsed.error.message as string)
        : null

    if (isLlamacpp500 && !innerMessage) {
      // 500 with no JSON body = router couldn't reach the child (crash). Recover.
      onLlamacppServerError?.()
      const synthMessage =
        'The model crashed and is being reloaded. Please retry.'
      const nextBody = JSON.stringify({
        error: {
          message: synthMessage,
          code: 500,
          type: 'llamacpp_server_error',
        },
      })
      return new Response(nextBody, {
        status: res.status,
        statusText: res.statusText,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (!innerMessage) return res

    // Sampling-rejection auto-retry: when the upstream complains about an
    // injected parameter (top_k on OpenAI, temp+top_p on Anthropic, etc.),
    // retry once with all our injected params stripped, surface a toast so
    // the user knows their popover knob was ignored this turn, and return
    // the bare-body response. Only attempts when we have a captured body
    // and the request looked like a chat completion / messages POST.
    if (
      rawBody &&
      res.status >= 400 &&
      res.status < 500 &&
      isSamplingParamRejection(innerMessage) &&
      Object.keys(parameters).length > 0
    ) {
      const bareInit = {
        ...init,
        body: JSON.stringify(buildBody(rawBody, false)),
      }
      const retry = await baseFetch(input, bareInit)
      if (retry.ok) {
        notifySamplingStripped(innerMessage)
        return retry
      }
      // Retry also failed — fall through to surface the cleaned original error.
    }

    const cleaned = cleanUpstreamErrorMessage(innerMessage)
    if (cleaned === innerMessage) return res
    const nextBody = JSON.stringify({
      ...parsed,
      error: { ...parsed!.error, message: cleaned },
    })
    return new Response(nextBody, {
      status: res.status,
      statusText: res.statusText,
      headers: res.headers,
    })
  }
}

/**
 * Narrow view of the service hub the send-time fit needs. The real hub is
 * wider; structural typing keeps this usable wherever a hub is at hand.
 */
interface AutofitServiceHub {
  models(): {
    getActiveModels(): Promise<string[]>
    evictForLoad?(id: string): Promise<boolean>
    readModelGguf(id: string): Promise<GgufMetadata | undefined>
    getModelExtraSizes?(id: string): Promise<
      | {
          mmprojBytes: number
          mtp: boolean
          mtpModelPath?: string
          mtpDraftBytes: number
          mtpDraftHeader?: GgufMetadata
        }
      | undefined
    >
    updateModelSettings(
      id: string,
      patch: Record<string, string | number | boolean | null | undefined>
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

function autofitSetting(settings: SettingsBag, key: string): unknown {
  return settings?.[key]?.controller_props?.value
}

/** GGUF shapes for fitting, kept for the life of the session like the plan hook's. */
const autofitShapeCache = new Map<string, Promise<ModelShape | undefined>>()

/** Longest wait for freed VRAM to show up in the reading. */
const VRAM_SETTLE_TIMEOUT_MS = 4000
const VRAM_SETTLE_INTERVAL_MS = 150

const totalGpuUsedMiB = (usage: SystemUsage | null): number =>
  usage ? usage.gpus.reduce((sum, gpu) => sum + (gpu.used_memory || 0), 0) : 0

/**
 * Waits for the VRAM a just-unloaded model held to leave the reading.
 *
 * The unload returns when the router has been told, not when the driver has
 * reclaimed the memory: the model's child process still has to exit. Planning
 * against the reading taken in that window budgets gigabytes that are already
 * gone, and the model loads with layers on the CPU that had a card waiting
 * for them. So the reading is polled until it stops falling.
 *
 * Two consecutive equal readings, because the figure can plateau briefly
 * while a large allocation is released in pieces. Returns the last reading,
 * or null when the poll never produced one.
 */
async function settleGpuUsage(
  getSystemUsage: () => Promise<SystemUsage | null>
): Promise<SystemUsage | null> {
  const deadline = Date.now() + VRAM_SETTLE_TIMEOUT_MS
  let last: SystemUsage | null = null
  let steady = 0
  while (Date.now() < deadline) {
    const usage = await getSystemUsage().catch(() => null)
    if (usage) {
      if (last && totalGpuUsedMiB(usage) >= totalGpuUsedMiB(last)) {
        if (++steady >= 2) return usage
      } else {
        steady = 0
      }
      last = usage
    }
    await new Promise((resolve) =>
      setTimeout(resolve, VRAM_SETTLE_INTERVAL_MS)
    )
  }
  return last
}

const lastFitNoticeAt = new Map<string, number>()
function notifyAutofitShortfall(
  modelId: string,
  shortfall: { requested: number; fits: number }
): void {
  const now = Date.now()
  if (now - (lastFitNoticeAt.get(modelId) ?? 0) < 5 * 60 * 1000) return
  lastFitNoticeAt.set(modelId, now)
  void import('sonner')
    .then(({ toast }) => {
      toast.warning('Context fitted to available memory', {
        description: `Fits ${shortfall.fits.toLocaleString()} of the requested ${shortfall.requested.toLocaleString()} tokens right now. Free memory or lower context for the full window.`,
      })
    })
    .catch(() => {
      console.warn('[autofit]', modelId, shortfall)
    })
}

/**
 * Send-time autofit: refits the stored context request against a fresh
 * machine snapshot before the router preset is built, so a load never runs
 * on a stale sheet-time placement.
 *
 * Only when the model is not already loaded — a sticky session is never
 * refit under itself — and never for hand-tuned (`plan_manual`) models.
 * Best effort throughout: anything missing or failing leaves the stored
 * settings untouched and the normal load proceeds. In particular the request
 * itself (`ctx_len`) is never rewritten, except to record a default when
 * none usable was stored; a request that exceeds every band loads with the
 * closest placement and a notice instead of being clamped.
 */
async function autofitPlacementForLoad(
  modelId: string,
  serviceHub: AutofitServiceHub
): Promise<void> {
  try {
    const models = serviceHub.models()
    if (!models || typeof models.getActiveModels !== 'function') return
    const activeModels = await models
      .getActiveModels()
      .catch(() => [] as string[])
    if (activeModels.includes(modelId)) return

    const { useModelProvider } = await import('@/hooks/useModelProvider')
    const provider = useModelProvider.getState().getProviderByName('llamacpp')
    const model = provider?.models?.find((m) => m.id === modelId) as
      | { embedding?: boolean; settings?: SettingsBag }
      | undefined
    if (!model || model.embedding === true) return
    if (autofitSetting(model.settings, 'plan_manual') === true) return

    const devices = (autofitSetting(model.settings, 'device') as string) || ''
    const storedCtxLen = Number(autofitSetting(model.settings, 'ctx_len')) || 0
    // Absent means nobody has pinned a context: `ctx_len` is seeded at 8192
    // on import, so the stored number alone never said whether it was chosen.
    const ctxAuto = autofitSetting(model.settings, 'ctx_auto') !== false
    // Recurrent state and the probe floor both scale with the effective
    // parallel slots, flash-attn, backend-build and MTP state (per-model
    // override, else global, else auto) — the same axes the calibration key
    // is built on.
    const globalSettings = (
      provider as {
        settings?: Array<{
          key?: string
          controller_props?: { value?: unknown }
        }>
      }
    )?.settings
    const globalValue = (key: string) =>
      globalSettings?.find((s) => s?.key === key)?.controller_props?.value
    // Legacy Jan fitting owns placement; Kuru Fit is opt-out, so installs
    // predating the key behave as fitted.
    if (globalValue('kuru_fit') === false) return
    const kvCacheType = effectiveCacheType(
      autofitSetting(model.settings, 'cache_type_k'),
      globalValue('cache_type_k')
    )
    const ubatch = effectiveUbatch(
      autofitSetting(model.settings, 'ubatch_size'),
      globalValue('ubatch_size')
    )
    const sequences = effectiveParallel(
      autofitSetting(model.settings, 'parallel'),
      globalValue('parallel')
    )
    const flashAttn = effectiveFlashAttn(
      autofitSetting(model.settings, 'flash_attn'),
      globalValue('flash_attn')
    )
    const backend = effectiveBackend(
      globalValue('version_backend'),
      globalValue('llamacpp_version'),
      globalValue('llamacpp_backend')
    )
    const ropeFactor = effectiveRopeFactor(
      globalValue('rope_scaling'),
      globalValue('rope_scale'),
      globalValue('rope_freq_scale')
    )
    const swaFull = globalValue('swa_full') === true
    // Per-model only, and it moves the whole cache off the cards.
    const noKvOffload =
      autofitSetting(model.settings, 'no_kv_offload') === true
    const mmprojOffload =
      autofitSetting(model.settings, 'offload_mmproj') !== false

    if (typeof models.readModelGguf !== 'function') return
    let pending = autofitShapeCache.get(modelId)
    if (!pending) {
      pending = models
        .readModelGguf(modelId)
        .then((header) => (header ? readModelShape(header) : undefined))
        .catch(() => undefined)
      autofitShapeCache.set(modelId, pending)
    }
    const cached = await pending
    if (!cached) return
    // Sizes model.yml folds into `size_bytes` that the header does not
    // cover. Missing on older engines; then the model plans as text-only
    // without MTP. Copied, not mutated: the cache above is shared.
    const extra = await models
      .getModelExtraSizes?.(modelId)
      .catch(() => undefined)
    const shape =
      extra && (extra.mmprojBytes || extra.mtpDraftBytes)
        ? { ...cached, mmprojBytes: extra.mmprojBytes ?? 0 }
        : cached
    const mtp = extra?.mtp === true
    const mtpDraftBytes = extra?.mtpDraftBytes ?? 0
    const mtpDraftKvPerToken = draftKvElementsPerToken(extra?.mtpDraftHeader)

    const hardware = serviceHub.hardware?.()
    if (!hardware || typeof hardware.getLlamacppDevices !== 'function') return

    // Run the eviction the load is about to run anyway, then wait for the
    // freed VRAM to leave the reading. Without this the incoming model is
    // planned against a card still holding the model it replaces, which on a
    // single-card machine at the default `models_max = 1` is the whole card:
    // the plan pushes layers onto the CPU and the load then finds the card
    // empty. This is a reorder, not extra work -- the unload happened either
    // way -- so it costs only the settle poll.
    const readUsage = () =>
      typeof hardware.getSystemUsage === 'function'
        ? hardware.getSystemUsage().catch(() => null)
        : Promise.resolve(null)
    const evicted =
      typeof models.evictForLoad === 'function'
        ? await models.evictForLoad(modelId).catch(() => false)
        : false

    // Sequential, not parallel: the device list carries its own free-VRAM
    // figure, which `machineMemoryFrom` falls back to, so it has to be read
    // after the eviction has settled too.
    const usage = evicted ? await settleGpuUsage(readUsage) : await readUsage()
    const devicesList = await hardware
      .getLlamacppDevices()
      .catch(() => undefined)
    if (!devicesList) return

    const { useHardware, gpuMemoryUsage, resolveGpuReserveMiB } = await import(
      '@/hooks/useHardware'
    )
    const { hardwareData, systemUsage: storedUsage, gpuReserveMiB } =
      useHardware.getState()
    if (!hardwareData || !hardwareData.total_memory) return
    const systemUsage = usage ?? storedUsage
    if (usage) useHardware.getState().updateSystemUsage(usage)

    const active = devicesList.filter((d) => d.activated !== false)
    // The model is not resident (checked above), so measured use is safe to
    // count — there is no own footprint inside the reading.
    const usedMiB = systemUsage?.used_memory
    const perDevice = Object.fromEntries(
      active.map((device) => [
        device.id,
        {
          usedMiB: gpuMemoryUsage(systemUsage, hardwareData, device.name)
            ?.used,
          reserveMiB: resolveGpuReserveMiB(
            systemUsage,
            hardwareData,
            device,
            gpuReserveMiB
          ),
        },
      ])
    )
    const ramUsedMiB =
      typeof usedMiB === 'number' && usedMiB > 0 ? usedMiB : undefined
    const machine = machineMemoryFrom(
      active,
      hardwareData,
      perDevice,
      ramUsedMiB
    )

    // This reading was taken with the model unloaded, which is the one thing
    // the context sheet cannot measure for itself once the model is resident.
    const { useLoadBaseline } = await import('@/hooks/useLoadBaseline')
    useLoadBaseline.getState().record(modelId, {
      perDeviceUsedMiB: Object.fromEntries(
        Object.entries(perDevice).flatMap(([id, entry]) =>
          typeof entry.usedMiB === 'number' ? [[id, entry.usedMiB]] : []
        )
      ),
      ramUsedMiB,
    })

    const { useModelCalibration } = await import('@/hooks/useModelCalibration')
    const calibrations =
      useModelCalibration.getState().byModel[modelId]
    const result = autofitPlacement({
      shape,
      machine,
      kvCacheType,
      ubatch,
      sequences,
      flashAttn,
      backend,
      mtp,
      mtpDraftBytes,
      mtpDraftKvPerToken,
      mmprojOffload,
      swaFull,
      ropeFactor,
      noKvOffload,
      devices,
      storedCtxLen,
      ctxAuto,
      calibrations,
      currentValue: (key) => autofitSetting(model.settings, key),
    })
    if (!result || Object.keys(result.patch).length === 0) {
      if (result?.shortfall) notifyAutofitShortfall(modelId, result.shortfall)
      return
    }

    // Both writes are awaited before startModel builds the router preset from
    // model.yml.
    const { applyPlannerPatch } = await import('./plannerPatch')
    await applyPlannerPatch(
      modelId,
      result.patch,
      typeof models.updateModelSettings === 'function'
        ? (id, patch) => models.updateModelSettings(id, patch)
        : undefined
    )
    if (result.shortfall) notifyAutofitShortfall(modelId, result.shortfall)
  } catch {
    // Fail open: the normal load proceeds on stored settings.
  }
}

let lastSamplingNoticeAt = 0
function notifySamplingStripped(reason: string): void {
  const now = Date.now()
  if (now - lastSamplingNoticeAt < 3000) return
  lastSamplingNoticeAt = now
  void import('sonner')
    .then(({ toast }) => {
      toast.warning('Sampling parameters dropped', {
        description: `${cleanUpstreamErrorMessage(reason)} — request retried without your sampling overrides. Adjust in the Sampling popover.`,
      })
    })
    .catch(() => {
      console.warn('[sampling-retry]', reason)
    })
}

/**
 * Drop `reasoning_content` / `reasoning` from assistant turns in the outgoing
 * request body. The Vercel AI SDK's openai-compatible model attaches
 * `reasoning_content` whenever a prior assistant message had a reasoning part
 * (see @ai-sdk/openai-compatible dist/index.js:260). Groq's strict validator
 * rejects this with `property 'reasoning_content' is unsupported`.
 */
export function stripAssistantReasoningInBody(
  body: Record<string, unknown>
): void {
  const messages = body.messages
  if (!Array.isArray(messages)) return
  for (const msg of messages) {
    if (!msg || typeof msg !== 'object') continue
    const m = msg as { role?: string; reasoning_content?: unknown; reasoning?: unknown }
    if (m.role !== 'assistant') continue
    if ('reasoning_content' in m) delete m.reasoning_content
    if ('reasoning' in m) delete m.reasoning
  }
}

/** Reads the global "Strip reasoning from context" toggle; defaults to false. */
function shouldStripReasoningFromContext(): boolean {
  return useGeneralSetting.getState().stripReasoningFromContext === true
}

/**
 * A local llama.cpp model "preserves reasoning" when its chat template accepts
 * a `preserve_thinking` kwarg that is on (user-set value, else the GGUF-detected
 * default). Such templates re-emit prior `<think>` from the resent
 * `reasoning_content`, so stripping that field would shrink earlier assistant
 * turns and force llama.cpp to reprocess the KV-cache prefix. When true, the
 * reasoning must be resent even if the global strip toggle is on.
 */
export function modelPreservesReasoning(
  provider: ProviderObject | undefined,
  modelId: string
): boolean {
  const model = provider?.models?.find((m) => m.id === modelId)
  if (!model) return false
  const raw: unknown =
    model.settings?.chat_template_kwargs?.controller_props?.value
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    const v = (raw as Record<string, unknown>).preserve_thinking
    if (typeof v === 'boolean') return v
  }
  return (
    model.template_kwargs?.find((k) => k.name === 'preserve_thinking')
      ?.default === true
  )
}

/** Wraps `inner` to strip reasoning fields from assistant messages before send. */
function withAssistantReasoningStripped(
  inner: typeof globalThis.fetch
): typeof globalThis.fetch {
  return async (input, init) => {
    if ((init?.method === 'POST' || !init?.method) && init?.body) {
      try {
        const body = JSON.parse(init.body as string)
        stripAssistantReasoningInBody(body)
        init = { ...init, body: JSON.stringify(body) }
      } catch {
        // non-JSON body; fall through
      }
    }
    return inner(input, init)
  }
}

// Rewrites any sentinel-bearing text content (planted by
// CustomChatTransport.encodeAudioAttachments) back into OpenAI `input_audio`
// content parts. Mutates `body.messages` in place.
export function decodeAudioSentinelsInBody(body: Record<string, unknown>): void {
  const messages = body.messages
  if (!Array.isArray(messages)) return
  for (const msg of messages) {
    if (!msg || typeof msg !== 'object') continue
    const m = msg as { role?: string; content?: unknown }
    if (m.role !== 'user') continue
    if (typeof m.content === 'string') {
      if (!hasAudioSentinel(m.content)) continue
      const split = splitAudioSentinels(m.content)
      if (split) m.content = split
      continue
    }
    if (!Array.isArray(m.content)) continue
    const next: unknown[] = []
    let touched = false
    for (const part of m.content) {
      if (
        part &&
        typeof part === 'object' &&
        (part as { type?: string }).type === 'text' &&
        typeof (part as { text?: string }).text === 'string' &&
        hasAudioSentinel((part as { text: string }).text)
      ) {
        const split = splitAudioSentinels((part as { text: string }).text)
        if (split) {
          next.push(...split)
          touched = true
          continue
        }
      }
      next.push(part)
    }
    if (touched) m.content = next
  }
}

// Rewrites any sentinel-bearing text content (planted by
// CustomChatTransport.encodeVideoAttachments) back into llama-server
// `input_video` content parts. Mutates `body.messages` in place. Runs after
// decodeAudioSentinelsInBody, so it also handles content already promoted to
// an array by the audio pass; the two sentinel markers are disjoint.
export function decodeVideoSentinelsInBody(body: Record<string, unknown>): void {
  const messages = body.messages
  if (!Array.isArray(messages)) return
  for (const msg of messages) {
    if (!msg || typeof msg !== 'object') continue
    const m = msg as { role?: string; content?: unknown }
    if (m.role !== 'user') continue
    if (typeof m.content === 'string') {
      if (!hasVideoSentinel(m.content)) continue
      const split = splitVideoSentinels(m.content)
      if (split) m.content = split
      continue
    }
    if (!Array.isArray(m.content)) continue
    const next: unknown[] = []
    let touched = false
    for (const part of m.content) {
      if (
        part &&
        typeof part === 'object' &&
        (part as { type?: string }).type === 'text' &&
        typeof (part as { text?: string }).text === 'string' &&
        hasVideoSentinel((part as { text: string }).text)
      ) {
        const split = splitVideoSentinels((part as { text: string }).text)
        if (split) {
          next.push(...split)
          touched = true
          continue
        }
      }
      next.push(part)
    }
    if (touched) m.content = next
  }
}

type ApiKeyHeaderMode =
  | 'authorization-bearer'
  | 'x-api-key'
  | 'x-goog-api-key'

/** Retries with the next key when the upstream returns 401, 403, or 429. */
function createApiKeyRotatingFetch(
  baseFetch: typeof globalThis.fetch,
  apiKeys: string[],
  parameters: Record<string, unknown>,
  headerMode: ApiKeyHeaderMode,
  filterNamedSseEvents = false
): typeof globalThis.fetch {
  const inner = createCustomFetch(
    baseFetch,
    parameters,
    false,
    undefined,
    filterNamedSseEvents
  )
  if (apiKeys.length <= 1) {
    return inner
  }
  return async (
    input: RequestInfo | URL,
    init?: RequestInit
  ): Promise<Response> => {
    for (let i = 0; i < apiKeys.length; i++) {
      const key = apiKeys[i]!
      const nextHeaders = new Headers(init?.headers as HeadersInit | undefined)
      if (headerMode === 'authorization-bearer') {
        nextHeaders.set('Authorization', `Bearer ${key}`)
      } else if (headerMode === 'x-goog-api-key') {
        nextHeaders.set('x-goog-api-key', key)
      } else {
        nextHeaders.set('x-api-key', key)
      }
      const res = await inner(input, { ...init, headers: nextHeaders })
      if ([401, 403, 429].includes(res.status) && i < apiKeys.length - 1) {
        res.body?.cancel().catch(() => {})
        continue
      }
      return res
    }
    throw new Error('API key rotation exhausted')
  }
}

// An empty apiKey still puts an empty auth header on the wire, which upstreams
// answer with misleading 401s (e.g. Anthropic's "x-api-key header is
// required"). Fail here with an actionable message instead.
function requireRemoteApiKey(
  provider: ProviderObject,
  keyChain: string[]
): string {
  const key = keyChain[0] ?? provider.api_key?.trim()
  if (!key) {
    throw new Error(
      `No API key configured for ${provider.provider}. Add one in Settings > Model Providers.`
    )
  }
  return key
}

function getRuntimeFetch(): typeof globalThis.fetch {
  const maybeWindow = globalThis as typeof globalThis & {
    __TAURI__?: unknown
    __TAURI_INTERNALS__?: unknown
  }
  const hasTauriRuntime =
    typeof maybeWindow.__TAURI__ !== 'undefined' ||
    typeof maybeWindow.__TAURI_INTERNALS__ !== 'undefined'

  return isPlatformTauri() && hasTauriRuntime
    ? (httpFetch as typeof globalThis.fetch)
    : globalThis.fetch
}

/**
 * Map of model keywords to their respective reasoning tags.
 * Used for models that use tags other than the default 'think'.
 */
const REASONING_TAG_MAP: Record<string, string> = {
  gemma: 'thought',
}

/**
 * The default tag used for reasoning extraction if no specific override is found.
 */
const DEFAULT_REASONING_TAG = 'think'

/**
 * Determines the reasoning tag name based on the model ID.
 * Defaults to 'think' if no specific override is found in the map.
 */
function getReasoningTagName(modelId: string): string {
  const lowerId = modelId.toLowerCase()
  for (const [keyword, tag] of Object.entries(REASONING_TAG_MAP)) {
    if (lowerId.includes(keyword)) {
      return tag
    }
  }
  return DEFAULT_REASONING_TAG
}

/**
 * Factory for creating language models based on provider type.
 * Supports native AI SDK providers (Anthropic, Google) and OpenAI-compatible providers.
 */
export class ModelFactory {
  /**
   * Create a language model instance based on the provider configuration
   */
  static async createModel(
    modelId: string,
    provider: ProviderObject,
    parameters: Record<string, unknown> = {}
  ): Promise<LanguageModel> {
    const providerName = provider.provider.toLowerCase()
    parameters = stripUnsupportedSamplers(parameters, provider, modelId)

    // Wire-format dispatch wins over name-based dispatch — custom providers
    // pointing at Anthropic-compatible proxies (LiteLLM, Bedrock gateways)
    // need the Anthropic SDK regardless of the user-chosen provider name.
    if (getProviderApiType(provider) === 'anthropic' && providerName !== 'anthropic') {
      return this.createAnthropicModel(modelId, provider, parameters)
    }

    switch (providerName) {
      case 'llamacpp':
        return this.createLlamaCppModel(modelId, provider, parameters)

      case 'mlx':
        return this.createMlxModel(modelId, provider, parameters)

      case 'anthropic':
        return this.createAnthropicModel(modelId, provider, parameters)

      case 'openai':
        return this.createOpenAIModel(modelId, provider, parameters)
      case 'google':
      case 'gemini':
        return this.createGoogleModel(modelId, provider, parameters)
      case 'azure':
      case 'groq':
      case 'together':
      case 'fireworks':
      case 'deepseek':
      case 'cohere':
      case 'perplexity':
      case 'moonshot':
      case 'minimax':
        return this.createOpenAICompatibleModel(modelId, provider)

      case 'mistral':
        return this.createMistralModel(modelId, provider, parameters)

      case 'xai':
        return this.createXaiModel(modelId, provider, parameters)

      default:
        return this.createOpenAICompatibleModel(modelId, provider, parameters)
    }
  }

  /**
   * Create a llamacpp model by starting the model and finding the running session
   */
  private static async createLlamaCppModel(
    modelId: string,
    provider?: ProviderObject,
    parameters: Record<string, unknown> = {}
  ): Promise<LanguageModel> {
    // Start the model first if provider is available
    if (provider) {
      try {
        const { useServiceStore } = await import('@/hooks/useServiceHub')
        const serviceHub = useServiceStore.getState().serviceHub

        if (serviceHub) {
          // Refit the stored request against a fresh snapshot before the
          // router preset is built. Never throws: failures keep stored
          // settings and the normal load proceeds.
          await autofitPlacementForLoad(modelId, serviceHub).catch(
            () => undefined
          )
          await serviceHub.models().startModel(provider, modelId)
        }
      } catch (error) {
        console.error('Failed to start llamacpp model:', error)
        // A serialized engine error is a plain object, so the previous
        // `instanceof Error` path stringified it into raw JSON for the user.
        throw new Error(
          i18n.t('model-errors:startModelFailed', {
            reason: describeEngineError(error),
          })
        )
      }
    }

    // Get session info which includes port and api_key
    const sessionInfo = await invoke<SessionInfo | null>(
      'plugin:llamacpp|find_session_by_model',
      { modelId }
    )

    if (!sessionInfo) {
      throw new Error(
        i18n.t('model-errors:noRunningSession', { model: modelId })
      )
    }

    const onLlamacppServerError = provider
      ? () => {
          void (async () => {
            try {
              const { useServiceStore } = await import('@/hooks/useServiceHub')
              const hub = useServiceStore.getState().serviceHub
              await hub?.models().reloadModel(provider, modelId)
            } catch (e) {
              console.warn('[llamacpp] reload after crash failed:', e)
            }
          })()
        }
      : undefined
    // The global toggle can strip reasoning_content from resent assistant turns,
    // but never for a preserve_thinking model: its template re-emits prior
    // <think> from that field, so stripping it would diverge the KV-cache prefix.
    let customFetch = createCustomFetch(
      httpFetch,
      parameters,
      true,
      onLlamacppServerError,
      true
    )
    if (
      shouldStripReasoningFromContext() &&
      !modelPreservesReasoning(provider, modelId)
    ) {
      customFetch = withAssistantReasoningStripped(customFetch)
    }

    return new OpenAICompatibleChatLanguageModel(modelId, {
      provider: 'llamacpp',
      headers: () => ({
        Authorization: `Bearer ${sessionInfo.api_key}`,
        Origin: 'tauri://localhost',
      }),
      url: ({ path }) => {
        const url = new URL(`http://localhost:${sessionInfo.port}/v1${path}`)
        return url.toString()
      },
      includeUsage: true,
      fetch: customFetch,
      metadataExtractor: providerMetadataExtractor,
    })
  }

  /**
   * Create an MLX model by starting the model and finding the running session.
   * MLX uses the same OpenAI-compatible API pattern as llamacpp.
   */
  private static async createMlxModel(
    modelId: string,
    provider?: ProviderObject,
    parameters: Record<string, unknown> = {}
  ): Promise<LanguageModel> {
    parameters = filterParameters(parameters, false)
    // Start the model first if provider is available
    if (provider) {
      try {
        const { useServiceStore } = await import('@/hooks/useServiceHub')
        const serviceHub = useServiceStore.getState().serviceHub

        if (serviceHub) {
          await serviceHub.models().startModel(provider, modelId)
        }
      } catch (error) {
        console.error('Failed to start MLX model:', error)
        throw new Error(
          i18n.t('model-errors:startModelFailed', {
            reason: describeEngineError(error),
          })
        )
      }
    }

    // Get session info which includes port and api_key
    const sessionInfo = await invoke<SessionInfo | null>(
      'plugin:mlx|find_mlx_session_by_model',
      { modelId }
    )

    if (!sessionInfo) {
      throw new Error(`No running MLX session found for model: ${modelId}`)
    }

    const baseUrl = `http://localhost:${sessionInfo.port}`
    const authHeaders = {
      Authorization: `Bearer ${sessionInfo.api_key}`,
      Origin: 'tauri://localhost',
    }

    // Share the common fetch (param normalisation + error-body cleaning that
    // rebuilds upstream errors from buffered text rather than re-decoding the
    // raw stream) with every other provider, then layer MLX's /cancel-on-abort
    // on top.
    let baseCustomFetch = createCustomFetch(
      httpFetch,
      parameters,
      false,
      undefined,
      true
    )
    if (shouldStripReasoningFromContext()) {
      baseCustomFetch = withAssistantReasoningStripped(baseCustomFetch)
    }
    const customFetch: typeof globalThis.fetch = async (
      input: RequestInfo | URL,
      init?: RequestInit
    ): Promise<Response> => {
      // When the request is aborted, also call the server's /cancel endpoint
      // to stop MLX inference immediately.
      if (init?.signal) {
        init.signal.addEventListener('abort', () => {
          httpFetch(`${baseUrl}/v1/cancel`, {
            method: 'POST',
            headers: { ...authHeaders, 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
          }).catch(() => {
            // Ignore cancel request errors
          })
        })
      }

      return baseCustomFetch(input, init)
    }

    const model = new OpenAICompatibleChatLanguageModel(modelId, {
      provider: 'mlx',
      headers: () => authHeaders,
      url: ({ path }) => {
        const url = new URL(`${baseUrl}/v1${path}`)
        return url.toString()
      },
      fetch: customFetch,
      metadataExtractor: providerMetadataExtractor,
    })

    return wrapLanguageModel({
      model: model,
      middleware: extractReasoningMiddleware({
        tagName: getReasoningTagName(modelId),
        separator: '\n',
      }),
    })
  }

  /**
   * Create an Anthropic model using the official AI SDK
   */
  private static createAnthropicModel(
    modelId: string,
    provider: ProviderObject,
    parameters: Record<string, unknown> = {}
  ): LanguageModel {
    const headers: Record<string, string> = {}

    if (provider.custom_header) {
      provider.custom_header.forEach((customHeader) => {
        headers[customHeader.header] = customHeader.value
      })
    }
    // Custom Anthropic providers may ship no custom_header; Anthropic rejects
    // browser-context requests (webview Origin) without the opt-in header.
    ensureAnthropicHeaders(provider, headers)

    const keyChain = providerRemoteApiKeyChain(provider)
    const fetchImpl =
      keyChain.length > 1
        ? createApiKeyRotatingFetch(
            getRuntimeFetch(),
            keyChain,
            parameters,
            'x-api-key'
          )
        : createCustomFetch(getRuntimeFetch(), parameters)

    const anthropic = createAnthropic({
      apiKey: requireRemoteApiKey(provider, keyChain),
      baseURL: provider.base_url,
      headers: Object.keys(headers).length > 0 ? headers : undefined,
      fetch: fetchImpl,
    })

    return anthropic(modelId)
  }


  /**
   * Create an OpenAI model using the official AI SDK
   */
  private static createOpenAIModel(
    modelId: string,
    provider: ProviderObject,
    parameters: Record<string, unknown> = {}
  ): LanguageModel {
    const headers: Record<string, string> = {}

    // Add custom headers if specified
    if (provider.custom_header) {
      provider.custom_header.forEach((customHeader) => {
        headers[customHeader.header] = customHeader.value
      })
    }

    const keyChain = providerRemoteApiKeyChain(provider)
    const fetchImpl =
      keyChain.length > 1
        ? createApiKeyRotatingFetch(
            getRuntimeFetch(),
            keyChain,
            parameters,
            'authorization-bearer'
          )
        : createCustomFetch(getRuntimeFetch(), parameters)

    const openai = createOpenAI({
      apiKey: requireRemoteApiKey(provider, keyChain),
      baseURL: provider.base_url,
      headers: Object.keys(headers).length > 0 ? headers : undefined,
      fetch: fetchImpl,
    })

    // The genuine OpenAI provider always supports the Responses API, so use it
    // unconditionally: it is a superset of Chat Completions and the only surface
    // that returns reasoning summaries. Custom OpenAI-compatible providers route
    // through createOpenAICompatibleModel, not here, so this cannot hit a proxy
    // that only implements /chat/completions.
    return openai.responses(modelId)
  }

  /**
   * Create a Mistral model using the official AI SDK. Needed for magistral-*,
   * which streams `delta.content` as an array of typed parts (thinking + text);
   * the generic openai-compatible schema rejects this with a Zod
   * "expected string, received array" error.
   */
  private static createMistralModel(
    modelId: string,
    provider: ProviderObject,
    parameters: Record<string, unknown> = {}
  ): LanguageModel {
    const headers: Record<string, string> = {}
    if (provider.custom_header) {
      provider.custom_header.forEach((customHeader) => {
        headers[customHeader.header] = customHeader.value
      })
    }

    const keyChain = providerRemoteApiKeyChain(provider)
    const fetchImpl =
      keyChain.length > 1
        ? createApiKeyRotatingFetch(
            getRuntimeFetch(),
            keyChain,
            parameters,
            'authorization-bearer'
          )
        : createCustomFetch(getRuntimeFetch(), parameters)

    const mistral = createMistral({
      apiKey: requireRemoteApiKey(provider, keyChain),
      baseURL: provider.base_url,
      headers: Object.keys(headers).length > 0 ? headers : undefined,
      fetch: fetchImpl,
    })

    return mistral(modelId)
  }

  /**
   * Create an XAI (Grok) model using the official AI SDK
   */
  private static createXaiModel(
    modelId: string,
    provider: ProviderObject,
    parameters: Record<string, unknown> = {}
  ): LanguageModel {
    const headers: Record<string, string> = {}

    // Add custom headers if specified
    if (provider.custom_header) {
      provider.custom_header.forEach((customHeader) => {
        headers[customHeader.header] = customHeader.value
      })
    }

    const keyChain = providerRemoteApiKeyChain(provider)
    const fetchImpl =
      keyChain.length > 1
        ? createApiKeyRotatingFetch(
            getRuntimeFetch(),
            keyChain,
            parameters,
            'authorization-bearer'
          )
        : createCustomFetch(getRuntimeFetch(), parameters)

    const xai = createXai({
      apiKey: requireRemoteApiKey(provider, keyChain),
      baseURL: provider.base_url,
      headers: Object.keys(headers).length > 0 ? headers : undefined,
      fetch: fetchImpl,
    })

    return xai(modelId)
  }

  // Native Google provider. Gemini 3 preview models require a
  // `thought_signature` to be round-tripped on tool-call replays, which
  // Google's /v1beta/openai compat layer does not surface — so we go native.
  // Stored base_url points at …/v1beta/openai for the compat path; the native
  // client wants the bare …/v1beta base.
  private static createGoogleModel(
    modelId: string,
    provider: ProviderObject,
    parameters: Record<string, unknown> = {}
  ): LanguageModel {
    const headers: Record<string, string> = {}
    if (provider.custom_header) {
      provider.custom_header.forEach((customHeader) => {
        headers[customHeader.header] = customHeader.value
      })
    }

    const keyChain = providerRemoteApiKeyChain(provider)
    // Rotate over configured keys on 401/403/429 (e.g. exhausted free-tier
    // quota). The native Google client authenticates via `x-goog-api-key`, so
    // the rotating fetch must override that header — not Authorization.
    const fetchImpl =
      keyChain.length > 1
        ? createApiKeyRotatingFetch(
            getRuntimeFetch(),
            keyChain,
            parameters,
            'x-goog-api-key'
          )
        : createCustomFetch(getRuntimeFetch(), parameters)

    const rawBase = provider.base_url?.trim()
    const baseURL = rawBase
      ? rawBase.replace(/\/openai\/?$/, '').replace(/\/$/, '')
      : 'https://generativelanguage.googleapis.com/v1beta'

    const google = createGoogleGenerativeAI({
      apiKey: requireRemoteApiKey(provider, keyChain),
      baseURL,
      headers: Object.keys(headers).length > 0 ? headers : undefined,
      fetch: fetchImpl,
    })

    return google(modelId)
  }

  /**
   * Create an OpenAI-compatible model for providers that support the OpenAI API format
   */
  private static createOpenAICompatibleModel(
    modelId: string,
    provider: ProviderObject,
    parameters: Record<string, unknown> = {}
  ): LanguageModel {
    const headers: Record<string, string> = {}

    // Add custom headers if specified
    if (provider.custom_header) {
      provider.custom_header.forEach((customHeader) => {
        headers[customHeader.header] = customHeader.value
      })
    }

    const keyChain = providerRemoteApiKeyChain(provider)
    if (keyChain.length === 1) {
      headers['Authorization'] = `Bearer ${keyChain[0]}`
    }

    let fetchImpl: typeof globalThis.fetch =
      keyChain.length > 1
        ? createApiKeyRotatingFetch(
            getRuntimeFetch(),
            keyChain,
            parameters,
            'authorization-bearer',
            true
          )
        : createCustomFetch(getRuntimeFetch(), parameters, false, undefined, true)

    // Groq's API rejects assistant `reasoning` fields, so it always strips
    // regardless of the toggle; other providers honor the global setting.
    if (provider.provider === 'groq' || shouldStripReasoningFromContext()) {
      fetchImpl = withAssistantReasoningStripped(fetchImpl)
    }

    const openAICompatible = createOpenAICompatible({
      name: provider.provider,
      baseURL: provider.base_url || 'https://api.openai.com/v1',
      headers,
      includeUsage: true,
      fetch: fetchImpl,
    })

    const model = openAICompatible.languageModel(modelId)

    return wrapLanguageModel({
      model,
      middleware: extractReasoningMiddleware({
        tagName: getReasoningTagName(modelId),
        separator: '\n',
      }),
    })
  }
}

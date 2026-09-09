import { describe, expect, it, vi, beforeEach } from 'vitest'
import type { UIMessage } from '@ai-sdk/react'

// Captures streamText payloads so we can assert what would actually reach
// the model provider for an RP-style thread that opens with a character
// greeting (an assistant turn before any user turn).
const streamTextCalls: Array<Record<string, unknown>> = []

const h = vi.hoisted(() => ({
  disabledTools: [] as string[],
  servers: [] as string[],
  getRelevantTools: vi.fn(),
  serviceHub: null as unknown,
}))

const mcpService = {
  getTools: vi.fn(async () => []),
  getToolsForServers: vi.fn(async () => []),
  getServerSummaries: vi.fn(async () => []),
}

h.serviceHub = {
  mcp: () => mcpService,
  rag: () => ({ getTools: async () => [] }),
}

vi.mock('ai', async () => {
  const actual = await vi.importActual<typeof import('ai')>('ai')
  return {
    ...actual,
    streamText: (args: Record<string, unknown>) => {
      streamTextCalls.push(args)
      return {
        toUIMessageStream: () =>
          new ReadableStream({
            start(controller) {
              controller.close()
            },
          }),
      }
    },
  }
})

const provider = { provider: 'openai', api_key: 'k', models: [] }
const selectedModel = { id: 'gpt', capabilities: ['tools'] }

vi.mock('@/hooks/useServiceHub', () => ({
  useServiceStore: { getState: () => ({ serviceHub: h.serviceHub }) },
}))
vi.mock('@/hooks/useToolAvailable', () => ({
  useToolAvailable: {
    getState: () => ({ getDisabledTools: () => h.disabledTools }),
  },
}))
vi.mock('@/hooks/useModelProvider', () => ({
  useModelProvider: {
    getState: () => ({
      selectedModel,
      selectedProvider: 'openai',
      getProviderByName: () => provider,
    }),
  },
}))
vi.mock('@/hooks/useCharacters', () => ({
  useCharacters: { getState: () => ({ currentCharacter: null }) },
  // Mirrors the real export: the transport resolves a thread's character
  // through it when composing the system prompt.
  resolveThreadCharacter: (thread: { assistants?: { id: string }[] } | undefined, characters: { id: string }[]) => {
    const embedded = thread?.assistants?.[0]
    if (!embedded || embedded.id === 'model-only') return undefined
    return characters.find((c) => c.id === embedded.id) ?? embedded
  },
}))
vi.mock('@/hooks/useThreads', () => ({
  useThreads: { getState: () => ({ threads: {} }) },
}))
vi.mock('@/hooks/useAttachments', () => ({
  useAttachments: { getState: () => ({ enabled: false }) },
}))
vi.mock('@/hooks/useMCPServers', () => ({
  useMCPServers: {
    getState: () => ({ settings: { enableSmartToolRouting: true } }),
  },
}))
vi.mock('@/hooks/useAppState', () => ({
  useAppState: {
    getState: () =>
      new Proxy(
        {},
        {
          get: () => () => undefined,
        }
      ),
  },
}))
vi.mock('@/hooks/useWebSearchConfig', () => ({
  useWebSearchConfig: { getState: () => ({ webSearchEnabled: false }) },
}))
vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn(async () => []) }))
vi.mock('@/lib/extension', () => ({
  ExtensionManager: { getInstance: () => ({ get: () => null }) },
}))
vi.mock('@/lib/llamacppRouterProps', () => ({
  getLlamacppExtension: () => null,
}))
vi.mock('@/lib/mcp-orchestrator', () => ({
  mcpOrchestrator: { getRelevantTools: h.getRelevantTools },
}))
vi.mock('@/lib/mcp-router-model-filter', () => ({
  isRouterModelSelectable: () => false,
}))
vi.mock('@/lib/reasoningProviderOptions', () => ({
  buildReasoningProviderOptions: () => undefined,
}))
vi.mock('@/lib/providerCaps', () => ({
  isPredefinedRemoteProvider: () => false,
  getProviderApiType: () => 'openai',
}))
vi.mock('../model-factory', () => ({
  ModelFactory: { createModel: vi.fn(async () => ({ modelId: 'gpt' })) },
}))

import { CustomChatTransport } from '../custom-chat-transport'

const user = (id: string, text: string): UIMessage =>
  ({ id, role: 'user', parts: [{ type: 'text', text }] }) as UIMessage
const assistant = (id: string, text: string): UIMessage =>
  ({ id, role: 'assistant', parts: [{ type: 'text', text }] }) as UIMessage

async function drain(stream: ReadableStream): Promise<void> {
  const reader = stream.getReader()
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done } = await reader.read()
    if (done) break
  }
}

const send = (transport: CustomChatTransport, messages: UIMessage[]) =>
  transport.sendMessages({
    chatId: 'thread-1',
    messages,
    abortSignal: undefined,
    trigger: 'submit-message',
    messageId: undefined,
  })

/**
 * Mirrors strict Jinja chat templates (llamacpp / Qwen3.5+): at most one
 * leading system turn, then strictly alternating user/assistant starting
 * with user.
 */
function expectTemplateValid(system: unknown, messages: unknown[]) {
  let i = 0
  if (typeof system === 'string' && system.length > 0) {
    expect(messages[0]?.role).not.toBe('system')
  } else if (
    messages.length > 0 &&
    messages[0].role === 'system' &&
    typeof (messages[0] as { content?: unknown }).content === 'string'
  ) {
    i = 1
  }
  let expected: 'user' | 'assistant' = 'user'
  for (; i < messages.length; i++) {
    expect(messages[i].role).toBe(expected)
    expected = expected === 'user' ? 'assistant' : 'user'
  }
}

describe('CustomChatTransport RP greeting payload', () => {
  beforeEach(() => {
    streamTextCalls.length = 0
    h.disabledTools = []
    h.servers = []
    h.getRelevantTools.mockReset()
    h.getRelevantTools.mockResolvedValue([])
  })

  it('folds a leading greeting into the system prompt and keeps roles alternating', async () => {
    const transport = new CustomChatTransport('char definition', 'thread-1')

    await drain(
      await send(transport, [
        assistant('greet', '*You wake up in an open field.*'),
        user('u1', 'I look around.'),
      ])
    )

    expect(streamTextCalls).toHaveLength(1)
    const call = streamTextCalls[0]

    // The character definition AND the greeting both live in the system.
    expect(call.system).toBe(
      'char definition\n\n*You wake up in an open field.*'
    )
    // Messages contain only genuine turns, alternating from user.
    const messages = call.messages as Array<{ role: string }>
    expect(messages.map((m) => m.role)).toEqual(['user'])
    expectTemplateValid(call.system, messages)
  })

  it('promotes the greeting to system when no system prompt exists', async () => {
    const transport = new CustomChatTransport(undefined, 'thread-1')

    await drain(
      await send(transport, [
        assistant('greet', 'Welcome, traveler!'),
        user('u1', 'hello'),
      ])
    )

    const call = streamTextCalls[0]
    expect(call.system).toBe('Welcome, traveler!')
    const messages = call.messages as Array<{ role: string }>
    expect(messages.map((m) => m.role)).toEqual(['user'])
    expectTemplateValid(call.system, messages)
  })

  it('keeps later turns alternating across the folded greeting', async () => {
    const transport = new CustomChatTransport('char definition', 'thread-1')

    await drain(
      await send(transport, [
        assistant('greet', 'Greetings!'),
        user('u1', 'hi'),
      ])
    )
    await drain(
      await send(transport, [
        assistant('greet', 'Greetings!'),
        user('u1', 'hi'),
        assistant('a1', 'Well met.'),
        user('u2', 'more'),
      ])
    )

    expect(streamTextCalls).toHaveLength(2)
    for (const call of streamTextCalls) {
      const messages = call.messages as Array<{ role: string }>
      expectTemplateValid(call.system, messages)
    }
    // Greeting folded into system on both turns; prefix stays byte-stable.
    const [firstCall, secondCall] = streamTextCalls
    expect(secondCall.system).toBe(firstCall.system)
    expect(String(firstCall.system)).toContain('Greetings!')
  })

  it('leaves normal chats without a greeting untouched', async () => {
    const transport = new CustomChatTransport('you are jan', 'thread-1')

    await drain(await send(transport, [user('u1', 'hello')]))

    const call = streamTextCalls[0]
    expect(call.system).toBe('you are jan')
    const messages = call.messages as Array<{ role: string }>
    expect(messages.map((m) => m.role)).toEqual(['user'])
  })
})

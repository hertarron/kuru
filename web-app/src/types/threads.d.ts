/**
 * The content type of the message.
 */
type ContentType = 'text' | 'image_url'

/**
 * The `ContentValue` type defines the shape of a content value object
 * @data_transfer_object
 */
type ContentValue = {
  value: string
  annotations: string[]
}

/**
 * The `ImageContentValue` type defines the shape of a content value object of image type
 * @data_transfer_object
 */
type ImageContentValue = {
  detail?: string
  url?: string
}

type ThreadContent = {
  type: ContentType
  text?: ContentValue
  image_url?: ImageContentValue
  role: ChatCompletionRole
}

type ChatCompletionRole = 'system' | 'assistant' | 'user' | 'tool'

type ThreadModel = {
  id: string
  provider: string
}

type Thread = {
  assistants?: ThreadAssistantInfo[]
  id: string
  title: string
  isFavorite?: boolean

  model?: ThreadModel
  updated: number
  order?: number
  metadata?: {
    project?: {
      id: string
      name: string
      updated_at: number
    }
    /** Per-chat lorebook overrides; see `lib/lorebook-selection.ts`. */
    lorebooks?: {
      extraIds?: string[]
      disabledIds?: string[]
      off?: boolean
    }
    /** Per-chat persona pin; see `lib/persona-selection.ts`. */
    persona?: {
      personaId?: string | null
    }
    [key: string]: unknown
  }
}

type Assistant = {
  avatar?: string
  id: string
  name: string
  created_at: number
  description?: string
  instructions: string
  parameters: Record<string, unknown>
  // tool_steps?: number
}

type CharacterKind = 'roleplay' | 'assistant'

/**
 * A roleplay character. Supersedes Assistant everywhere users can see;
 * stored in the same assistant.json files so legacy data migrates by
 * normalization alone. Extra fields mirror the community character card
 * spec (V2/V3 subset) so imported cards round-trip naturally.
 */
type Character = Assistant & {
  /**
   * Roleplay character or plain coding/task assistant. Drives everything
   * opinionated about RP: an `assistant` gets no greeting seeded, no world
   * info, no persona, and its chatbox hides those pills. Inferred once at
   * migration for characters that predate the field; see `inferCharacterKind`.
   */
  kind?: CharacterKind
  /** Short trait summary from the card's `personality` field. */
  personality?: string
  /** Opening scene / premise from the card's `scenario` field. */
  scenario?: string
  /** The character's opening message for new chats. */
  first_mes?: string
  /** Extra opening messages shown as swipeable versions of the greeting. */
  alternate_greetings?: string[]
  /** Example dialogue in `<START> {{user}}:/{{char}}:` block format. */
  mes_example?: string
  /** Card-provided system prompt; overrides composed prompts when present. */
  system_prompt?: string
  post_history_instructions?: string
  tags?: string[]
  creator_notes?: string
  creator?: string
  character_version?: string
  /**
   * Lorebooks attached to this character, by id. Always references into the
   * shared library — a card's embedded book is imported as its own book and
   * then referenced, never copied in here.
   */
  lorebookIds?: string[]
}

type TokenSpeed = {
  message: string
  tokenSpeed: number
  tokenCount: number
  lastTimestamp: number
}

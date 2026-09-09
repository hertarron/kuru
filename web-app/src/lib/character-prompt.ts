import { renderInstructions } from './instructionTemplate'
import { buildPersonaPrompt, type Persona } from './persona'

/** Which block of the composed definition a part is. */
export type CharacterPromptSlot =
  | 'system-prompt'
  | 'definition'
  | 'personality'
  | 'scenario'
  | 'instructions'
  | 'example-dialogue'
  | 'lorebook'
  | 'persona'

export interface CharacterPromptPart {
  /** Visualizer/snapshot label for this block. */
  label: string
  /** Block text with {{char}}/{{user}} resolved. */
  text: string
  /** Lets callers place world info relative to a block instead of by label. */
  slot: CharacterPromptSlot
}

/**
 * World-info blocks to weave into the composed definition. The at-depth slot
 * isn't here: it lands in the message array, not the system string.
 */
export interface CharacterPromptWorldInfo {
  beforeChar?: string
  afterChar?: string
  emTop?: string
  emBottom?: string
}

/**
 * The character's composed definition as ordered, labeled blocks. Mirrors
 * community card conventions: a card-provided system_prompt (style note)
 * leads, followed by description, personality, scenario and example
 * dialogue; legacy assistant instructions remain first-class content.
 * Splitting into parts lets the context visualizer show what each block
 * contributes while `buildCharacterSystemPrompt` joins them unchanged.
 *
 * `worldInfo` blocks are placed by ST's position semantics: before/after the
 * definition, and around the example dialogue. With no example dialogue to
 * bracket, those two land at the end of the definition rather than being
 * dropped.
 */
export function buildCharacterPromptParts(
  character: Character | undefined,
  opts: {
    userName?: string
    worldInfo?: CharacterPromptWorldInfo
    persona?: Persona
  } = {}
): CharacterPromptPart[] {
  const raw: Array<[CharacterPromptSlot, string, string]> = []
  const push = (slot: CharacterPromptSlot, label: string, text?: string) => {
    const t = text?.trim()
    if (t) raw.push([slot, label, t])
  }

  const wi = opts.worldInfo
  push('lorebook', 'World info (before character)', wi?.beforeChar)

  if (character) {
    push('system-prompt', 'System prompt', character.system_prompt)
    push('definition', 'Definition', character.description)
    if (character.personality?.trim()) {
      push(
        'personality',
        'Personality',
        `Personality: ${character.personality.trim()}`
      )
    }
    if (character.scenario?.trim()) {
      push('scenario', 'Scenario', `Scenario: ${character.scenario.trim()}`)
    }
    push('instructions', 'Instructions', character.instructions)

    // After the character's own sheet: the model reads who it is, then who it
    // is speaking to. Before the after-character world info so a lorebook
    // entry can still qualify the persona it just met.
    push('persona', 'Persona', buildPersonaPrompt(opts.persona))

    push('lorebook', 'World info (after character)', wi?.afterChar)

    if (character.mes_example?.trim()) {
      push('lorebook', 'World info (before examples)', wi?.emTop)
      push(
        'example-dialogue',
        'Example dialogue',
        `Example dialogue:\n${character.mes_example.trim()}`
      )
      push('lorebook', 'World info (after examples)', wi?.emBottom)
    } else {
      // Nothing to bracket, so the example-slot entries still get a place
      // rather than silently vanishing from the prompt.
      push('lorebook', 'World info (before examples)', wi?.emTop)
      push('lorebook', 'World info (after examples)', wi?.emBottom)
    }
  } else {
    push('persona', 'Persona', buildPersonaPrompt(opts.persona))
    push('lorebook', 'World info (after character)', wi?.afterChar)
    push('lorebook', 'World info (before examples)', wi?.emTop)
    push('lorebook', 'World info (after examples)', wi?.emBottom)
  }

  // The persona's name is what `{{user}}` means; `userName` is the legacy
  // fallback for setups that never made a persona.
  const vars = {
    char: character?.name,
    user: opts.persona?.name || opts.userName,
  }
  return raw.map(([slot, label, text]) => ({
    slot,
    label,
    // Lorebook content is rendered by the scanner, so this is a no-op for it.
    text: renderInstructions(text, vars),
  }))
}

/**
 * Compose the system message for a chat from its Character. The exact join
 * of `buildCharacterPromptParts` — legacy assistant instructions stay in the
 * backbone so migrated setups behave exactly as before.
 */
export function buildCharacterSystemPrompt(
  character: Character | undefined,
  opts: {
    userName?: string
    worldInfo?: CharacterPromptWorldInfo
    persona?: Persona
  } = {}
): string | undefined {
  const parts = buildCharacterPromptParts(character, opts)
  if (!parts.length) return undefined
  return parts.map((p) => p.text).join('\n\n')
}

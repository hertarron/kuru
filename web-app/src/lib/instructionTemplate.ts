import { formatDate } from '@/utils/formatDate'

type RenderOptions = {
  /** {{char}} replacement -- the character's name. */
  char?: string
  /** {{user}} replacement -- the user's configured persona name. */
  user?: string
}

/**
 * Render prompt text by replacing supported placeholders.
 * Supported placeholders:
 * - {{current_date}}: Inserts today's date (UTC, long month), e.g., August 16, 2025.
 * - {{char}}: The active character's name (left as-is when unknown).
 * - {{user}}: The user's persona name (left as-is when unset).
 */
export function renderInstructions(
  instructions: string,
  opts?: RenderOptions
): string
export function renderInstructions(
  instructions?: string,
  opts?: RenderOptions
): string | undefined
export function renderInstructions(
  instructions?: string,
  opts?: RenderOptions
): string | undefined {
  if (!instructions) return instructions

  const currentDateStr = formatDate(new Date(), { includeTime: false })

  // Replace macros (allow spaces inside braces).
  let rendered = instructions
  rendered = rendered.replace(/\{\{\s*current_date\s*\}\}/gi, currentDateStr)
  if (opts?.char) {
    rendered = rendered.replace(/\{\{\s*char\s*\}\}/gi, opts.char)
  }
  if (opts?.user) {
    rendered = rendered.replace(/\{\{\s*user\s*\}\}/gi, opts.user)
  }
  return rendered
}

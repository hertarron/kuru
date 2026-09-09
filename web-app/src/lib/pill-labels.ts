/**
 * The forms a pill label can take, shortest first. Cuts fall on word
 * boundaries only: "Arkan - Fant..." names nothing that "Arkan" does not.
 *
 * Index 0 is the empty form, the pill down to its icon.
 */
export function labelForms(text: string | undefined): string[] {
  const full = (text ?? '').trim()
  if (!full) return ['']
  // A name that has spaces breaks on them, so "Jean-Luc Picard" keeps its
  // hyphen. A model id has no spaces, and its own separators are the only
  // word boundaries it has: "Gemma-3-1b-It" -> "Gemma", "Gemma-3", ...
  const boundary = /\s/.test(full) ? /(\s+)/ : /([-_.:/]+)/
  const parts = full.split(boundary)

  const forms = ['']
  let grown = ''
  for (let i = 0; i < parts.length; i += 2) {
    grown += (i > 0 ? parts[i - 1] : '') + parts[i]
    // A form ending in its separator ("Arkan -") reads as unfinished.
    const form = grown.replace(/[\s\-_.:/]+$/, '')
    if (form && form !== forms[forms.length - 1]) forms.push(form)
  }
  return forms
}

export type PillKey = 'model' | 'character' | 'persona' | 'lorebook'

/**
 * The order labels grow in as the row gains room; one entry moves one pill up
 * one form. The model leads because a provider logo covers many models, while
 * a character and a persona each have their own portrait. The lorebook count
 * follows: one digit, the most meaning for the least width.
 */
export const GROWTH_ORDER: PillKey[] = [
  'model',
  'lorebook',
  'character',
  'persona',
  'model',
  'model',
  'lorebook',
  'character',
  'persona',
]

/** Once the staged order is spent, the rest grows here while room is left. */
const OVERFLOW_ORDER: PillKey[] = ['model', 'character', 'persona']

export type PillMetrics = {
  /** Candidate labels, shortest first, from `labelForms`. */
  forms: string[]
  /** Rendered width of each form, same order. */
  formWidths: number[]
  /** Pill width with no label: padding, icon, chevron, cog. */
  fixed: number
  /** Gap between the icon and the label, absent when there is no label. */
  labelGap: number
}

/**
 * Chooses the longest set of forms that fit `available` together. A step that
 * does not fit is skipped rather than ending the walk, so one long character
 * name cannot pin the persona beside it to an icon.
 */
export function fitPillLabels(
  pills: Map<PillKey, PillMetrics>,
  available: number,
  gap: number
): Map<PillKey, string> {
  const level = new Map<PillKey, number>()
  for (const key of pills.keys()) level.set(key, 0)

  const width = () => {
    let total = Math.max(0, pills.size - 1) * gap
    for (const [key, pill] of pills) {
      const at = level.get(key) ?? 0
      total += pill.fixed
      if (at > 0) total += pill.labelGap + pill.formWidths[at]
    }
    return total
  }

  const steps = [...GROWTH_ORDER, ...OVERFLOW_ORDER]
  // Every remaining stage of the longest label, so a model id with many
  // segments can still reach its full form on a wide row.
  const longest = Math.max(0, ...[...pills.values()].map((p) => p.forms.length))
  for (let i = 0; i < longest; i++) steps.push(...OVERFLOW_ORDER)

  for (const key of steps) {
    const pill = pills.get(key)
    if (!pill) continue
    const at = level.get(key) ?? 0
    if (at >= pill.forms.length - 1) continue
    level.set(key, at + 1)
    if (width() > available) level.set(key, at)
  }

  const chosen = new Map<PillKey, string>()
  for (const [key, pill] of pills) chosen.set(key, pill.forms[level.get(key) ?? 0])
  return chosen
}

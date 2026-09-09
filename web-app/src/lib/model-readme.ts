/**
 * Hugging Face model cards are markdown with a YAML header and, very often, a
 * block of hand-written HTML banner markup at the top: a centred `div`, a
 * banner image, a row of `<a>` badges. Rendered as markdown that header prints
 * as loose `key: value` text and the banner collapses into a run of link
 * labels with no links. This strips the header and rewrites the HTML that
 * appears in practice into markdown; anything it does not recognise is left
 * alone rather than guessed at.
 */

const FRONTMATTER = /^\uFEFF?---\r?\n[\s\S]*?\r?\n---\r?\n?/
const FENCE = /```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)/g
/** Layout-only tags: the text inside them is content, the tag is not. */
const WRAPPERS =
  /<\/?(?:div|p|span|center|figure|picture|source|font|small|table|tbody|thead|tr|td|th|h[1-6]|b|i|strong|em|u|sub|sup)\b[^>]*>/gi
// A private-use character: a README can hold any printable text, but not this.
const FENCE_MARK = '\uE000FENCE'
const FENCE_TOKEN = new RegExp(FENCE_MARK + '([0-9]+)' + FENCE_MARK, 'g')

export function cleanModelReadme(markdown: string): string {
  if (!markdown) return ''

  // Fenced code is documentation of HTML, not HTML to rewrite.
  const fences: string[] = []
  let body = markdown.replace(FENCE, (block) => {
    fences.push(block)
    return `${FENCE_MARK}${fences.length - 1}${FENCE_MARK}`
  })

  body = body.replace(FRONTMATTER, '')
  body = body.replace(/<!--[\s\S]*?-->/g, '')
  body = body.replace(/<img\b[^>]*>/gi, '')
  body = body.replace(/<br\s*\/?>/gi, '\n')
  // Badge rows are links; keeping them as markdown keeps them clickable.
  body = body.replace(
    /<a\b[^>]*href\s*=\s*["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi,
    (_all, href: string, label: string) => {
      const text = label.replace(/<[^>]+>/g, '').trim()
      return text ? `[${text}](${href.trim()})` : ''
    }
  )
  body = body.replace(WRAPPERS, '')
  body = body.replace(/[ \t]+$/gm, '')
  body = body.replace(/\n{3,}/g, '\n\n')

  return body
    .replace(FENCE_TOKEN, (_all, i: string) => fences[Number(i)])
    .trim()
}

/** The repo page behind a catalog model's README URL. */
export function huggingFaceRepoUrl(
  readmeUrl?: string,
  developer?: string,
  modelName?: string
): string | undefined {
  const fromReadme = readmeUrl?.match(
    /^(https:\/\/huggingface\.co\/[^/]+\/[^/]+)\/resolve\//
  )?.[1]
  if (fromReadme) return fromReadme
  if (!modelName) return undefined
  if (modelName.includes('/')) return `https://huggingface.co/${modelName}`
  return developer
    ? `https://huggingface.co/${developer}/${modelName}`
    : undefined
}

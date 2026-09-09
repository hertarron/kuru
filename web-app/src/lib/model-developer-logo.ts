/**
 * A logo for a catalog model, chosen from the model's own name.
 *
 * The catalog's `developer` field is whoever uploaded the repo, so requants and
 * finetunes carry the uploader rather than the model: `unsloth/DeepSeek-R1` is
 * a DeepSeek. Matching the title first keeps the mark on the family that
 * actually made the weights, and the developer is consulted only when the
 * title says nothing.
 *
 * Assets are the provider logos in `public/images/model-provider`. Families
 * with no file yet fall back to the card's monogram; adding one is a file plus
 * a row here. Order is significant: the first match wins, so a family that
 * builds on another (a Llama distill, a Nemotron) is listed above it.
 */
export type ModelLogo = {
  src: string
  /**
   * The art is white, so it is invisible on a light background. The card
   * inverts it in the light theme rather than shipping two files.
   */
  lightArt?: boolean
}

const LOGO_PATTERNS: [RegExp, ModelLogo][] = [
  [/\b(jan|janhq|menlo)\b/, { src: '/images/model-provider/jan.png' }],
  [
    /\b(mistral|mistralai|mixtral|ministral|magistral|codestral|devstral)/,
    { src: '/images/model-provider/mistral.svg' },
  ],
  [/\b(gemma|gemini|google)/, { src: '/images/model-provider/gemini.svg' }],
  [/\b(nemotron|nvidia)/, { src: '/images/model-provider/nvidia.svg' }],
  [/\bdeepseek/, { src: '/images/model-provider/deepseek.svg' }],
  [/\b(qwen|qwq|qvq|tongyi)/, { src: '/images/model-provider/qwen.svg' }],
  [
    /\b(openai|gpt-oss|gpt-[0-9j])/,
    { src: '/images/model-provider/openai.svg', lightArt: true },
  ],
  [/\b(cohere|command-[ar])/, { src: '/images/model-provider/cohere.svg' }],
  [/\b(anthropic|claude)/, { src: '/images/model-provider/anthropic.svg' }],
  [/\b(xai|grok)/, { src: '/images/model-provider/xai.svg' }],
  [/\bminimax/, { src: '/images/model-provider/minimax.svg' }],
  [/\b(kimi|moonshot)/, { src: '/images/model-provider/kimi.svg' }],
  [/\b(glm|chatglm|zhipu|thudm)/, { src: '/images/model-provider/zhipu.svg' }],
  [
    /\b(microsoft|phi-[0-9])/,
    { src: '/images/model-provider/microsoft.svg' },
  ],
  // Last of the families: a Llama derivative usually names its real parent
  // first (DeepSeek-R1-Distill-Llama, Nemotron), and that parent should win.
  [/llama|\bmeta\b/, { src: '/images/model-provider/meta.svg' }],
]

// Underscores and slashes separate the org from the model in catalog ids, so
// flatten every separator to one the word boundaries can see.
const normalize = (value: string) =>
  value.toLowerCase().replace(/[_/.\s]+/g, '-')

const match = (value?: string | null) =>
  value
    ? LOGO_PATTERNS.find(([pattern]) => pattern.test(normalize(value)))?.[1]
    : undefined

export function getModelLogo(
  modelName?: string | null,
  developer?: string | null
): ModelLogo | undefined {
  return match(modelName) ?? match(developer)
}

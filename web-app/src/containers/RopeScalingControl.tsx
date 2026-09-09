import { CardItem } from '@/containers/Card'
import { DynamicControllerSetting } from '@/containers/dynamicControllerSetting'

type SettingLike = {
  key: string
  title?: string
  description?: string
  controller_type?: string
  controller_props?: Record<string, unknown> & { value?: unknown }
}

type RopeScalingControlProps = {
  /** Raw provider settings; the four rope keys are read from here. */
  settings: SettingLike[]
  /** Persisted like any other provider edit (yaml + router restart). */
  onWrite: (patch: Record<string, string | number>) => void
}

/**
 * The four RoPE knobs as one group of `CardItem` rows, matching every other
 * row in Advanced.
 *
 * Only the method is on show at rest: the scale factor is meaningless while
 * scaling is off, and `rope_freq_scale` is `rope_scale` written as its inverse
 * — the same knob twice — so it appears only when a stored value would
 * otherwise be stranded out of sight.
 *
 * Kuru-specific: the planner multiplies the trained-context cap by the factor
 * read here (`rope_scale`, else `1 / rope_freq_scale`), so a scaled model
 * plans past its trained context instead of shortfalling at it.
 */
export function RopeScalingControl({
  settings,
  onWrite,
}: RopeScalingControlProps) {
  const numeric = (key: string) => {
    const n = Number(settings.find((s) => s.key === key)?.controller_props?.value)
    return Number.isFinite(n) ? n : undefined
  }
  const method =
    (settings.find((s) => s.key === 'rope_scaling')?.controller_props
      ?.value as string) || 'none'
  const freqScale = numeric('rope_freq_scale')

  // Writing a half-typed entry ("", "-") would drop the key in `preset.ts`,
  // which silently reverts the setting the user is in the middle of typing.
  const writeNumber = (key: string) => (next: string | number | boolean) => {
    if (typeof next === 'number' && Number.isFinite(next)) onWrite({ [key]: next })
  }

  return (
    <>
      <CardItem
        title="RoPE scaling"
        description="Runs a model past the context length it was trained for. Linear stretches positions evenly; YaRN holds up better beyond about 2x. Both cost quality, so leave this off unless the model asks for it."
        actions={
          <DynamicControllerSetting
            controllerType="dropdown"
            controllerProps={{
              value: method,
              options: [
                { value: 'none', name: 'Off' },
                { value: 'linear', name: 'Linear' },
                { value: 'yarn', name: 'YaRN' },
              ],
            }}
            onChange={(next) => onWrite({ rope_scaling: String(next) })}
          />
        }
      />

      {method !== 'none' && (
        <CardItem
          title="Scale factor"
          description="How far to stretch. 2 doubles the context the model can address."
            actions={
            <DynamicControllerSetting
              controllerType="input"
              controllerProps={{
                value: numeric('rope_scale') ?? 1,
                type: 'number',
                min: 0,
                step: 0.01,
              }}
              onChange={writeNumber('rope_scale')}
            />
          }
        />
      )}

      <CardItem
        title="Frequency base"
        description="Overrides the model's own RoPE base frequency. 0 keeps it. Set this only if the model gives you a number."
        actions={
          <DynamicControllerSetting
            controllerType="input"
            controllerProps={{
              value: numeric('rope_freq_base') ?? 0,
              type: 'number',
              min: 0,
              step: 1000,
            }}
            onChange={writeNumber('rope_freq_base')}
          />
        }
      />

      {freqScale !== undefined && freqScale !== 1 && (
        <CardItem
          title="Frequency scale"
          description="The scale factor written as its inverse (0.5 means 2x). Set it back to 1 to use Scale factor instead."
            actions={
            <DynamicControllerSetting
              controllerType="input"
              controllerProps={{
                value: freqScale,
                type: 'number',
                min: 0,
                step: 0.01,
              }}
              onChange={writeNumber('rope_freq_scale')}
            />
          }
        />
      )}
    </>
  )
}

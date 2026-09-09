import { useMemo } from 'react'
import {
  IconAdjustmentsHorizontal,
  IconSettings,
  IconUser,
} from '@tabler/icons-react'
import { Link } from '@tanstack/react-router'

import { route } from '@/constants/routes'

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Button } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'

import { AvatarEmoji } from '@/containers/AvatarEmoji'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { ParametersSection } from '@/containers/ParametersSection'
import { useCharacters } from '@/hooks/useCharacters'
import { useModelProvider } from '@/hooks/useModelProvider'
import { paramsSettings, type ParamDef } from '@/lib/predefinedParams'
import { isPredefinedRemoteProvider } from '@/lib/providerCaps'
import { cn } from '@/lib/utils'

interface SamplerPopoverProps {
  /** Provider ID of the currently-selected model, if any. Used to scope the
   *  chip palette to a single provider's capabilities. */
  providerId?: string
  /** Currently-selected model ID. Drives model-family rejection (e.g. o1/gpt-5
   *  reasoning models reject temperature/top_p). */
  modelId?: string
  /** Optional character switcher. When omitted, the header shows the active
   *  character name as static text. */
  characterSwitcher?: {
    characters: Character[]
    currentThread: Thread | undefined
    selectedCharacterId: string | undefined
    setSelectedCharacterId: (id: string) => void
    updateCurrentThreadCharacter: (character?: Character | Assistant) => void
  }
}

export function SamplerPopover({
  providerId,
  modelId,
  characterSwitcher,
}: SamplerPopoverProps) {
  const currentCharacter = useCharacters((s) => s.currentCharacter)
  const updateCharacter = useCharacters((s) => s.updateCharacter)
  const charactersLoading = useCharacters((s) => s.loading)
  const providers = useModelProvider((s) => s.providers)

  const scopedProviders = useMemo(() => {
    if (providerId) {
      const p = providers.find((x) => x.provider === providerId)
      return p ? [p] : []
    }
    return providers.filter((p) => p.active)
  }, [providers, providerId])

  // The header must match the switcher's checkmark in every context:
  //  - inside a thread: the thread's assigned character, including "None"
  //    (model-only) which must NOT fall back to the global default;
  //  - off-thread with a switcher (home screen): the dropdown selection
  //    (selectedCharacterId), since that's what a new chat will use;
  //  - no switcher at all: the global current character.
  const inThread = !!characterSwitcher?.currentThread
  const threadCharacter = characterSwitcher?.currentThread?.assistants?.[0]
  const isThreadCharacter =
    !!threadCharacter && threadCharacter.id !== 'model-only'
  const selectedCharacter = characterSwitcher
    ? characterSwitcher.characters.find(
        (c) => c.id === characterSwitcher.selectedCharacterId
      )
    : currentCharacter
  const activeCharacter: Character | undefined = inThread
    ? isThreadCharacter
      ? threadCharacter
      : undefined
    : selectedCharacter

  const params = activeCharacter?.parameters ?? {}
  const samplerKeys = Object.keys(params).filter((k) => k in paramsSettings)
  const hasOverrides = samplerKeys.length > 0

  const writeParams = (next: Record<string, unknown>) => {
    if (!activeCharacter) return
    const updated = { ...activeCharacter, parameters: next }
    if (isThreadCharacter && characterSwitcher) {
      // Sync the thread's copy so inference picks it up immediately; mirror
      // into the canonical character only when it still exists.
      characterSwitcher.updateCurrentThreadCharacter(updated)
      if (characterSwitcher.characters.some((c) => c.id === updated.id)) {
        updateCharacter(updated)
      }
    } else {
      updateCharacter(updated)
    }
  }

  const handleToggle = (def: ParamDef) => {
    if (def.key in params) {
      const next = { ...params }
      delete next[def.key]
      writeParams(next)
    } else {
      writeParams({ ...params, [def.key]: def.value })
    }
  }

  const handleChange = (key: string, value: unknown) => {
    writeParams({ ...params, [key]: value })
  }

  const handleRemove = (key: string) => {
    const next = { ...params }
    delete next[key]
    writeParams(next)
  }

  const handleAddMany = (values: Record<string, unknown>) => {
    writeParams({ ...params, ...values })
  }

  const handleRemoveMany = (keys: string[]) => {
    const next = { ...params }
    for (const k of keys) delete next[k]
    writeParams(next)
  }

  const handleResetAll = () => {
    const next: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(params)) {
      if (!(k in paramsSettings)) next[k] = v
    }
    writeParams(next)
  }

  if (isPredefinedRemoteProvider(providerId)) return null

  const triggerLabel = charactersLoading
    ? 'Loading character…'
    : activeCharacter
      ? `Sampling — ${activeCharacter.name}`
      : 'Sampling'

  return (
    <Popover>
      <Tooltip>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label="Sampling parameters"
              className="relative"
              disabled={charactersLoading}
            >
              <IconAdjustmentsHorizontal
                size={18}
                className={cn(
                  'text-muted-foreground',
                  hasOverrides && 'text-primary'
                )}
              />
              {hasOverrides && (
                <span
                  aria-hidden
                  className="absolute top-0.5 right-0.5 size-1.5 rounded-full bg-primary"
                />
              )}
            </Button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent>
          <p>{triggerLabel}</p>
        </TooltipContent>
      </Tooltip>

      <PopoverContent
        align="start"
        side="top"
        sideOffset={8}
        collisionPadding={12}
        avoidCollisions
        className="w-[380px] p-0 flex flex-col max-h-[var(--radix-popover-content-available-height)]"
      >
        <div className="flex items-center justify-between gap-2 px-4 pt-3 pb-2 border-b">
          <CharacterHeader currentCharacter={activeCharacter} />
          <div className="flex items-center gap-1 shrink-0">
            {hasOverrides && (
              <Button variant="ghost" size="sm" onClick={handleResetAll}>
                Reset all
              </Button>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon-sm" asChild>
                  <Link to={route.settings.characters}>
                    <IconSettings size={16} className="text-muted-foreground" />
                  </Link>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Open character settings</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
        {activeCharacter ? (
          <div className="flex-1 min-h-0 overflow-y-auto px-4 py-3">
            <ParametersSection
              params={params}
              providers={scopedProviders}
              providerId={providerId}
              modelId={modelId}
              onToggle={handleToggle}
              onChange={handleChange}
              onRemove={handleRemove}
              onAddMany={handleAddMany}
              onRemoveMany={handleRemoveMany}
            />
          </div>
        ) : (
          <div className="text-xs text-muted-foreground px-4 py-3">
            Pick a character above to configure sampling.
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}

interface CharacterHeaderProps {
  currentCharacter: Character | undefined
}

/**
 * Whose sampling settings these are. A label, not a switcher: the character
 * pill in the chatbox toolbar is the one place characters get swapped, so a
 * second dropdown here would just be a second door to the same room.
 */
function CharacterHeader({ currentCharacter }: CharacterHeaderProps) {
  const { t } = useTranslation()
  return (
    <div className="flex items-center gap-1.5 min-w-0">
      {currentCharacter?.avatar ? (
        <AvatarEmoji
          avatar={currentCharacter.avatar}
          imageClassName="size-4 object-contain"
          textClassName="text-sm"
        />
      ) : (
        <IconUser size={14} className="text-muted-foreground" />
      )}
      <span className="text-sm font-medium truncate">
        {currentCharacter?.name ?? t('common:noCharacter')}
      </span>
    </div>
  )
}



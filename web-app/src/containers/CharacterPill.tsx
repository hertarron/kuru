import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { AvatarEmoji } from '@/containers/AvatarEmoji'
import { IconSettings, IconX } from '@tabler/icons-react'
import { Button } from '@/components/ui/button'
import { useNavigate } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { cn } from '@/lib/utils'
import { PILL_CHEVRON } from '@/constants/layout'
import { usePillLabel } from '@/hooks/usePillLabel'
import { labelForms } from '@/lib/pill-labels'
import { Fzf } from 'fzf'
import { ChevronsUpDown } from 'lucide-react'
import {
  defaultCharacter,
  useCharacters,
} from '@/hooks/useCharacters'
import { useCharacterSwitcher } from '@/hooks/useCharacterSwitcher'
import { useThreads } from '@/hooks/useThreads'
import { usePopoverAlign } from '@/hooks/usePopoverAlign'
import { useTranslation } from '@/i18n/react-i18next-compat'

type SearchableCharacter = {
  character: Character
  searchStr: string
}

/**
 * The character switcher pill in the thread and home headers -- the RP
 * counterpart of the model pill. Shows the active character (the thread's
 * embedded one when inside a chat, otherwise the last-used selection for new
 * chats) and opens a searchable picker. The cog jumps to the character's
 * editor page.
 */
export function CharacterPill({
  thread,
  compact = false,
}: {
  thread?: Thread
  /** Toolbar-sized pill for the chatbox row rather than the header. */
  compact?: boolean
}) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const {
    characters,
    currentCharacter,
    setCurrentCharacter,
    loading: charactersLoading,
  } = useCharacters()
  const updateCurrentThreadCharacter = useThreads(
    (state) => state.updateCurrentThreadCharacter
  )
  const open = useCharacterSwitcher((s) => s.open)
  const setOpen = useCharacterSwitcher((s) => s.setOpen)
  const setCycleHandler = useCharacterSwitcher((s) => s.setCycleHandler)
  // Panel is w-72; measured at open so it hugs the pill's free side.
  const { pillRef, align, measure } = usePopoverAlign(288)

  const [searchValue, setSearchValue] = useState('')
  const searchInputRef = useRef<HTMLInputElement>(null)

  const embedded = thread?.assistants?.[0]
  const isThreadEmbedded = !!embedded && embedded.id !== 'model-only'

  // Inside a thread the embedded copy wins; on home the store selection does.
  // While the library hydrates there is no selection yet -- don't flash the
  // built-in "Assistant" before the user's default lands.
  const activeCharacter = useMemo(() => {
    if (thread) return isThreadEmbedded ? embedded : undefined
    return currentCharacter ?? (charactersLoading ? undefined : defaultCharacter)
  }, [thread, isThreadEmbedded, embedded, currentCharacter, charactersLoading])

  const switchTo = (character: Character | undefined) => {
    if (thread) {
      updateCurrentThreadCharacter(character)
    } else {
      setCurrentCharacter(character)
    }
  }

  // Global Cmd/Ctrl+J cycling, same mechanism the model/assistant switchers used.
  const cycleRef = useRef<() => void>(() => {})
  cycleRef.current = () => {
    if (characters.length <= 1) return
    const activeId = thread ? (isThreadEmbedded ? embedded?.id : undefined) : currentCharacter?.id
    const idx = characters.findIndex((c) => c.id === activeId)
    const next = characters[(idx + 1) % characters.length]
    switchTo(next)
  }

  useEffect(() => {
    const handler = () => cycleRef.current()
    setCycleHandler(handler)
    return () => {
      setCycleHandler(null)
      setOpen(false)
    }
  }, [setCycleHandler, setOpen])

  const filtered = useMemo(() => {
    if (!searchValue.trim()) return null
    const fzf = new Fzf(characters, {
      selector: (c) => `${c.name} ${c.description ?? ''}`,
    })
    return fzf
      .find(searchValue)
      .map((r): SearchableCharacter => ({ character: r.item, searchStr: searchValue }))
  }, [searchValue, characters])

  const list = filtered ?? characters.map((c) => ({ character: c, searchStr: '' }))

  const displayName = activeCharacter?.name ?? t('common:noCharacter')
  // The row decides how much of the name fits; see PillRow.
  const forms = useMemo(() => labelForms(displayName), [displayName])
  const {
    label: fittedName,
    pillRef: measurePill,
    labelRef,
  } = usePillLabel('character', forms)
  const setPill = useCallback(
    (el: HTMLDivElement | null) => {
      pillRef.current = el
      measurePill(el)
    },
    [pillRef, measurePill]
  )

  return (
    <Popover
      open={open}
      onOpenChange={(v) => {
        if (v) measure()
        setOpen(v)
        if (!v) setSearchValue('')
      }}
    >
      {/* The whole pill is the trigger -- the same arrangement the model
          pill uses, so the panel anchors to the pill's edges. The cog
          opts out with stopPropagation below. */}
      <PopoverTrigger asChild>
        <div ref={setPill} className={cn(
            // Header pills stand alone; the compact form sits in the chatbox
            // toolbar row at the same 28px height as its buttons. min-w-0
            // lets the row squeeze it once space runs out.
            'border relative z-20 flex items-center gap-1.5 rounded-full min-w-0 cursor-pointer',
            compact ? 'pl-2.5 pr-1 h-7 text-xs' : 'pl-4 pr-2 h-10'
          )}>
        <button
          type="button"
          className="font-medium cursor-pointer flex items-center gap-1.5 relative z-20 min-w-0"
          aria-label={t('common:switchCharacter')}
        >
          <span className="flex items-center gap-1.5 min-w-0">
            <AvatarEmoji
              avatar={activeCharacter?.avatar}
              imageClassName="size-4 object-contain"
              textClassName="text-sm"
            />
            {(!compact || fittedName) && (
              <span
                ref={labelRef}
                className={cn(
                  'text-foreground truncate leading-normal',
                  !activeCharacter && 'text-muted-foreground'
                )}
              >
                {compact ? fittedName : displayName}
              </span>
            )}
          </span>
          <ChevronsUpDown
            className={cn(
              'size-4 shrink-0 text-muted-foreground',
              compact && PILL_CHEVRON
            )}
          />
        </button>

        {/* Inside the trigger box, but a click edits the character rather
            than opening the picker. */}
        <Button
          variant="ghost"
          size="icon-xs"
          className="shrink-0"
          aria-label={
            activeCharacter
              ? t('common:editCharacter')
              : t('common:manageCharacters')
          }
          onClick={(e) => {
            e.stopPropagation()
            navigate(
              activeCharacter
                ? {
                    to: route.characters.detail,
                    params: { characterId: activeCharacter.id },
                  }
                : { to: route.settings.characters }
            )
          }}
        >
          <IconSettings size={18} className="text-muted-foreground" />
        </Button>
        </div>
      </PopoverTrigger>

      <PopoverContent
          className="w-72 min-w-70 max-w-[90vw] p-0 backdrop-blur-2xl bg-background/95 border"
          align={align}
          side={compact ? 'top' : 'bottom'}
        >
          <div className="flex flex-col size-full">
            {/* Search input */}
            <div className="relative p-2 border-b">
              <input
                ref={searchInputRef}
                value={searchValue}
                onChange={(e) => setSearchValue(e.target.value)}
                placeholder={t('common:searchCharacters')}
                className="text-sm font-normal outline-0 w-full"
              />
              {searchValue.length > 0 && (
                <div className="absolute right-2 top-0 bottom-0 flex items-center justify-center">
                  <IconX
                    size={16}
                    className="text-muted-foreground cursor-pointer"
                    onClick={() => setSearchValue('')}
                  />
                </div>
              )}
            </div>

            {/* Character list */}
            <div className="max-h-80 overflow-y-auto py-1">
              {list.length === 0 ? (
                <div className="py-3 px-4 text-sm text-muted-foreground">
                  {t('common:noCharactersFound', { searchValue })}
                </div>
              ) : (
                list.map(({ character }) => {
                  const isSelected =
                    !!activeCharacter && activeCharacter.id === character.id
                  return (
                    <div
                      key={character.id}
                      onClick={() => {
                        switchTo(character)
                        setOpen(false)
                      }}
                      className={cn(
                        'mx-1 mb-1 px-2 py-1.5 rounded-sm cursor-pointer flex items-center gap-2 transition-all duration-200',
                        'hover:bg-secondary/40',
                        isSelected &&
                          'bg-primary/15 hover:bg-primary/15 ring-1 ring-primary/40'
                      )}
                    >
                      <div className="size-6 shrink-0 flex items-center justify-center bg-secondary dark:bg-secondary/40 rounded-md overflow-hidden">
                        <AvatarEmoji
                          avatar={character.avatar}
                          imageClassName="size-5 object-contain"
                          textClassName="text-base"
                        />
                      </div>
                      <div className="flex flex-col min-w-0 flex-1">
                        <span className="text-sm truncate">{character.name}</span>
                        {character.description && (
                          <span className="text-xs text-muted-foreground truncate">
                            {character.description}
                          </span>
                        )}
                      </div>
                      {isSelected && (
                        <span className="ml-auto text-xs text-muted-foreground shrink-0">
                          ✓
                        </span>
                      )}
                    </div>
                  )
                })
              )}
            </div>

            {/* Manage footer */}
            <div className="border-t p-1 flex items-center justify-between">
              <button
                type="button"
                className="text-xs text-muted-foreground px-2 py-1.5 rounded-sm hover:bg-secondary/40 cursor-pointer"
                onClick={() => {
                  setOpen(false)
                  navigate({ to: route.settings.characters })
                }}
              >
                {t('common:manageCharacters')}
              </button>
            </div>
          </div>
        </PopoverContent>
    </Popover>
  )
}

export default CharacterPill


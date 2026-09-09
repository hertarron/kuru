import { createFileRoute, useNavigate, useRouter } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { useEffect, useMemo, useRef, useState } from 'react'

import HeaderPage from '@/containers/HeaderPage'
import { AvatarEmoji } from '@/containers/AvatarEmoji'
import { IconBook, IconChevronLeft, IconChevronRight, IconCheck, IconFileUpload, IconMoodSmile, IconPlus, IconTrash, IconX, IconChevronDown } from '@tabler/icons-react'
import EmojiPicker, { EmojiClickData, Theme } from 'emoji-picker-react'
import { generateId } from 'ai'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { DeleteCharacterDialog } from '@/containers/dialogs/DeleteCharacterDialog'
import { useCharacters, defaultCharacter } from '@/hooks/useCharacters'
import { cn } from '@/lib/utils'
import {
  cardToCharacter,
  fileToAvatarDataUrl,
  parseCharacterCard,
} from '@/lib/character-card'
import { adoptCardLorebook } from '@/hooks/useLorebooks'
import { inferCharacterKind } from '@/lib/character-card'
import { useTheme } from '@/hooks/useTheme'
import { useModelProvider } from '@/hooks/useModelProvider'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import { useLorebooks } from '@/hooks/useLorebooks'
import {
  type ParamDef,
} from '@/lib/predefinedParams'
import { ParametersSection } from '@/containers/ParametersSection'
import { useTranslation } from '@/i18n/react-i18next-compat'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.characters.detail as any)({
  component: CharacterDetailContent,
})

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <label className="text-sm font-medium">{children}</label>
}

function FieldHint({ children }: { children: React.ReactNode }) {
  return <div className="text-xs text-muted-foreground">{children}</div>
}

const inputSpellCheckProps = (enabled: boolean) => ({
  spellCheck: enabled,
  'data-gramm': enabled,
  'data-gramm_editor': enabled,
  'data-gramm_grammarly': enabled,
})

function CharacterDetailContent() {
  const { characterId } = Route.useParams()
  const navigate = useNavigate()
  const router = useRouter()
  const { t } = useTranslation()
  const { characters, addCharacter, updateCharacter, deleteCharacter } =
    useCharacters()
  const { isDark } = useTheme()
  const spellCheckChatInput = useGeneralSetting((s) => s.spellCheckChatInput)
  const providers = useModelProvider((s) => s.providers)
  const activeProviders = useMemo(
    () => providers.filter((p) => p.active),
    [providers]
  )

  const isNew = characterId === 'new'
  const existing = characters.find((c) => c.id === characterId)

  // Form state
  const [avatar, setAvatar] = useState<string | undefined>(existing?.avatar)
  const [name, setName] = useState(existing?.name ?? '')
  const [description, setDescription] = useState(existing?.description ?? '')
  const [personality, setPersonality] = useState(existing?.personality ?? '')
  // Existing characters keep whatever they were inferred as until the user
  // says otherwise; a character made here is a roleplay one by default.
  const [kind, setKind] = useState<CharacterKind>(
    existing ? inferCharacterKind(existing) : 'roleplay'
  )
  const [scenario, setScenario] = useState(existing?.scenario ?? '')
  const [firstMes, setFirstMes] = useState(existing?.first_mes ?? '')
  const [alternateGreetings, setAlternateGreetings] = useState<string[]>(
    existing?.alternate_greetings ?? []
  )
  const [mesExample, setMesExample] = useState(existing?.mes_example ?? '')
  const [systemPromptOverride, setSystemPromptOverride] = useState(
    existing?.system_prompt ?? ''
  )
  const [creatorNotes, setCreatorNotes] = useState(existing?.creator_notes ?? '')
  const [tagsInput, setTagsInput] = useState(
    (existing?.tags ?? []).join(', ')
  )
  const [params, setParams] = useState<Record<string, unknown>>(
    () => existing?.parameters ?? {}
  )
  const [lorebookIds, setLorebookIds] = useState<string[]>(
    existing?.lorebookIds ?? []
  )
  const allLorebooks = useLorebooks((s) => s.lorebooks)
  const [showEmojiPicker, setShowEmojiPicker] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [nameError, setNameError] = useState<string | null>(null)
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)
  // Brief visual ack on the Save button — the write itself is fire-and-forget.
  const [saveState, setSaveState] = useState<'idle' | 'saved'>('idle')
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(
    () => () => {
      if (savedTimerRef.current) clearTimeout(savedTimerRef.current)
    },
    []
  )
  const emojiPickerRef = useRef<HTMLDivElement>(null)
  const emojiTriggerRef = useRef<HTMLDivElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const cardInputRef = useRef<HTMLInputElement>(null)
  const [importError, setImportError] = useState<string | null>(null)
  // Greetings: index 0 is the primary greeting, 1..n are the alternates.
  // One text box flips between them via the ‹ › arrows.
  const [greetIndex, setGreetIndex] = useState(0)
  const totalGreetings = 1 + alternateGreetings.length

  useEffect(() => {
    if (!isNew && !existing) {
      // Unknown id: back to the library.
      navigate({ to: route.characters.index })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isNew, !!existing])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        emojiPickerRef.current &&
        emojiTriggerRef.current &&
        !emojiPickerRef.current.contains(event.target as Node) &&
        !emojiTriggerRef.current.contains(event.target as Node)
      ) {
        setShowEmojiPicker(false)
      }
    }
    if (showEmojiPicker) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showEmojiPicker])

  if (!isNew && !existing) return null

  const spellProps = inputSpellCheckProps(spellCheckChatInput)

  const buildCharacter = (): Character => ({
    ...(existing ?? {}),
    avatar,
    id: existing?.id || generateId(),
    name: name.trim() || t('characters:unnamed'),
    created_at: existing?.created_at || Date.now() / 1000,
    description,
    kind,
    instructions:
      systemPromptOverride.trim() !== '' ? '' : (existing?.instructions ?? ''),
    personality: personality || undefined,
    scenario: scenario || undefined,
    first_mes: firstMes || undefined,
    alternate_greetings: alternateGreetings.length
      ? alternateGreetings.filter((g) => g.trim())
      : undefined,
    mes_example: mesExample || undefined,
    system_prompt: systemPromptOverride || undefined,
    tags: tagsInput.trim()
      ? tagsInput.split(',').map((s) => s.trim()).filter(Boolean)
      : undefined,
    creator_notes: creatorNotes || undefined,
    lorebookIds: lorebookIds.length ? lorebookIds : undefined,
    parameters: params,
  })

  const attachedBooks = lorebookIds
    .map((id) => allLorebooks.find((b) => b.id === id))
    .filter((b): b is NonNullable<typeof b> => !!b)
  const attachableBooks = allLorebooks.filter(
    (b) => !lorebookIds.includes(b.id)
  )

  const handleSave = () => {
    if (!name.trim()) {
      setNameError(t('characters:nameRequired'))
      return
    }
    const character = buildCharacter()
    if (existing) {
      updateCharacter(character)
    } else {
      addCharacter(character)
    }
    if (isNew) {
      navigate({
        to: route.characters.detail,
        params: { characterId: character.id },
        replace: true,
      })
    }
    setNameError(null)
    setSaveState('saved')
    if (savedTimerRef.current) clearTimeout(savedTimerRef.current)
    savedTimerRef.current = setTimeout(() => setSaveState('idle'), 1500)
  }

  /**
   * Loads a .png/.json card into the form rather than straight into the
   * library, so an import is reviewed and saved like anything typed here.
   */
  const handleImportCard = async (file: File | undefined) => {
    if (!file) return
    setImportError(null)
    try {
      const card = await parseCharacterCard(await file.arrayBuffer(), file.name)
      const imported = cardToCharacter(card)
      setName(imported.name ?? '')
      setDescription(imported.description ?? '')
      setPersonality(imported.personality ?? '')
      // Anything arriving as a card is a roleplay character by definition.
      setKind('roleplay')
      setScenario(imported.scenario ?? '')
      setFirstMes(imported.first_mes ?? '')
      setAlternateGreetings(imported.alternate_greetings ?? [])
      setGreetIndex(0)
      setMesExample(imported.mes_example ?? '')
      setSystemPromptOverride(imported.system_prompt ?? '')
      setCreatorNotes(imported.creator_notes ?? '')
      setTagsInput((imported.tags ?? []).join(', '))
      if (card.character_book) {
        setLorebookIds([adoptCardLorebook(card.character_book)])
      }
      if (file.name.toLowerCase().endsWith('.png')) {
        try {
          setAvatar(await fileToAvatarDataUrl(file))
        } catch {
          // Avatar is optional; the rest of the card still imported.
        }
      }
      setNameError(null)
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e))
    } finally {
      if (cardInputRef.current) cardInputRef.current.value = ''
    }
  }

  // Back means "where I came from" — the editor is reachable from settings,
  // the hub, a chat header and the library, and each of those expects to get
  // its scroll position back.
  const goBack = () => {
    if (router.history.canGoBack()) router.history.back()
    else navigate({ to: route.characters.index })
  }

  const handleStartChat = () => {
    const character = buildCharacter()
    if (existing) updateCharacter(character)
    else addCharacter(character)
    const { setCurrentCharacter } = useCharacters.getState()
    setCurrentCharacter(character)
    navigate({ to: route.home })
  }

  const handleDelete = () => {
    if (!existing) return
    deleteCharacter(existing.id)
    navigate({ to: route.characters.index })
  }

  const handleAvatarImage = async (file: File | undefined) => {
    if (!file) return
    try {
      setAvatar(await fileToAvatarDataUrl(file))
    } catch {
      // Ignore unreadable images; the picker stays usable.
    }
  }

  return (
    <div className="flex flex-col h-full w-full">
      <HeaderPage>
        {/* Wrapper ignores clicks so the root drag-strip stays grabbable in
            the blank middle; each cluster opts back in. */}
        <div className={cn('pointer-events-none flex items-center justify-between w-full pr-3 h-8 relative z-20', !IS_MACOS && 'pr-30')}>
          <div className="pointer-events-auto flex items-center gap-3 min-w-0">
            <button
              type="button"
              className="cursor-pointer"
              onClick={goBack}
              aria-label={t('common:back')}
            >
              <IconChevronLeft size={20} className="text-muted-foreground" />
            </button>
            <span className="font-medium text-base font-studio truncate">
              {isNew ? t('characters:newTitle') : name}
            </span>
          </div>
          <div className="pointer-events-auto flex items-center gap-2 shrink-0">
            {!isNew && existing?.id !== defaultCharacter.id && (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setDeleteConfirmOpen(true)}
                >
                  <IconTrash size={16} className="text-destructive" />
                  {t('common:delete')}
                </Button>
              </>
            )}
            {!isNew && (
              <Button variant="outline" size="sm" onClick={handleStartChat}>
                {t('characters:startChat')}
              </Button>
            )}
            {isNew && (
              <>
                <input
                  ref={cardInputRef}
                  type="file"
                  accept=".png,.json,application/json,image/png"
                  className="hidden"
                  onChange={(e) => handleImportCard(e.target.files?.[0])}
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => cardInputRef.current?.click()}
                >
                  <IconFileUpload size={16} />
                  {t('common:importCharacter')}
                </Button>
              </>
            )}
            <Button
              size="sm"
              variant={saveState === 'saved' ? 'outline' : 'default'}
              onClick={handleSave}
            >
              {saveState === 'saved' ? (
                <>
                  <IconCheck size={16} />
                  Saved
                </>
              ) : (
                t('common:save')
              )}
            </Button>
          </div>
        </div>
      </HeaderPage>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 pb-[106px] pt-6 flex flex-col gap-5">
          {importError && (
            <p className="text-sm text-destructive">{importError}</p>
          )}
          {/* Identity */}
          <div className="flex items-center gap-4">
            <div className="relative">
              <div
                ref={emojiTriggerRef}
                className="size-16 border rounded-xl flex items-center justify-center cursor-pointer bg-secondary/30 hover:bg-secondary/50 transition-colors overflow-hidden"
                onClick={() => setShowEmojiPicker(!showEmojiPicker)}
              >
                {avatar ? (
                  <AvatarEmoji
                    avatar={avatar}
                    imageClassName="size-full object-cover"
                    textClassName="text-3xl"
                  />
                ) : (
                  <IconMoodSmile size={26} className="text-muted-foreground" />
                )}
              </div>
              <div className="relative" ref={emojiPickerRef}>
                <EmojiPicker
                  open={showEmojiPicker}
                  theme={isDark ? ('dark' as Theme) : ('light' as Theme)}
                  className="absolute! z-40! overflow-y-auto!"
                  height={350}
                  lazyLoadEmojis
                  previewConfig={{ showPreview: false }}
                  onEmojiClick={(emojiData: EmojiClickData) => {
                    if (emojiData.isCustom && emojiData.imageUrl) {
                      setAvatar(emojiData.imageUrl)
                    } else {
                      setAvatar(emojiData.emoji)
                    }
                    setShowEmojiPicker(false)
                  }}
                />
              </div>
            </div>
            <div className="flex flex-col gap-2 min-w-0 flex-1">
              <label className="text-sm">{t('common:name')}</label>
              <Input
                value={name}
                onChange={(e) => {
                  setName(e.target.value)
                  if (e.target.value.trim()) setNameError(null)
                }}
                placeholder={t('characters:namePlaceholder')}
              />
              {nameError && (
                <div className="text-xs text-destructive">{nameError}</div>
              )}
              <div className="flex items-center gap-2">
                <input
                  ref={imageInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) =>
                    handleAvatarImage(e.target.files?.[0])
                  }
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => imageInputRef.current?.click()}
                >
                  {t('characters:uploadPortrait')}
                </Button>
                {avatar && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setAvatar(undefined)}
                  >
                    <IconX size={14} />
                    {t('common:remove')}
                  </Button>
                )}
              </div>
            </div>
          </div>

          {/* Roleplay vs assistant: the switch that decides whether any of
              the RP machinery applies to this character at all. */}
          <div className="flex flex-col gap-2">
            <SectionLabel>Kind</SectionLabel>
            <div className="flex items-center gap-2">
              {(
                [
                  ['roleplay', 'Roleplay character', 'Greeting, world info and personas apply.'],
                  ['assistant', 'Assistant', 'A coding or task helper. No greeting, no world info, no persona.'],
                ] as Array<[CharacterKind, string, string]>
              ).map(([value, label, hint]) => (
                <Button
                  key={value}
                  variant={kind === value ? 'default' : 'outline'}
                  size="sm"
                  // The outline variant carries a border and default does not,
                  // so the row jumps on select without these.
                  className="transition-none border border-transparent"
                  onClick={() => setKind(value)}
                  title={hint}
                >
                  {label}
                </Button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              {kind === 'roleplay'
                ? 'Greetings are seeded into new chats, and the world-info and persona pills appear in the chatbox.'
                : 'New chats start empty, attached lorebooks and personas are ignored, and their pills are hidden.'}
            </p>
          </div>

          {/* Lorebooks */}
          <div className="flex flex-col gap-2">
            <SectionLabel>Lorebooks</SectionLabel>
            {attachedBooks.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {attachedBooks.map((book) => (
                  <span
                    key={book.id}
                    className="flex items-center gap-1.5 text-xs pl-2.5 pr-1.5 py-1 rounded-full bg-secondary"
                  >
                    <IconBook className="size-3.5 text-muted-foreground" />
                    <span className="truncate max-w-52">{book.name}</span>
                    <button
                      type="button"
                      title="Detach"
                      className="text-muted-foreground hover:text-foreground cursor-pointer"
                      onClick={() =>
                        setLorebookIds(
                          lorebookIds.filter((id) => id !== book.id)
                        )
                      }
                    >
                      <IconX className="size-3.5" />
                    </button>
                  </span>
                ))}
              </div>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="self-start"
                  disabled={attachableBooks.length === 0}
                >
                  <IconPlus className="size-4" />
                  Attach lorebook
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="max-h-80 overflow-y-auto">
                {attachableBooks.map((book) => (
                  <DropdownMenuItem
                    key={book.id}
                    className="cursor-pointer"
                    onClick={() => setLorebookIds([...lorebookIds, book.id])}
                  >
                    {book.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <FieldHint>
              {allLorebooks.length === 0
                ? 'No lorebooks yet. Create or import one under Settings › Lorebooks.'
                : 'Attached books are referenced, not copied — editing a book updates every character using it.'}
            </FieldHint>
          </div>

          {/* Core definitions */}
          <div className="flex flex-col gap-2">
            <SectionLabel>{t('characters:description')}</SectionLabel>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={5}
              placeholder={t('characters:descriptionPlaceholder')}
              {...spellProps}
            />
            <FieldHint>{t('characters:descriptionHint')}</FieldHint>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div className="flex flex-col gap-2">
              <SectionLabel>{t('characters:personality')}</SectionLabel>
              <Textarea
                value={personality}
                onChange={(e) => setPersonality(e.target.value)}
                rows={4}
                placeholder={t('characters:personalityPlaceholder')}
                {...spellProps}
              />
              <FieldHint>{t('characters:personalityHint')}</FieldHint>
            </div>
            <div className="flex flex-col gap-2">
              <SectionLabel>{t('characters:scenario')}</SectionLabel>
              <Textarea
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                rows={4}
                placeholder={t('characters:scenarioPlaceholder')}
                {...spellProps}
              />
              <FieldHint>{t('characters:scenarioHint')}</FieldHint>
            </div>
          </div>

          {/* Greetings — one text box; index 0 is the primary greeting,
              1..n the alternates, flipped with the ‹ › arrows. */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <SectionLabel>{t('characters:greeting')}</SectionLabel>
              <div className="flex items-center gap-1">
                {totalGreetings > 1 && (
                  <span className="text-xs text-muted-foreground tabular-nums mr-1">
                    {greetIndex + 1}/{totalGreetings}
                  </span>
                )}
                <Button
                  variant="ghost"
                  size="icon-xs"
                  disabled={greetIndex === 0}
                  onClick={() => setGreetIndex(greetIndex - 1)}
                  aria-label="Previous greeting"
                >
                  <IconChevronLeft size={14} />
                </Button>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  disabled={greetIndex >= totalGreetings - 1}
                  onClick={() => setGreetIndex(greetIndex + 1)}
                  aria-label="Next greeting"
                >
                  <IconChevronRight size={14} />
                </Button>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  onClick={() => {
                    setAlternateGreetings((prev) => [...prev, ''])
                    setGreetIndex(totalGreetings)
                  }}
                  aria-label="Add alternate greeting"
                >
                  <IconPlus size={14} />
                </Button>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  disabled={greetIndex === 0}
                  onClick={() => {
                    setAlternateGreetings((prev) =>
                      prev.filter((_, j) => j !== greetIndex - 1)
                    )
                    setGreetIndex(Math.max(0, greetIndex - 1))
                  }}
                  aria-label="Remove greeting"
                >
                  <IconX size={14} />
                </Button>
              </div>
            </div>
            <Textarea
              value={
                greetIndex === 0
                  ? firstMes
                  : (alternateGreetings[greetIndex - 1] ?? '')
              }
              onChange={(e) => {
                if (greetIndex === 0) {
                  setFirstMes(e.target.value)
                } else {
                  setAlternateGreetings((prev) =>
                    prev.map((g, j) =>
                      j === greetIndex - 1 ? e.target.value : g
                    )
                  )
                }
              }}
              rows={4}
              placeholder={t('characters:greetingPlaceholder')}
              {...spellProps}
            />
            <FieldHint>{t('characters:greetingHint')}</FieldHint>
          </div>

          {/* Example dialogue */}
          <div className="flex flex-col gap-2">
            <SectionLabel>{t('characters:exampleDialogue')}</SectionLabel>
            <Textarea
              value={mesExample}
              onChange={(e) => setMesExample(e.target.value)}
              rows={5}
              placeholder={'<START>\n{{user}}: ...\n{{char}}: ...'}
              {...spellProps}
            />
            <FieldHint>{t('characters:exampleDialogueHint')}</FieldHint>
          </div>

          {/* Advanced */}
          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="flex items-center gap-1.5 text-sm font-medium text-muted-foreground hover:text-foreground cursor-pointer self-start"
              >
                <IconChevronDown
                  size={16}
                  className={
                    advancedOpen ? 'rotate-180 transition-transform' : 'transition-transform'
                  }
                />
                {t('characters:advancedDefinitions')}
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="flex flex-col gap-5 mt-4">
              <div className="flex flex-col gap-2">
                <SectionLabel>{t('characters:systemPromptOverride')}</SectionLabel>
                <Textarea
                  value={systemPromptOverride}
                  onChange={(e) => setSystemPromptOverride(e.target.value)}
                  rows={4}
                  placeholder={t('characters:systemPromptOverrideHint')}
                  {...spellProps}
                />
              </div>
              <div className="flex flex-col gap-2">
                <SectionLabel>{t('characters:tags')}</SectionLabel>
                <Input
                  value={tagsInput}
                  onChange={(e) => setTagsInput(e.target.value)}
                  placeholder="fantasy, mentor, magic"
                />
              </div>
              <div className="flex flex-col gap-2">
                <SectionLabel>{t('characters:creatorNotes')}</SectionLabel>
                <Textarea
                  value={creatorNotes}
                  onChange={(e) => setCreatorNotes(e.target.value)}
                  rows={3}
                  {...spellProps}
                />
              </div>
            </CollapsibleContent>
          </Collapsible>

          {/* Sampling parameters */}
          <div className="flex flex-col gap-2 mt-2">
            <SectionLabel>{t('characters:parameters')}</SectionLabel>
            <ParametersSection
              params={params}
              providers={activeProviders}
              onToggle={(def: ParamDef) => {
                setParams((prev) => {
                  if (def.key in prev) {
                    const next = { ...prev }
                    delete next[def.key]
                    return next
                  }
                  return { ...prev, [def.key]: def.value }
                })
              }}
              onChange={(key: string, value: unknown) =>
                setParams((prev) => ({ ...prev, [key]: value }))
              }
              onRemove={(key: string) =>
                setParams((prev) => {
                  const next = { ...prev }
                  delete next[key]
                  return next
                })
              }
              onAddMany={(values: Record<string, unknown>) =>
                setParams((prev) => ({ ...prev, ...values }))
              }
              onRemoveMany={(keys: string[]) =>
                setParams((prev) => {
                  const next = { ...prev }
                  for (const k of keys) delete next[k]
                  return next
                })
              }
            />
          </div>
        </div>
      </div>

      <DeleteCharacterDialog
        open={deleteConfirmOpen}
        onOpenChange={setDeleteConfirmOpen}
        onConfirm={handleDelete}
      />
    </div>
  )
}

export default CharacterDetailContent

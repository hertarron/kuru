import { createFileRoute, useNavigate, useRouter } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { useEffect, useRef, useState } from 'react'

import HeaderPage from '@/containers/HeaderPage'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  IconChevronDown,
  IconChevronLeft,
  IconFileUpload,
  IconPencil,
  IconPlus,
  IconTrash,
  IconDownload,
} from '@tabler/icons-react'
import { ChevronsUpDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useLorebooks, createEmptyEntry } from '@/hooks/useLorebooks'
import {
  LorebookLogic,
  LorebookPosition,
  effectivePosition,
  formatKeys,
  normalizeLorebook,
  parseKeys,
  type Lorebook,
  type LorebookEntry,
} from '@/lib/lorebook'
import { exportLorebookFile } from '@/lib/lorebook-export'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.lorebooks.detail as any)({
  component: LorebookDetailContent,
})

const POSITION_LABELS: Record<number, string> = {
  [LorebookPosition.beforeChar]: 'Before character',
  [LorebookPosition.afterChar]: 'After character',
  [LorebookPosition.ANTop]: 'Top of author’s note',
  [LorebookPosition.ANBottom]: 'Bottom of author’s note',
  [LorebookPosition.atDepth]: 'At depth',
  [LorebookPosition.EMTop]: 'Before examples',
  [LorebookPosition.EMBottom]: 'After examples',
  [LorebookPosition.outlet]: 'Outlet',
}

const LOGIC_LABELS: Record<number, string> = {
  [LorebookLogic.AND_ANY]: 'any of',
  [LorebookLogic.AND_ALL]: 'all of',
  [LorebookLogic.NOT_ANY]: 'none of',
  [LorebookLogic.NOT_ALL]: 'not all of',
}

const ROLE_LABELS: Record<number, string> = {
  0: 'System',
  1: 'User',
  2: 'Assistant',
}

type Preset =
  | 'director'
  | 'character'
  | 'ambient'
  | 'world'
  | 'detail'
  | 'examples'
  | 'custom'

/**
 * The preset is derived, not stored: a book imported from anywhere else has no
 * idea what our presets are, and storing a fourth source of truth would let it
 * drift from the fields it claims to describe. Order is deliberately not part
 * of the match -- an entry the author re-ranked is still recognisably the
 * placement it started as.
 */
function presetOf(entry: LorebookEntry): Preset {
  if (entry.position === LorebookPosition.atDepth && entry.role === 0) {
    if (entry.depth === 0) return 'director'
    if (entry.depth === 1) return 'character'
    if (entry.depth === 4) return 'ambient'
    return 'custom'
  }
  if (entry.position === LorebookPosition.beforeChar) return 'world'
  if (entry.position === LorebookPosition.afterChar) return 'detail'
  if (
    entry.position === LorebookPosition.EMTop ||
    entry.position === LorebookPosition.EMBottom
  )
    return 'examples'
  return 'custom'
}

const atDepth = (
  entry: LorebookEntry,
  depth: number,
  order: number
): LorebookEntry => ({
  ...entry,
  position: LorebookPosition.atDepth,
  depth,
  role: 0,
  order,
})

function applyPreset(entry: LorebookEntry, preset: Preset): LorebookEntry {
  switch (preset) {
    case 'director':
      return atDepth(entry, 0, 300)
    case 'character':
      return atDepth(entry, 1, 200)
    case 'ambient':
      return atDepth(entry, 4, 100)
    case 'world':
      return { ...entry, position: LorebookPosition.beforeChar, order: 100 }
    case 'detail':
      return { ...entry, position: LorebookPosition.afterChar, order: 100 }
    case 'examples':
      return { ...entry, position: LorebookPosition.EMBottom, order: 100 }
    default:
      return entry
  }
}

const PRESET_ORDER: Preset[] = [
  'world',
  'detail',
  'examples',
  'ambient',
  'character',
  'director',
  'custom',
]

const PRESET_LABELS: Record<Preset, string> = {
  world: 'World',
  detail: 'Detail',
  examples: 'Examples',
  ambient: 'Ambient',
  character: 'Character',
  director: 'Director',
  custom: 'Custom',
}

const PRESET_HINTS: Record<Preset, string> = {
  world:
    'Before the character definition, as background the model reads first: places, factions, history.',
  detail:
    'After the character definition, for facts about the character that would clutter the description.',
  examples:
    'Beside the example dialogue, for style and formatting you want copied rather than facts.',
  ambient:
    'A few messages back in the chat: present, but not pressing on the next reply.',
  character:
    'Just before your newest message, where the model weighs it most — the usual choice for people, items and places that come up mid-scene.',
  director:
    'Appended to your newest message: the last thing the model reads before it answers. Strong, and overused it flattens the prose.',
  custom: 'Position and prominence set by hand below.',
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <label className="text-xs font-medium">{children}</label>
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  disabled,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  disabled?: boolean
}) {
  return (
    <div className={cn('flex flex-col gap-1', disabled && 'opacity-60')}>
      <FieldLabel>{label}</FieldLabel>
      <Input
        type="number"
        min={min}
        max={max}
        disabled={disabled}
        value={value}
        onChange={(e) => {
          const next = Number(e.target.value)
          if (Number.isFinite(next)) onChange(next)
        }}
        className="w-28"
      />
    </div>
  )
}

function Picker({
  label,
  value,
  options,
  onChange,
  className,
  disabled,
}: {
  label: string
  value: string
  options: { value: number; label: string }[]
  onChange: (value: number) => void
  className?: string
  disabled?: boolean
}) {
  return (
    <div className={cn('flex flex-col gap-1', disabled && 'opacity-60')}>
      <FieldLabel>{label}</FieldLabel>
      <DropdownMenu>
        <DropdownMenuTrigger asChild disabled={disabled}>
          <Button
            variant="outline"
            size="sm"
            disabled={disabled}
            className={cn('justify-between font-normal', className)}
          >
            <span className="truncate">{value}</span>
            <ChevronsUpDown className="size-4 shrink-0 text-muted-foreground ml-2" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {options.map((option) => (
            <DropdownMenuItem
              key={option.value}
              className="cursor-pointer"
              onClick={() => onChange(option.value)}
            >
              {option.label}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}


/** Beginner cheat sheet behind the "regex" link on the trigger field. */
const REGEX_EXAMPLES: [string, string][] = [
  ['/dragons?/', '? makes the character before it optional: dragon, dragons.'],
  [
    '/(sword|blade) of dawn/',
    '| picks between alternatives inside a longer phrase.',
  ],
  ['/wyrm(s|lings)?/', 'Brackets group what the | and ? apply to.'],
  ['/Rune/', 'A pattern is case-sensitive: Rune, but not rune.'],
  ['/orc/i', 'i ignores capitalisation, and matches inside orchard too.'],
  ['/room \\d+/', '\\d is any digit, + means one or more: room 7, room 214.'],
  [
    '/knight.*castle/',
    '. is any character, * means any number of them: knight rides to the castle.',
  ],
  ['/dr\\. holt/i', 'A backslash makes a special character literal: dr. holt.'],
]

function RegexHelp() {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="text-blue-600 dark:text-blue-400 cursor-pointer hover:text-blue-500"
        >
          regex
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[26rem] max-w-[90vw] max-h-96 overflow-y-auto p-3"
      >
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium">Regex triggers</span>
            <p className="text-xs text-muted-foreground">
              Several triggers already mean “any of these”. A pattern is for
              what a word list cannot say — and runs exactly as written, so add
              the <code>i</code> flag to ignore capitalisation.
            </p>
          </div>
          <div className="flex flex-col gap-2">
            {REGEX_EXAMPLES.map(([pattern, meaning]) => (
              <div key={pattern} className="flex flex-col gap-0.5">
                <code className="text-xs bg-secondary rounded px-1.5 py-0.5 self-start">
                  {pattern}
                </code>
                <span className="text-xs text-muted-foreground">{meaning}</span>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            A pattern that will not compile is kept as plain text, so a stray
            slash never breaks the entry.
          </p>
        </div>
      </PopoverContent>
    </Popover>
  )
}



type PreviewRow = { label: string; entry?: boolean; muted?: boolean }

/**
 * Where this entry's text lands in the finished prompt. The rows mirror
 * `buildCharacterPromptParts` and `applyDepthInjections`, so a placement that
 * changes here changes there.
 */
function placementRows(entry: LorebookEntry): PreviewRow[] {
  const { position } = effectivePosition(entry)
  const mine: PreviewRow = { label: 'This entry', entry: true }
  const rows: PreviewRow[] = []

  const depth = Math.max(0, Math.round(entry.depth))
  const isDepth = position === LorebookPosition.atDepth

  if (position === LorebookPosition.beforeChar) rows.push(mine)
  rows.push({ label: 'Character definition' })
  if (position === LorebookPosition.afterChar) rows.push(mine)
  if (position === LorebookPosition.EMTop) rows.push(mine)
  rows.push({ label: 'Example dialogue', muted: true })
  if (position === LorebookPosition.EMBottom) rows.push(mine)
  rows.push({ label: '⋯ earlier chat', muted: true })
  if (isDepth && depth >= 3) rows.push(mine)
  rows.push({ label: 'Reply before last', muted: true })
  if (isDepth && depth === 2) rows.push(mine)
  rows.push({ label: 'Character’s last reply', muted: true })
  if (isDepth && depth === 1) rows.push(mine)
  rows.push({ label: 'Your newest message' })
  if (isDepth && depth === 0) rows.push(mine)
  return rows
}

function PlacementPreview({ entry }: { entry: LorebookEntry }) {
  const rows = placementRows(entry)
  const depth = Math.max(0, Math.round(entry.depth))
  const deep =
    effectivePosition(entry).position === LorebookPosition.atDepth && depth >= 3

  return (
    <div className="flex flex-col gap-1 w-56 shrink-0 self-stretch">
      <span className="text-[11px] text-muted-foreground">
        Where it lands in the prompt
      </span>
      <div className="flex flex-col gap-0.5 rounded-md border p-2 flex-1">
        {rows.map((row, i) => (
          <div
            key={`${row.label}-${i}`}
            className={cn(
              'text-[10px] leading-tight rounded px-1.5 py-1 truncate flex-1 flex items-center min-h-5',
              row.entry
                ? 'bg-blue-500/15 text-blue-600 dark:text-blue-400 font-medium'
                : row.muted
                  ? 'bg-secondary/40 text-muted-foreground'
                  : 'bg-secondary text-muted-foreground'
            )}
          >
            {row.entry && deep ? `This entry (depth ${depth})` : row.label}
          </div>
        ))}
      </div>
      <span className="text-[10px] text-muted-foreground leading-snug">
        The top block is the system prompt; below it is the chat, oldest first.
      </span>
    </div>
  )
}

function EntryCard({
  entry,
  recursive,
  onChange,
  onDelete,
  defaultOpen,
}: {
  entry: LorebookEntry
  /** The book allows entries to trigger each other. */
  recursive: boolean
  onChange: (entry: LorebookEntry) => void
  onDelete: () => void
  defaultOpen: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  const [keysInput, setKeysInput] = useState(formatKeys(entry.keys))
  const [secondaryInput, setSecondaryInput] = useState(
    formatKeys(entry.secondaryKeys)
  )

  const derivedPreset = presetOf(entry)
  // Custom is not a state an entry can be in -- it is the absence of a preset
  // -- so the button only reveals the manual controls, leaving the fields
  // alone until they are actually edited.
  const [customOpen, setCustomOpen] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const showCustom = derivedPreset === 'custom' || customOpen
  const preset: Preset = showCustom ? 'custom' : derivedPreset
  const isAtDepth = entry.position === LorebookPosition.atDepth
  // ST's per-entry recursion opt-outs. They live in `extra` under ST's own
  // names, so they round-trip; the scanner already honours them.
  const flag = (name: string) => entry.extra[name] === true
  const setFlag = (name: string, value: boolean) =>
    onChange({ ...entry, extra: { ...entry.extra, [name]: value } })
  const { note } = effectivePosition(entry)
  const title = entry.comment || entry.keys[0] || 'Untitled entry'

  const commitKeys = (raw: string) => {
    const keys = parseKeys(raw)
    // Empty keys is the one input that means "always on"; `constant` exists
    // only to carry that through to imports and exports.
    onChange({ ...entry, keys, constant: keys.length === 0 })
  }

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="rounded-lg border bg-secondary/20"
    >
      <div className="flex items-center gap-2 px-3 py-2.5">
        <CollapsibleTrigger asChild>
          <button
            type="button"
            title={open ? 'Collapse' : 'Expand'}
            className="flex items-center shrink-0"
          >
            <IconChevronDown
              className={cn(
                'size-4 text-muted-foreground transition-transform',
                !open && '-rotate-90'
              )}
            />
          </button>
        </CollapsibleTrigger>
        <Button
          variant="ghost"
          size="icon-xs"
          title="Rename entry"
          onClick={() => setRenaming(true)}
        >
          <IconPencil className="size-3.5 text-muted-foreground" />
        </Button>
        {renaming ? (
          <Input
            autoFocus
            value={entry.comment}
            placeholder={entry.keys[0] ?? 'Untitled entry'}
            className="h-7 flex-1 min-w-0"
            onChange={(e) => onChange({ ...entry, comment: e.target.value })}
            onBlur={() => setRenaming(false)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === 'Escape') setRenaming(false)
            }}
          />
        ) : (
          <button
            type="button"
            className="flex items-center gap-2 min-w-0 flex-1 text-left cursor-pointer"
            onClick={() => setOpen(!open)}
          >
            <span className="text-sm font-medium truncate">{title}</span>
            {entry.keys.length === 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-500/15 text-blue-600 dark:text-blue-400 leading-none shrink-0">
                always on
              </span>
            )}
            <span className="text-[11px] text-muted-foreground shrink-0 ml-auto">
              {PRESET_LABELS[preset]}
            </span>
          </button>
        )}
        <Switch
          checked={entry.enabled}
          onCheckedChange={(enabled) => onChange({ ...entry, enabled })}
          title={entry.enabled ? 'Enabled' : 'Disabled'}
        />
        <Button variant="ghost" size="icon-xs" title="Delete entry" onClick={onDelete}>
          <IconTrash className="size-4 text-destructive" />
        </Button>
      </div>

      <CollapsibleContent className="px-3 pb-3 flex flex-col gap-3">
        <div className="flex flex-col gap-1">
          <FieldLabel>Triggers</FieldLabel>
          <Input
            value={keysInput}
            placeholder="dragon, old kingdom, /wyrm(s)?/i"
            onChange={(e) => setKeysInput(e.target.value)}
            onBlur={(e) => commitKeys(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            Comma separated, whole words, capitalisation ignored. Wrap in
            slashes for a <RegexHelp />. Leave empty to keep this entry always
            on.
          </p>
        </div>

        <div className="flex flex-col gap-1">
          <FieldLabel>Content</FieldLabel>
          <Textarea
            value={entry.content}
            placeholder="What the model should know when this entry fires."
            className="min-h-28"
            onChange={(e) => onChange({ ...entry, content: e.target.value })}
          />
        </div>

        <div className="flex flex-col gap-1">
          <FieldLabel>Placement</FieldLabel>
          <div className="flex flex-wrap items-stretch gap-4">
            <div className="flex flex-col gap-2 flex-1 min-w-72">
              <div className="flex flex-wrap gap-2">
                {PRESET_ORDER.map((option) => (
                  <Button
                    key={option}
                    size="sm"
                    variant={preset === option ? 'default' : 'outline'}
                    className={cn(
                      // Selected/unselected must be the same size, or the row
                      // jumps as the border comes and goes.
                      'transition-none',
                      preset === option && 'border border-transparent'
                    )}
                    onClick={() => {
                      setCustomOpen(option === 'custom')
                      if (option !== 'custom') {
                        onChange(applyPreset(entry, option))
                      }
                    }}
                  >
                    {PRESET_LABELS[option]}
                  </Button>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {PRESET_HINTS[preset]}
              </p>
              {note && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  {note}
                </p>
              )}

              {/* Always rendered, so the numbers behind each preset stay
                  visible; only Custom unlocks them. */}
              <div className="flex flex-col gap-2 rounded-md border border-dashed p-3">
                <div className="flex flex-wrap gap-3">
                  <Picker
                    label="Position"
                    className="w-44"
                    disabled={!showCustom}
                    value={POSITION_LABELS[entry.position] ?? 'Unknown'}
                    options={Object.entries(POSITION_LABELS).map(
                      ([v, label]) => ({ value: Number(v), label })
                    )}
                    onChange={(position) => onChange({ ...entry, position })}
                  />
                  <NumberField
                    label="Depth"
                    min={0}
                    disabled={!showCustom || !isAtDepth}
                    value={entry.depth}
                    onChange={(depth) => onChange({ ...entry, depth })}
                  />
                  <Picker
                    label="Speaker"
                    className="w-28"
                    disabled={!showCustom || !isAtDepth}
                    value={ROLE_LABELS[entry.role ?? 0] ?? 'System'}
                    options={Object.entries(ROLE_LABELS).map(([v, label]) => ({
                      value: Number(v),
                      label,
                    }))}
                    onChange={(role) => onChange({ ...entry, role })}
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  Depth counts messages back from the newest: 0 appends to it, 1
                  sits just before it.
                </p>
                <div className="flex flex-wrap gap-3">
                  <NumberField
                    label="Prominence"
                    disabled={!showCustom}
                    value={entry.order}
                    onChange={(order) => onChange({ ...entry, order })}
                  />
                  <NumberField
                    label="Probability %"
                    min={0}
                    max={100}
                    disabled={!showCustom}
                    value={entry.probability}
                    onChange={(probability) =>
                      onChange({ ...entry, probability })
                    }
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  Prominence ranks entries sharing a slot: higher goes in first,
                  cut last. Probability is the chance a matched entry fires.
                </p>
              </div>
            </div>
            <PlacementPreview entry={entry} />
          </div>
        </div>

        {/* Everything a first entry does not need: a beginner should never
            meet selective logic on the way to writing one. */}
        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground cursor-pointer"
            >
              <IconChevronDown
                className={cn(
                  'size-4 transition-transform',
                  !advancedOpen && '-rotate-90'
                )}
              />
              Advanced
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent className="flex flex-col gap-3 pt-3">
            <div className="flex flex-col gap-1">
              <FieldLabel>Only fire when the chat also has</FieldLabel>
              <div className="flex flex-wrap items-end gap-2">
                <Picker
                  label=""
                  className="w-40"
                  value={LOGIC_LABELS[entry.selectiveLogic] ?? 'any of'}
                  options={Object.entries(LOGIC_LABELS).map(([v, label]) => ({
                    value: Number(v),
                    label,
                  }))}
                  onChange={(selectiveLogic) =>
                    onChange({ ...entry, selectiveLogic })
                  }
                />
                <Input
                  value={secondaryInput}
                  placeholder="Second set of keys, comma separated"
                  className="flex-1 min-w-52"
                  onChange={(e) => setSecondaryInput(e.target.value)}
                  onBlur={(e) =>
                    onChange({
                      ...entry,
                      secondaryKeys: parseKeys(e.target.value),
                    })
                  }
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Triggers are an or-list: any one of them fires the entry. These
                narrow that down — a second condition the chat has to meet as
                well, or one it must not. Leave empty and the triggers decide
                alone.
              </p>
            </div>

            {/* Greyed rather than hidden, so the book-level switch explains
                itself from here. */}
            <div
              className={cn(
                'flex flex-wrap gap-x-6 gap-y-2',
                !recursive && 'opacity-60'
              )}
              title={
                recursive
                  ? undefined
                  : 'Turn on “Entries can trigger each other” for this book to use these.'
              }
            >
              <label className="flex items-center gap-2 text-xs">
                <Switch
                  disabled={!recursive}
                  checked={!flag('preventRecursion')}
                  onCheckedChange={(on) => setFlag('preventRecursion', !on)}
                />
                Other entries can trigger this one
              </label>
              <label className="flex items-center gap-2 text-xs">
                <Switch
                  disabled={!recursive}
                  checked={!flag('excludeRecursion')}
                  onCheckedChange={(on) => setFlag('excludeRecursion', !on)}
                />
                This content can trigger others
              </label>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </CollapsibleContent>
    </Collapsible>
  )
}

function LorebookDetailContent() {
  const { lorebookId } = Route.useParams()
  const navigate = useNavigate()
  const router = useRouter()
  const lorebooks = useLorebooks((s) => s.lorebooks)
  const loading = useLorebooks((s) => s.loading)
  const updateLorebook = useLorebooks((s) => s.updateLorebook)

  const stored = lorebooks.find((b) => b.id === lorebookId)
  const [draft, setDraft] = useState<Lorebook | undefined>(stored)
  const [savedAt, setSavedAt] = useState<number | null>(null)
  const [newEntryId, setNewEntryId] = useState<string | null>(null)
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [importError, setImportError] = useState<string | null>(null)

  // The store finishes loading after this route mounts on a cold start.
  useEffect(() => {
    if (!draft && stored) setDraft(stored)
  }, [draft, stored])

  useEffect(
    () => () => {
      if (saveTimer.current) clearTimeout(saveTimer.current)
    },
    []
  )

  const edit = (next: Lorebook) => {
    setDraft(next)
    if (saveTimer.current) clearTimeout(saveTimer.current)
    saveTimer.current = setTimeout(() => {
      updateLorebook(next)
      setSavedAt(Date.now())
    }, 500)
  }

  useEffect(() => {
    if (savedAt === null) return
    const timer = setTimeout(() => setSavedAt(null), 1500)
    return () => clearTimeout(timer)
  }, [savedAt])

  if (!draft) {
    return (
      <div className="flex flex-col h-full w-full items-center justify-center">
        <p className="text-sm text-muted-foreground">
          {loading ? 'Loading…' : 'This lorebook no longer exists.'}
        </p>
      </div>
    )
  }

  /**
   * Reads a world-info export into this book. Entries replace what is here,
   * but the book keeps its own id and its always-active flag: the file is a
   * source of entries, not a replacement identity.
   */
  const handleImportFile = async (file: File | undefined) => {
    if (!file || !draft) return
    setImportError(null)
    try {
      const raw = JSON.parse(await file.text())
      const imported = normalizeLorebook(raw, {
        id: draft.id,
        name: file.name.replace(/\.json$/i, ''),
      })
      if (!imported) throw new Error('No lorebook entries found in that file.')
      edit({
        ...imported,
        id: draft.id,
        name: draft.name.trim() ? draft.name : imported.name,
        description: draft.description.trim()
          ? draft.description
          : imported.description,
        global: draft.global,
      })
    } catch (e) {
      setImportError(e instanceof Error ? e.message : String(e))
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  // Back means "where I came from": the editor is opened from the library,
  // from the chat pill and from the Hub, each expecting its own page back.
  const goBack = () => {
    if (router.history.canGoBack()) router.history.back()
    else navigate({ to: route.settings.lorebooks })
  }

  const addEntry = () => {
    const entry = createEmptyEntry()
    setNewEntryId(entry.id)
    edit({ ...draft, entries: [...draft.entries, entry] })
  }

  return (
    <div className="flex flex-col h-full w-full">
      <HeaderPage>
        {/* Wrapper ignores clicks so the root drag-strip stays grabbable in the
            blank middle; the title cluster opts back in. */}
        <div
          className={cn(
            'pointer-events-none flex items-center w-full pr-3 h-8 relative z-20',
            !IS_MACOS && 'pr-30'
          )}
        >
          <div className="pointer-events-auto flex items-center gap-2 min-w-0">
            <button
              type="button"
              className="cursor-pointer"
              onClick={goBack}
              aria-label="Back to lorebooks"
            >
              <IconChevronLeft size={20} className="text-muted-foreground" />
            </button>
            <span className="font-medium text-base font-studio truncate">
              {draft.name || 'Untitled lorebook'}
            </span>
            <span
              className={cn(
                'text-xs text-muted-foreground transition-opacity ml-2',
                savedAt ? 'opacity-100' : 'opacity-0'
              )}
            >
              ✓ Saved
            </span>
          </div>
          <div className="pointer-events-auto ml-auto flex items-center">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={(e) => handleImportFile(e.target.files?.[0])}
            />
            <Button
              size="sm"
              variant="link"
              title="Replace this book's entries and scan settings from a .json file"
              onClick={() => fileInputRef.current?.click()}
            >
              <IconFileUpload className="size-4" />
              Import
            </Button>
            <Button
              size="sm"
              variant="link"
              title="Save this book as a .json file other apps can import"
              onClick={() => exportLorebookFile(draft)}
            >
              <IconDownload className="size-4" />
              Export
            </Button>
          </div>
        </div>
      </HeaderPage>

      <div className="flex flex-col gap-4 p-4 w-full overflow-y-auto">
        <div className="flex flex-col gap-3 max-w-3xl w-full mx-auto">
          {importError && (
            <p className="text-sm text-destructive">{importError}</p>
          )}
          <div className="flex flex-col gap-1">
            <FieldLabel>Name</FieldLabel>
            <Input
              value={draft.name}
              onChange={(e) => edit({ ...draft, name: e.target.value })}
            />
          </div>
          <div className="flex flex-col gap-1">
            <FieldLabel>Description</FieldLabel>
            <Input
              value={draft.description}
              placeholder="What this book covers. Only for you."
              onChange={(e) => edit({ ...draft, description: e.target.value })}
            />
          </div>
          <div className="flex flex-wrap gap-x-6 gap-y-2">
            <label className="flex items-center gap-2 text-xs">
              <Switch
                checked={draft.global}
                onCheckedChange={(global) => edit({ ...draft, global })}
              />
              Always active in every chat
            </label>
            <label
              className="flex items-center gap-2 text-xs"
              title="An entry that fires can trigger other entries in this book, up to the step limit in settings."
            >
              <Switch
                checked={draft.recursiveScanning}
                onCheckedChange={(recursiveScanning) =>
                  edit({ ...draft, recursiveScanning })
                }
              />
              Entries can trigger each other
            </label>
          </div>

          <div className="flex items-center justify-between mt-4">
            <h2 className="text-sm font-studio font-medium">
              {draft.entries.length}{' '}
              {draft.entries.length === 1 ? 'entry' : 'entries'}
            </h2>
            <Button size="sm" variant="outline" onClick={addEntry}>
              <IconPlus className="size-4" />
              Add entry
            </Button>
          </div>
          <div className="flex flex-col gap-2 pb-8">
            {draft.entries.map((entry, index) => (
              <EntryCard
                key={entry.id}
                entry={entry}
                recursive={draft.recursiveScanning}
                defaultOpen={entry.id === newEntryId}
                onChange={(next) =>
                  edit({
                    ...draft,
                    entries: draft.entries.map((e, i) =>
                      i === index ? next : e
                    ),
                  })
                }
                onDelete={() =>
                  edit({
                    ...draft,
                    entries: draft.entries.filter((_, i) => i !== index),
                  })
                }
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default LorebookDetailContent

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from '@tanstack/react-router'
import { route } from '@/constants/routes'

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import {
  IconArrowLeft,
  IconChevronLeft,
  IconChevronRight,
  IconPencil,
} from '@tabler/icons-react'
import { AvatarEmoji } from '@/containers/AvatarEmoji'
import { cn } from '@/lib/utils'
import {
  chubCharacterPathFromUrl,
  fetchChubCharacterNode,
  loadChubCharacter,
  type ChubCharacterResult,
} from '@/lib/chub'
import { LorebookPosition, type Lorebook } from '@/lib/lorebook'
import { useLorebooks } from '@/hooks/useLorebooks'

/**
 * One preview surface for both halves of the Characters hub: a Chub search
 * result (fetched on open) and a card already in the library. Same sections,
 * same order, so what you inspect before importing is what you get after.
 */
export type CharacterPreviewSource =
  | { kind: 'chub'; node: ChubCharacterResult }
  | { kind: 'library'; character: Character }

interface CharacterPreviewSheetProps {
  source: CharacterPreviewSource | null
  onOpenChange: (open: boolean) => void
  /** Chub only. Absent when the card is already in the library. */
  onImport?: (node: ChubCharacterResult) => void
  /** Keyed by fullPath, since following a link can change which card is shown. */
  importState?: (fullPath: string) => 'idle' | 'importing' | 'imported'
}

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

function Section({
  label,
  children,
  count,
}: {
  label: string
  children: React.ReactNode
  count?: string
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        {count && (
          <span className="text-[11px] text-muted-foreground">{count}</span>
        )}
      </div>
      {children}
    </div>
  )
}

const URL_PATTERN = /https?:\/\/[^\s<>"')\]]+/g

/**
 * Bare URLs in card text, made clickable. Chub character links stay inside
 * the app -- creator notes routinely point at a sequel or a fork, and the
 * useful thing there is that card's preview, not a browser tab.
 */
function LinkedText({
  text,
  onOpenChub,
}: {
  text: string
  onOpenChub?: (fullPath: string) => void
}) {
  const parts: React.ReactNode[] = []
  let last = 0
  for (const match of text.matchAll(URL_PATTERN)) {
    const url = match[0]
    const start = match.index
    if (start > last) parts.push(text.slice(last, start))
    last = start + url.length

    const chubPath = onOpenChub ? chubCharacterPathFromUrl(url) : null
    parts.push(
      chubPath ? (
        <button
          key={start}
          onClick={() => onOpenChub?.(chubPath)}
          className="text-left underline underline-offset-2 decoration-dotted text-foreground hover:text-foreground/70 cursor-pointer break-all"
        >
          {url}
        </button>
      ) : (
        <a
          key={start}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="underline underline-offset-2 decoration-dotted hover:text-foreground break-all"
        >
          {url}
        </a>
      )
    )
  }
  if (last < text.length) parts.push(text.slice(last))
  return <>{parts}</>
}

/** Boxed field height before it starts scrolling, in px (11rem). */
const BODY_CAP_PX = 176

/**
 * Card fields run from one line to several thousand words. Short ones size to
 * their content; long ones cap and scroll, with a bottom-right grip to drag
 * them taller.
 *
 * The cap has to be a real `height` rather than `max-height`, because
 * max-height clamps the inline height the resize grip writes and the box
 * could never be dragged past it. That means only capping boxes that
 * actually need it, which means measuring.
 *
 * `plain` drops the box for fields that read as prose about the card rather
 * than content of it, and lets them flow at their natural length.
 */
function Body({
  text,
  className,
  plain,
  linkify,
  onOpenChub,
}: {
  text: string
  className?: string
  plain?: boolean
  /** Turn bare URLs into links; chub character links navigate in place. */
  linkify?: boolean
  onOpenChub?: (fullPath: string) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [capped, setCapped] = useState(false)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el || plain) return
    // Drop any height the grip wrote for the previous text (the greeting
    // pager swaps text in place) before deciding whether this one overflows.
    el.style.height = ''
    setCapped(el.scrollHeight > BODY_CAP_PX)
  }, [text, plain])

  return (
    <div
      ref={ref}
      className={cn(
        !plain && 'rounded-md border bg-secondary/20 p-3',
        capped && 'h-44 min-h-40 overflow-auto resize-y',
        className
      )}
    >
      <p className="text-sm whitespace-pre-wrap leading-relaxed text-foreground/90">
        {linkify ? <LinkedText text={text} onOpenChub={onOpenChub} /> : text}
      </p>
    </div>
  )
}

/** Greeting plus alternates, in the one-box-and-arrows pattern. */
function GreetingPager({ greetings }: { greetings: string[] }) {
  const [index, setIndex] = useState(0)
  const safeIndex = Math.min(index, greetings.length - 1)

  return (
    <Section
      label="Greeting"
      count={
        greetings.length > 1
          ? `${safeIndex + 1}/${greetings.length}`
          : undefined
      }
    >
      <Body text={greetings[safeIndex]} />
      {greetings.length > 1 && (
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setIndex((i) => Math.max(0, i - 1))}
            disabled={safeIndex === 0}
          >
            <IconChevronLeft size={14} />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() =>
              setIndex((i) => Math.min(greetings.length - 1, i + 1))
            }
            disabled={safeIndex === greetings.length - 1}
          >
            <IconChevronRight size={14} />
          </Button>
        </div>
      )}
    </Section>
  )
}

function WorldInfoSection({ book }: { book: Lorebook }) {
  const [expanded, setExpanded] = useState(false)
  const shown = expanded ? book.entries : book.entries.slice(0, 3)

  return (
    <Section
      label="World info"
      count={`${book.entries.length} ${book.entries.length === 1 ? 'entry' : 'entries'}`}
    >
      <div className="flex flex-col gap-2">
        {shown.map((entry) => (
          <div
            key={entry.id}
            className="rounded-md border p-2.5 flex flex-col gap-1.5"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium truncate">
                {entry.comment || entry.keys[0] || 'Untitled entry'}
              </span>
              <span className="text-[11px] text-muted-foreground shrink-0">
                {POSITION_LABELS[entry.position] ?? 'Unknown position'}
                {entry.position === LorebookPosition.atDepth &&
                  ` ${entry.depth}`}
              </span>
            </div>
            <div className="flex flex-wrap gap-1">
              {entry.constant ? (
                <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-blue-500/15 text-blue-600 dark:text-blue-400">
                  always on
                </span>
              ) : (
                entry.keys.slice(0, 5).map((key) => (
                  <span
                    key={key}
                    className="text-[11px] px-1.5 py-0.5 rounded-full bg-secondary text-muted-foreground truncate max-w-40"
                  >
                    {key}
                  </span>
                ))
              )}
              {!entry.constant && entry.keys.length > 5 && (
                <span className="text-[11px] text-muted-foreground self-center">
                  +{entry.keys.length - 5}
                </span>
              )}
            </div>
            <p className="text-xs text-muted-foreground line-clamp-2 whitespace-pre-wrap">
              {entry.content}
            </p>
          </div>
        ))}
        {book.entries.length > 3 && (
          <Button
            variant="ghost"
            size="sm"
            className="self-start"
            onClick={() => setExpanded((e) => !e)}
          >
            {expanded ? 'Show less' : `Show all ${book.entries.length} entries`}
          </Button>
        )}
      </div>
    </Section>
  )
}

/** Whitespace- and case-insensitive containment, for comparing two blurbs. */
const contains = (haystack: string | undefined, needle: string | undefined) => {
  const h = haystack?.replace(/\s+/g, ' ').trim().toLowerCase()
  const n = needle?.replace(/\s+/g, ' ').trim().toLowerCase()
  return !!h && !!n && h.includes(n)
}

/**
 * A card carries creator notes baked into the PNG; the Chub listing carries a
 * page blurb in `description`. They overlap but are not the same field — one
 * sampled card's baked notes were cut off mid-word at 213 chars while the page
 * blurb ran to 508, and another pair shared no text at all. So show whichever
 * of the two the other does not already contain, rather than picking one and
 * dropping whatever the reader would have learnt from the other.
 */
function distinctNotes(cardNotes?: string, chubNotes?: string) {
  const card = cardNotes?.trim() || undefined
  const chub = chubNotes?.trim() || undefined
  if (!card) return { cardNotes: undefined, chubNotes: chub }
  if (!chub) return { cardNotes: card, chubNotes: undefined }
  if (contains(chub, card)) return { cardNotes: undefined, chubNotes: chub }
  if (contains(card, chub)) return { cardNotes: card, chubNotes: undefined }
  return { cardNotes: card, chubNotes: chub }
}

export function CharacterPreviewSheet({
  source,
  onOpenChange,
  onImport,
  importState,
}: CharacterPreviewSheetProps) {
  const navigate = useNavigate()
  const [character, setCharacter] = useState<Character | null>(null)
  const [lorebook, setLorebook] = useState<Lorebook | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  // Chub cards reached by following a link out of another card's notes.
  // Kept as a stack so Back walks home instead of closing the sheet.
  const [linkStack, setLinkStack] = useState<ChubCharacterResult[]>([])
  const [imageViewerOpen, setImageViewerOpen] = useState(false)

  // Parents build `source` inline, so its identity changes every render.
  // Everything downstream keys off the card it names instead, or the sheet
  // would reset its link stack and refetch on every render.
  const sourceKey = source
    ? source.kind === 'chub'
      ? `chub:${source.node.fullPath}`
      : `library:${source.character.id}`
    : null
  const sourceRef = useRef(source)
  sourceRef.current = source

  useEffect(() => {
    setLinkStack([])
  }, [sourceKey])

  const linked = linkStack[linkStack.length - 1]
  const active = useMemo<CharacterPreviewSource | null>(
    () => (linked ? { kind: 'chub', node: linked } : sourceRef.current),
    // sourceRef always holds the current source; sourceKey says when it changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [linked, sourceKey]
  )

  const openChubPath = (fullPath: string) => {
    setLoading(true)
    setError(null)
    fetchChubCharacterNode(fullPath)
      .then((node) => setLinkStack((prev) => [...prev, node]))
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })
  }

  useEffect(() => {
    if (!active) return
    const source = active
    if (source.kind === 'library') {
      setCharacter(source.character)
      // A library character references books rather than embedding one; the
      // preview shows the first attached book, the same slot a card's embedded
      // book occupies.
      const attachedId = source.character.lorebookIds?.[0]
      setLorebook(
        attachedId
          ? (useLorebooks
              .getState()
              .lorebooks.find((b) => b.id === attachedId) ?? null)
          : null
      )
      setError(null)
      setLoading(false)
      return
    }

    let cancelled = false
    setCharacter(null)
    setLorebook(null)
    setError(null)
    setLoading(true)
    loadChubCharacter(source.node)
      .then(({ character: c, lorebook: book }) => {
        if (cancelled) return
        setCharacter(c)
        setLorebook(book)
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [active])

  const node = active?.kind === 'chub' ? active.node : null
  const title = character?.name ?? node?.name ?? ''
  const greetings = [
    character?.first_mes,
    ...(character?.alternate_greetings ?? []),
  ].filter((g): g is string => !!g?.trim())

  const { cardNotes, chubNotes } = distinctNotes(
    character?.creator_notes,
    node?.description
  )
  // The tagline is the one-liner from the listing. It is often just the head of
  // the blurb, so only show it when it says something the notes do not.
  const tagline =
    node?.tagline?.trim() &&
    !contains(cardNotes, node.tagline) &&
    !contains(chubNotes, node.tagline)
      ? node.tagline.trim()
      : undefined

  const state = node ? (importState?.(node.fullPath) ?? 'idle') : 'idle'

  // Real images only -- an emoji avatar has nothing worth a full view.
  const imageUrl =
    typeof character?.avatar === 'string' &&
    (character.avatar.startsWith('data:image/') ||
      character.avatar.startsWith('/images/'))
      ? character.avatar
      : node?.avatar_url

  const stats = [
    node?.starCount != null ? `★ ${node.starCount}` : undefined,
    node?.nTokens != null ? `${node.nTokens} tokens` : undefined,
    node?.nMessages != null ? `${node.nMessages} chats` : undefined,
    character?.creator ? `by ${character.creator}` : undefined,
    character?.character_version
      ? `v${character.character_version}`
      : undefined,
  ].filter(Boolean)

  return (
    <Sheet
      open={!!source}
      onOpenChange={(open) => {
        if (!open) onOpenChange(false)
      }}
    >
      <SheetContent className="sm:max-w-xl flex flex-col p-0">
        <SheetHeader className="px-4">
          <div className="flex items-start gap-3 pr-6">
            {linkStack.length > 0 && (
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Back"
                className="shrink-0 -ml-1 self-center"
                onClick={() => setLinkStack((prev) => prev.slice(0, -1))}
              >
                <IconArrowLeft size={16} />
              </Button>
            )}
            {/* Fixed height, natural width: portrait cards stay slim, squarer
                art grows sideways instead of being cropped. Click for the
                full-resolution view. */}
            {imageUrl ? (
              <button
                type="button"
                className="shrink-0 cursor-zoom-in"
                onClick={() => setImageViewerOpen(true)}
                aria-label="View full image"
              >
                <img
                  src={imageUrl}
                  alt=""
                  className="h-24 w-auto max-w-40 object-cover rounded-lg border bg-secondary dark:bg-secondary/40"
                />
              </button>
            ) : (
              <div className="h-24 w-16 shrink-0 flex items-center justify-center bg-secondary dark:bg-secondary/40 rounded-lg overflow-hidden border">
                <AvatarEmoji
                  avatar={character?.avatar}
                  textClassName="text-2xl"
                />
              </div>
            )}
            <div className="min-w-0 flex-1 flex flex-col gap-1">
              <SheetTitle className="truncate">{title}</SheetTitle>
              <SheetDescription className="truncate">
                {loading
                  ? 'Loading card…'
                  : stats.length
                    ? stats.join(' · ')
                    : 'Character card'}
              </SheetDescription>
              {!!character?.tags?.length && (
                <div className="flex flex-wrap gap-1 mt-0.5">
                  {character.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[11px] px-1.5 py-0.5 rounded-full bg-foreground/8 text-muted-foreground"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto px-4 pb-4 flex flex-col gap-5">
          {error && <p className="text-sm text-destructive">{error}</p>}

          {loading && (
            <div className="flex flex-col gap-3 animate-pulse pt-2">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="flex flex-col gap-2">
                  <div className="h-3 bg-muted rounded w-24" />
                  <div className="h-16 bg-muted rounded w-full" />
                </div>
              ))}
            </div>
          )}

          {character && (
            <>
              {tagline && (
                <p className="text-sm text-muted-foreground italic">
                  {tagline}
                </p>
              )}

              {cardNotes && (
                <Section label="Creator notes">
                  <Body
                    text={cardNotes}
                    plain
                    linkify
                    onOpenChub={openChubPath}
                  />
                </Section>
              )}

              {chubNotes && (
                <Section
                  label={cardNotes ? 'From the Chub page' : 'Creator notes'}
                >
                  <Body
                    text={chubNotes}
                    plain
                    linkify
                    onOpenChub={openChubPath}
                  />
                </Section>
              )}

              {lorebook && <WorldInfoSection book={lorebook} />}

              {greetings.length > 0 && <GreetingPager greetings={greetings} />}

              {character.description?.trim() && (
                <Section label="Description">
                  <Body text={character.description} />
                </Section>
              )}
              {character.personality?.trim() && (
                <Section label="Personality">
                  <Body text={character.personality} />
                </Section>
              )}
              {character.scenario?.trim() && (
                <Section label="Scenario">
                  <Body text={character.scenario} />
                </Section>
              )}
              {character.system_prompt?.trim() && (
                <Section label="System prompt">
                  <Body text={character.system_prompt} />
                </Section>
              )}
              {character.post_history_instructions?.trim() && (
                <Section label="Post-history instructions">
                  <Body text={character.post_history_instructions} />
                </Section>
              )}
              {character.mes_example?.trim() && (
                <Section label="Example dialogue">
                  <Body text={character.mes_example} />
                </Section>
              )}
            </>
          )}
        </div>

        <div className="border-t p-4 flex items-center justify-end gap-2">
          {active?.kind === 'library' && character && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                onOpenChange(false)
                navigate({
                  to: route.characters.detail,
                  params: { characterId: character.id },
                })
              }}
            >
              <IconPencil size={16} />
              Edit
            </Button>
          )}
          {node && onImport && (
            <Button
              size="sm"
              disabled={state !== 'idle' || loading || !!error}
              onClick={() => onImport(node)}
              className={cn(state === 'imported' && 'pointer-events-none')}
            >
              {state === 'imported'
                ? '✓ In library'
                : state === 'importing'
                  ? 'Downloading...'
                  : 'Download'}
            </Button>
          )}
        </div>

        {/* Full view: a plain zoom popup that hugs the image -- no card
            chrome. Outside click and clicking the image both dismiss. */}
        <Dialog
          open={imageViewerOpen && !!imageUrl}
          onOpenChange={(open) => !open && setImageViewerOpen(false)}
        >
          <DialogContent
            showCloseButton={false}
            className="w-auto max-w-[90vw] max-h-[90vh] p-0 gap-0 overflow-hidden bg-transparent border-0 shadow-none"
          >
            <DialogTitle className="sr-only">{title || 'Image'}</DialogTitle>
            {imageUrl && (
              <img
                src={imageUrl}
                alt=""
                onClick={() => setImageViewerOpen(false)}
                className="block max-w-full max-h-[85vh] object-contain rounded-lg cursor-zoom-out"
              />
            )}
          </DialogContent>
        </Dialog>
      </SheetContent>
    </Sheet>
  )
}

export default CharacterPreviewSheet

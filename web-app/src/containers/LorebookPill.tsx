import { useCallback, useMemo } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { IconBook, IconSettings, IconWorld } from '@tabler/icons-react'
import { useNavigate } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { cn } from '@/lib/utils'
import { PILL_CHEVRON } from '@/constants/layout'
import { usePillLabel } from '@/hooks/usePillLabel'
import { ChevronsUpDown } from 'lucide-react'
import { defaultCharacter, useCharacters } from '@/hooks/useCharacters'
import { useLorebooks } from '@/hooks/useLorebooks'
import { useThreads } from '@/hooks/useThreads'
import { usePendingLorebooks } from '@/stores/pending-lorebooks'
import {
  resolveLorebooks,
  type LorebookOrigin,
  type ThreadLorebookState,
} from '@/lib/lorebook-selection'
import { usePopoverAlign } from '@/hooks/usePopoverAlign'

const ORIGIN_LABELS: Record<LorebookOrigin, string> = {
  character: 'from the character',
  global: 'always active',
  chat: 'added to this chat',
}

/**
 * The world-info switcher pill, beside the model and character pills. Shows how
 * many books will feed the next prompt and lets a chat add, mute, or silence
 * all of them without touching the library.
 */
export function LorebookPill({
  thread,
  compact = false,
}: {
  thread?: Thread
  /** Toolbar-sized pill for the chatbox row rather than the header. */
  compact?: boolean
}) {
  const navigate = useNavigate()
  // Panel is w-80; measured at open so it hugs the pill's free side.
  const { pillRef, align, measure } = usePopoverAlign(320)
  const library = useLorebooks((s) => s.lorebooks)
  const currentCharacter = useCharacters((s) => s.currentCharacter)
  const updateThread = useThreads((s) => s.updateThread)

  const embedded = thread?.assistants?.[0] as Character | undefined
  const isThreadEmbedded = !!embedded && embedded.id !== 'model-only'
  // Attachments live on the library character, so a thread's embedded copy is
  // only a lookup key — editing a character must reach chats already open.
  const characters = useCharacters((s) => s.characters)
  const activeCharacter = thread
    ? isThreadEmbedded
      ? (characters.find((c) => c.id === embedded?.id) ?? embedded)
      : undefined
    : (currentCharacter ?? defaultCharacter)

  // On the home screen the picks are staged until a thread exists.
  const pending = usePendingLorebooks((s) => s.pending)
  const patchPending = usePendingLorebooks((s) => s.patch)

  const threadState: ThreadLorebookState = useMemo(
    () => (thread ? (thread.metadata?.lorebooks ?? {}) : pending),
    [thread, pending]
  )

  const resolved = useMemo(
    () =>
      resolveLorebooks({
        library,
        characterLorebookIds: activeCharacter?.lorebookIds,
        threadState,
      }),
    [library, activeCharacter?.lorebookIds, threadState]
  )

  const inPlayIds = new Set(resolved.map((r) => r.book.id))
  const available = library.filter((b) => !inPlayIds.has(b.id))
  const activeCount = resolved.filter((r) => r.enabled).length

  // The cog edits the book this chat is actually running on; with several (or
  // none) in play there is no single answer, so it greys out.
  const cogBook =
    resolved.length === 1 && !threadState.off ? resolved[0].book : undefined

  const patchThread = (patch: ThreadLorebookState) => {
    if (!thread) {
      patchPending(patch)
      return
    }
    updateThread(thread.id, {
      metadata: {
        ...thread.metadata,
        lorebooks: { ...threadState, ...patch },
      },
    })
  }

  const toggleEnabled = (id: string, enabled: boolean) => {
    const disabled = new Set(threadState.disabledIds ?? [])
    if (enabled) disabled.delete(id)
    else disabled.add(id)
    patchThread({ disabledIds: [...disabled] })
  }

  const attachToChat = (id: string) =>
    patchThread({ extraIds: [...(threadState.extraIds ?? []), id] })

  const detachFromChat = (id: string) =>
    patchThread({
      extraIds: (threadState.extraIds ?? []).filter((x) => x !== id),
    })

  const label = threadState.off
    ? 'World info off'
    : activeCount === 0
      ? 'No lorebooks'
      : `${activeCount} lorebook${activeCount === 1 ? '' : 's'}`
  // When the row gets tight the label degrades to just the number; off still
  // reads as a muted 0.
  const shortLabel = threadState.off ? '0' : `${activeCount}`
  const labelTone =
    activeCount > 0 && !threadState.off
      ? 'text-foreground'
      : 'text-muted-foreground'
  // Explicit forms rather than labelForms: the first word of "No lorebooks"
  // or "World info off" says the opposite of the whole, so the count stands
  // in for the label instead.
  const forms = useMemo(() => ['', shortLabel, label], [shortLabel, label])
  const {
    label: fittedLabel,
    pillRef: measurePill,
    labelRef,
  } = usePillLabel('lorebook', forms)
  const setPill = useCallback(
    (el: HTMLDivElement | null) => {
      pillRef.current = el
      measurePill(el)
    },
    [pillRef, measurePill]
  )

  return (
      <Popover onOpenChange={(open) => open && measure()}>
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
            aria-label="World info for this chat"
          >
            <IconBook
              className={cn(
                'size-4 shrink-0',
                labelTone
              )}
            />
            {(!compact || fittedLabel) && (
              <span
                ref={labelRef}
                className={cn(
                  'truncate leading-normal',
                  fittedLabel === shortLabel && 'tabular-nums',
                  labelTone
                )}
              >
                {compact ? fittedLabel : label}
              </span>
            )}
            <ChevronsUpDown
              className={cn(
                'size-4 shrink-0 text-muted-foreground',
                compact && PILL_CHEVRON
              )}
            />
          </button>

          {/* Inside the trigger box, but a click edits the book rather than
              opening the per-chat list. */}
          <Button
            variant="ghost"
            size="icon-xs"
            className="shrink-0"
            aria-label={cogBook ? `Edit ${cogBook.name}` : 'No lorebook to edit'}
            disabled={!cogBook}
            onClick={(e) => {
              e.stopPropagation()
              if (cogBook)
                navigate({
                  to: route.lorebooks.detail,
                  params: { lorebookId: cogBook.id },
                })
            }}
          >
            <IconSettings size={18} className="text-muted-foreground" />
          </Button>
          </div>
        </PopoverTrigger>

        <PopoverContent
          className="w-80 min-w-72 max-w-[90vw] p-0 backdrop-blur-2xl bg-background/95 border"
          align={align}
          side={compact ? 'top' : 'bottom'}
        >
          <div className="flex flex-col size-full">
            <div className="max-h-80 overflow-y-auto py-1">
              {library.length === 0 && (
                <div className="py-3 px-4 text-sm text-muted-foreground">
                  No lorebooks yet. Create or import one to give this chat some
                  world info.
                </div>
              )}

              {resolved.length > 0 && (
                <div className="px-2 pt-1 pb-0.5 text-[11px] uppercase tracking-wide text-muted-foreground">
                  In this chat
                </div>
              )}
              {resolved.map(({ book, origin, enabled }) => (
                <div
                  key={book.id}
                  className="mx-1 mb-1 px-2 py-1.5 rounded-sm flex items-center gap-2"
                >
                  {origin === 'global' ? (
                    <IconWorld className="size-4 shrink-0 text-muted-foreground" />
                  ) : (
                    <IconBook className="size-4 shrink-0 text-muted-foreground" />
                  )}
                  <div className="flex flex-col min-w-0 flex-1">
                    <span className="text-sm truncate">{book.name}</span>
                    <span className="text-xs text-muted-foreground truncate">
                      {ORIGIN_LABELS[origin]} ·{' '}
                      {book.entries.length === 1
                        ? '1 entry'
                        : `${book.entries.length} entries`}
                    </span>
                  </div>
                  {origin === 'chat' && (
                    <button
                      type="button"
                      className="text-xs text-muted-foreground hover:text-foreground cursor-pointer shrink-0"
                      onClick={() => detachFromChat(book.id)}
                    >
                      Remove
                    </button>
                  )}
                  <Switch
                    checked={enabled}
                    disabled={threadState.off}
                    onCheckedChange={(next) => toggleEnabled(book.id, next)}
                    title={enabled ? 'On for this chat' : 'Off for this chat'}
                  />
                </div>
              ))}

              {available.length > 0 && (
                <>
                  <div className="px-2 pt-2 pb-0.5 text-[11px] uppercase tracking-wide text-muted-foreground">
                    Add to this chat
                  </div>
                  {available.map((book) => (
                    <button
                      key={book.id}
                      type="button"
                      className="w-full text-left mx-1 mb-1 px-2 py-1.5 rounded-sm flex items-center gap-2 hover:bg-secondary/40 cursor-pointer disabled:opacity-50 disabled:cursor-default"
                      onClick={() => attachToChat(book.id)}
                    >
                      <IconBook className="size-4 shrink-0 text-muted-foreground" />
                      <div className="flex flex-col min-w-0 flex-1">
                        <span className="text-sm truncate">{book.name}</span>
                        <span className="text-xs text-muted-foreground truncate">
                          {book.entries.length === 1
                            ? '1 entry'
                            : `${book.entries.length} entries`}
                        </span>
                      </div>
                    </button>
                  ))}
                </>
              )}
            </div>

            <label className="border-t px-3 py-2 flex items-center justify-between gap-2 text-sm cursor-pointer">
              <span>World info in this chat</span>
              <Switch
                checked={!threadState.off}
                onCheckedChange={(on) => patchThread({ off: !on })}
              />
            </label>

            <div className="border-t p-1 flex items-center justify-between">
              <button
                type="button"
                className="text-xs text-muted-foreground px-2 py-1.5 rounded-sm hover:bg-secondary/40 cursor-pointer"
                onClick={() => navigate({ to: route.settings.lorebooks })}
              >
                Manage lorebooks
              </button>
            </div>
          </div>
        </PopoverContent>
      </Popover>
  )
}

export default LorebookPill

import { useCallback, useMemo } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Button } from '@/components/ui/button'
import { IconCheck, IconSettings, IconUserCircle } from '@tabler/icons-react'
import { useNavigate } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { cn } from '@/lib/utils'
import { PILL_CHEVRON } from '@/constants/layout'
import { usePillLabel } from '@/hooks/usePillLabel'
import { labelForms } from '@/lib/pill-labels'
import { ChevronsUpDown } from 'lucide-react'
import { AvatarEmoji } from '@/containers/AvatarEmoji'
import { usePersonas } from '@/hooks/usePersonas'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import { useThreads } from '@/hooks/useThreads'
import { usePendingPersona } from '@/stores/pending-persona'
import { usePopoverAlign } from '@/hooks/usePopoverAlign'
import {
  resolvePersona,
  isPersonaPinned,
  type ThreadPersonaState,
} from '@/lib/persona-selection'

/**
 * The persona switcher pill, beside the model, character and world-info pills.
 * Shows who the user is playing in this chat and lets a single chat depart
 * from the default without changing it everywhere.
 */
export function PersonaPill({
  thread,
  compact = false,
}: {
  thread?: Thread
  /** Toolbar-sized pill for the chatbox row rather than the header. */
  compact?: boolean
}) {
  const navigate = useNavigate()
  // Panel is w-72; measured at open so it hugs the pill's free side.
  const { pillRef, align, measure } = usePopoverAlign(288)
  const personas = usePersonas((s) => s.personas)
  const activePersonaId = useGeneralSetting((s) => s.activePersonaId)
  const updateThread = useThreads((s) => s.updateThread)

  // On the home screen the pick is staged until a thread exists.
  const pending = usePendingPersona((s) => s.pending)
  const patchPending = usePendingPersona((s) => s.patch)

  const threadState: ThreadPersonaState = useMemo(
    () => (thread ? (thread.metadata?.persona ?? {}) : pending),
    [thread, pending]
  )

  const persona = useMemo(
    () =>
      resolvePersona({
        library: personas,
        activeId: activePersonaId,
        threadState,
      }),
    [personas, activePersonaId, threadState]
  )

  const pinned = isPersonaPinned(threadState)

  const patchThread = (patch: ThreadPersonaState) => {
    if (!thread) {
      patchPending(patch)
      return
    }
    updateThread(thread.id, {
      metadata: { ...thread.metadata, persona: { ...threadState, ...patch } },
    })
  }

  /** Picking the default persona clears the pin rather than duplicating it. */
  const pick = (id: string | null) =>
    patchThread({ personaId: id === activePersonaId ? undefined : id })

  const label = persona ? persona.name : 'No persona'
  // The row decides how much of the name fits; see PillRow.
  const forms = useMemo(() => labelForms(label), [label])
  const {
    label: fittedLabel,
    pillRef: measurePill,
    labelRef,
  } = usePillLabel('persona', forms)
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
        <div
          ref={setPill}
          className={cn(
            // Header pills stand alone; the compact form sits in the chatbox
            // toolbar row at the same 28px height as its buttons. min-w-0
            // lets the row squeeze it once space runs out.
            'border relative z-20 flex items-center gap-1.5 rounded-full min-w-0 cursor-pointer',
            compact ? 'pl-2.5 pr-1 h-7 text-xs' : 'pl-4 pr-2 h-10'
          )}
        >
        <button
          type="button"
          className="font-medium cursor-pointer flex items-center gap-1.5 relative z-20 min-w-0"
          aria-label="Persona for this chat"
        >
          {persona?.avatar ? (
            <AvatarEmoji
              avatar={persona.avatar}
              imageClassName="w-4 h-4 rounded-full object-cover"
              textClassName="text-sm leading-none"
            />
          ) : (
            <IconUserCircle
              className={cn(
                'size-4 shrink-0',
                persona ? 'text-foreground' : 'text-muted-foreground'
              )}
            />
          )}
          {(!compact || fittedLabel) && (
            <span
              ref={labelRef}
              className={cn(
                'truncate leading-normal',
                persona ? 'text-foreground' : 'text-muted-foreground'
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

        {/* Inside the trigger box, but a click edits the persona rather than
            opening the picker. */}
        <Button
          variant="ghost"
          size="icon-xs"
          className="shrink-0"
          aria-label="Edit personas"
          onClick={(e) => {
            e.stopPropagation()
            navigate({ to: route.settings.personas })
          }}
        >
          <IconSettings size={18} className="text-muted-foreground" />
        </Button>
        </div>
      </PopoverTrigger>

      <PopoverContent
          className="w-72 min-w-72 max-w-[90vw] p-0 backdrop-blur-2xl bg-background/95 border"
          align={align}
          side={compact ? 'top' : 'bottom'}
        >
          <div className="flex flex-col size-full">
            <div className="max-h-80 overflow-y-auto py-1">
              {personas.length === 0 && (
                <div className="py-3 px-4 text-sm text-muted-foreground">
                  No personas yet. Create one to tell characters who they are
                  talking to.
                </div>
              )}

              {personas.map((p) => {
                const selected = persona?.id === p.id
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => pick(p.id)}
                    className="w-full mx-1 mb-1 px-2 py-1.5 rounded-sm flex items-center gap-2 text-left hover:bg-secondary/40 cursor-pointer"
                  >
                    <span className="size-4 shrink-0 flex items-center justify-center">
                      {p.avatar ? (
                        <AvatarEmoji
                          avatar={p.avatar}
                          imageClassName="w-4 h-4 rounded-full object-cover"
                          textClassName="text-sm leading-none"
                        />
                      ) : (
                        <IconUserCircle className="size-4 text-muted-foreground" />
                      )}
                    </span>
                    <div className="flex flex-col min-w-0 flex-1">
                      <span className="truncate text-sm">{p.name}</span>
                      {p.id === activePersonaId && (
                        <span className="text-[11px] text-muted-foreground">
                          your default
                        </span>
                      )}
                    </div>
                    {selected && (
                      <IconCheck className="size-4 shrink-0 text-foreground" />
                    )}
                  </button>
                )
              })}

              {personas.length > 0 && (
                <button
                  type="button"
                  onClick={() => pick(null)}
                  className="w-full mx-1 mb-1 px-2 py-1.5 rounded-sm flex items-center gap-2 text-left hover:bg-secondary/40 cursor-pointer"
                >
                  <span className="size-4 shrink-0" />
                  <span className="flex-1 text-sm text-muted-foreground">
                    No persona in this chat
                  </span>
                  {!persona && (
                    <IconCheck className="size-4 shrink-0 text-foreground" />
                  )}
                </button>
              )}
            </div>

            {pinned && (
              <div className="border-t px-3 py-2 flex items-center justify-between gap-2">
                <span className="text-[11px] text-muted-foreground">
                  Pinned to this chat
                </span>
                <Button
                  variant="link"
                  size="sm"
                  className="h-auto p-0 text-xs"
                  onClick={() => patchThread({ personaId: undefined })}
                >
                  Follow default
                </Button>
              </div>
            )}

            {/* Same footer the character and lorebook pills have: a quiet
                link out to the full management page. */}
            <div className="border-t p-1 flex items-center justify-between">
              <button
                type="button"
                className="text-xs text-muted-foreground px-2 py-1.5 rounded-sm hover:bg-secondary/40 cursor-pointer"
                onClick={() => navigate({ to: route.settings.personas })}
              >
                Manage personas
              </button>
            </div>
          </div>
        </PopoverContent>
    </Popover>
  )
}

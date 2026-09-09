import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { useState } from 'react'
import {
  DndContext,
  DragOverlay,
  PointerSensor,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'

import HeaderPage from '@/containers/HeaderPage'
import SettingsMenu from '@/containers/SettingsMenu'
import { Card, CardItem } from '@/containers/Card'
import { Switch } from '@/components/ui/switch'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  IconBook,
  IconDotsVertical,
  IconPlus,
  IconWorld,
  IconX,
} from '@tabler/icons-react'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { cn } from '@/lib/utils'
import { useLorebooks, createEmptyLorebook } from '@/hooks/useLorebooks'
import { useLorebookSettings } from '@/hooks/useLorebookSettings'
import type { Lorebook } from '@/lib/lorebook'
import { exportLorebookFile } from '@/lib/lorebook-export'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.settings.lorebooks as any)({
  component: LorebooksSettingsContent,
})

const GLOBAL_ZONE_ID = 'lorebook-global-zone'

const entryCountLabel = (book: Lorebook) =>
  `${book.entries.length} ${book.entries.length === 1 ? 'entry' : 'entries'}`

function GlobalZone({ children }: { children: React.ReactNode }) {
  const { isOver, setNodeRef } = useDroppable({ id: GLOBAL_ZONE_ID })
  return (
    <div
      ref={setNodeRef}
      className={cn(
        'rounded-lg border border-dashed p-3 flex flex-col gap-2 transition-colors',
        isOver ? 'border-primary bg-primary/5' : 'border-border'
      )}
    >
      {children}
    </div>
  )
}

function DraggableRow({
  book,
  children,
}: {
  book: Lorebook
  children: React.ReactNode
}) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: book.id,
  })

  // The whole row is the drag target. The sensor's distance threshold is what
  // keeps a plain click on the row (or on the buttons inside it) working.
  return (
    <div
      ref={setNodeRef}
      {...listeners}
      {...attributes}
      className={cn('outline-none', isDragging && 'opacity-40')}
    >
      {children}
    </div>
  )
}

/**
 * How the scanner behaves everywhere. Per-book and per-chat controls narrow
 * these; nothing here can be overridden upward.
 */
function BehaviourCard() {
  const settings = useLorebookSettings()
  // Recursion is per book now, so the step count only bites when some book
  // asks for it.
  const anyRecursive = useLorebooks((s) =>
    s.lorebooks.some((b) => b.recursiveScanning)
  )

  const numberField = (
    value: number,
    onChange: (n: number) => void,
    opts: { min: number; max: number; disabled?: boolean }
  ) => (
    <Input
      type="number"
      min={opts.min}
      max={opts.max}
      disabled={opts.disabled}
      value={value}
      className="w-20 h-8"
      onChange={(e) => {
        const n = Number(e.target.value)
        if (Number.isNaN(n)) return
        onChange(Math.min(opts.max, Math.max(opts.min, Math.round(n))))
      }}
    />
  )

  return (
    <Card title="World info behaviour">
      <CardItem
        title="World info enabled"
        description="Off means no lorebook entry reaches any prompt, whatever is attached."
        actions={
          <Switch
            checked={settings.enabled}
            onCheckedChange={(v) => settings.set('enabled', v)}
          />
        }
      />
      <CardItem
        title="Scan depth"
        description="How many recent messages are searched for an entry’s keys."
        actions={numberField(
          settings.scanDepth,
          (n) => settings.set('scanDepth', n),
          { min: 1, max: 50 }
        )}
      />
      <CardItem
        title="Token budget"
        description="Percent of the context window world info may take. Lowest-prominence entries are dropped first. 0 removes the cap."
        actions={numberField(
          settings.budgetPercent,
          (n) => settings.set('budgetPercent', n),
          { min: 0, max: 100 }
        )}
      />
      <CardItem
        title="Recursion steps"
        description="How many times the scan repeats over newly activated content, in books whose entries may trigger each other."
        actions={numberField(
          settings.maxRecursionSteps,
          (n) => settings.set('maxRecursionSteps', n),
          { min: 1, max: 5, disabled: !anyRecursive }
        )}
      />
    </Card>
  )
}

function LorebooksSettingsContent() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const lorebooks = useLorebooks((s) => s.lorebooks)
  const addLorebook = useLorebooks((s) => s.addLorebook)
  const deleteLorebook = useLorebooks((s) => s.deleteLorebook)
  const setGlobal = useLorebooks((s) => s.setGlobal)

  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [draggingId, setDraggingId] = useState<string | null>(null)

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } })
  )

  const sorted = lorebooks.slice().sort((a, b) => a.name.localeCompare(b.name))
  const globals = sorted.filter((b) => b.global)
  const deleting = lorebooks.find((b) => b.id === deletingId)
  const dragging = lorebooks.find((b) => b.id === draggingId)

  const openEditor = (lorebookId: string) =>
    navigate({ to: route.lorebooks.detail, params: { lorebookId } })

  const createNew = () => {
    const book = addLorebook(createEmptyLorebook())
    openEditor(book.id)
  }

  const handleDragEnd = (event: DragEndEvent) => {
    setDraggingId(null)
    if (event.over?.id === GLOBAL_ZONE_ID) {
      setGlobal(String(event.active.id), true)
    }
  }

  return (
    <div className="flex flex-col h-full w-full">
      <HeaderPage className="h-auto pt-[calc(var(--spacing)*1.7)] pb-[calc(var(--spacing)*2.5)]">
        <div
          className={cn(
            'flex items-center justify-between w-full mr-2 pr-3',
            !IS_MACOS && 'pr-30'
          )}
        >
          <span className="font-medium text-base font-studio">
            {t('common:settings')}
          </span>
        </div>
      </HeaderPage>
      <div className="flex flex-1 min-h-0">
        <div className="flex size-full">
          <SettingsMenu />
          <div className="flex flex-col gap-4 p-4 pt-0 w-full overflow-y-auto">
            <DndContext
              sensors={sensors}
              onDragStart={(e) => setDraggingId(String(e.active.id))}
              onDragCancel={() => setDraggingId(null)}
              onDragEnd={handleDragEnd}
            >
              <Card>
                <h1 className="text-foreground font-studio font-medium text-sm mb-1">
                  Always active in every chat
                </h1>
                <p className="text-xs text-muted-foreground mb-3">
                  Books here apply to every character and every chat, without
                  being attached to anything.
                </p>
                <GlobalZone>
                  {globals.length === 0 ? (
                    <p className="text-xs text-muted-foreground py-2 text-center">
                      Drag a lorebook here, or click a lorebook’s globe.
                    </p>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {globals.map((book) => (
                        <span
                          key={book.id}
                          className="flex items-center gap-1.5 text-xs pl-2.5 pr-1.5 py-1 rounded-full bg-secondary"
                        >
                          <span className="truncate max-w-52">{book.name}</span>
                          <button
                            type="button"
                            title="Remove from always active"
                            className="text-muted-foreground hover:text-foreground"
                            onClick={() => setGlobal(book.id, false)}
                          >
                            <IconX className="size-3.5" />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </GlobalZone>

                <div className="flex items-center justify-between mt-6 mb-2">
                  <h1 className="text-foreground font-studio font-medium text-sm">
                    Your lorebooks
                  </h1>
                  <Button size="sm" variant="outline" onClick={createNew}>
                    <IconPlus className="size-4" />
                    New lorebook
                  </Button>
                </div>

                {sorted.length === 0 && (
                  <p className="text-xs text-muted-foreground py-6 text-center">
                    No lorebooks yet. Create one, or import a book from the Hub.
                  </p>
                )}

                {sorted.map((book) => (
                  <DraggableRow key={book.id} book={book}>
                    <div
                      className="group flex items-center gap-3 px-3 py-3 rounded-lg my-1 bg-secondary/20 hover:bg-secondary/60 dark:hover:bg-secondary/50 transition-colors cursor-grab active:cursor-grabbing"
                      role="button"
                      tabIndex={0}
                      onClick={() => openEditor(book.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') openEditor(book.id)
                      }}
                    >
                      <div className="size-9 shrink-0 flex items-center justify-center bg-secondary dark:bg-secondary/40 rounded-lg">
                        <IconBook className="size-5 text-muted-foreground" />
                      </div>
                      <div className="flex flex-col min-w-0 flex-1">
                        <span className="text-sm font-studio font-medium truncate">
                          {book.name}
                        </span>
                        <p className="text-xs text-muted-foreground line-clamp-1 mt-0.5">
                          {entryCountLabel(book)}
                          {book.description ? ` · ${book.description}` : ''}
                        </p>
                      </div>
                      <div
                        className="flex items-center gap-1 shrink-0"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title={
                            book.global
                              ? 'Remove from always active'
                              : 'Make active in every chat'
                          }
                          aria-pressed={book.global}
                          onClick={() => setGlobal(book.id, !book.global)}
                        >
                          <IconWorld
                            className={cn(
                              'size-5',
                              book.global
                                ? 'text-primary'
                                : 'text-muted-foreground/40'
                            )}
                          />
                        </Button>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon-sm" title="More">
                              <IconDotsVertical className="text-muted-foreground size-5" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem
                              className="cursor-pointer"
                              onClick={() => openEditor(book.id)}
                            >
                              {t('common:edit')}
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="cursor-pointer"
                              onClick={() => exportLorebookFile(book)}
                            >
                              Export as JSON
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="cursor-pointer text-destructive"
                              onClick={() => setDeletingId(book.id)}
                            >
                              {t('common:delete')}
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  </DraggableRow>
                ))}
              </Card>

              <DragOverlay>
                {dragging && (
                  <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary shadow-lg text-sm">
                    <IconBook className="size-4 text-muted-foreground" />
                    <span className="truncate max-w-52">{dragging.name}</span>
                  </div>
                )}
              </DragOverlay>
            </DndContext>
            <BehaviourCard />
          </div>
        </div>
      </div>

      <Dialog
        open={!!deletingId}
        onOpenChange={(open) => {
          if (!open) setDeletingId(null)
        }}
      >
        <DialogContent className="sm:max-w-[425px] max-w-[90vw]">
          <DialogHeader>
            <DialogTitle>Delete lorebook</DialogTitle>
            <DialogDescription>
              {deleting
                ? `"${deleting.name}" and its ${entryCountLabel(deleting)} will be removed. Characters using it lose it too.`
                : ''}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex flex-col-reverse sm:flex-row sm:justify-end gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDeletingId(null)}
              className="w-full sm:w-auto"
            >
              {t('common:cancel')}
            </Button>
            <Button
              variant="destructive"
              size="sm"
              className="w-full sm:w-auto"
              onClick={() => {
                if (deletingId) deleteLorebook(deletingId)
                setDeletingId(null)
              }}
            >
              {t('common:delete')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default LorebooksSettingsContent

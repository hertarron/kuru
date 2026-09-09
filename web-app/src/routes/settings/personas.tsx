import { createFileRoute } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { useRef, useState } from 'react'

import HeaderPage from '@/containers/HeaderPage'
import SettingsMenu from '@/containers/SettingsMenu'
import { Card } from '@/containers/Card'
import { AvatarEmoji } from '@/containers/AvatarEmoji'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { IconCheck, IconPlus, IconTrash } from '@tabler/icons-react'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { cn } from '@/lib/utils'
import { usePersonas } from '@/hooks/usePersonas'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import { createEmptyPersona, type Persona } from '@/lib/persona'
import { fileToAvatarDataUrl } from '@/lib/character-card'

const isImageAvatar = (a: string | undefined): a is string =>
  !!a && (a.startsWith('data:image/') || a.startsWith('/images/'))

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.settings.personas as any)({
  component: PersonasSettingsContent,
})

const USER_TOKEN = '{{user}}'

function PersonaCard({
  persona,
  isActive,
  onActivate,
  onChange,
  onDelete,
}: {
  persona: Persona
  isActive: boolean
  onActivate: () => void
  onChange: (next: Persona) => void
  onDelete: () => void
}) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  return (
    <Card>
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <div className="size-8 shrink-0 rounded-full bg-main-view-fg/5 flex items-center justify-center">
            <AvatarEmoji
              avatar={persona.avatar || '🧑'}
              imageClassName="w-8 h-8 rounded-full object-cover"
            />
          </div>
          <Input
            value={persona.name}
            onChange={(e) => onChange({ ...persona, name: e.target.value })}
            placeholder="Name"
            className="h-8 font-medium"
          />
          <Button
            variant={isActive ? 'default' : 'outline'}
            size="sm"
            className="h-8 shrink-0 border border-transparent transition-none"
            onClick={onActivate}
            disabled={isActive}
            title="Use this persona in chats that have not picked one"
          >
            {isActive && <IconCheck size={14} className="mr-1" />}
            {isActive ? 'Default' : 'Make default'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 shrink-0 text-muted-foreground"
            onClick={onDelete}
            title="Delete persona"
          >
            <IconTrash size={16} />
          </Button>
        </div>

        <div className="flex flex-col gap-1">
          {/* Real images now; an old emoji value still previews until an
              upload replaces or Remove clears it. */}
          <label className="text-xs text-muted-foreground">Avatar</label>
          <div className="flex items-center gap-2">
            <button
              type="button"
              title="Upload an image"
              className="size-12 rounded-full overflow-hidden border bg-main-view-fg/5 flex items-center justify-center cursor-pointer hover:opacity-90 transition-opacity"
              onClick={() => fileInputRef.current?.click()}
            >
              <AvatarEmoji
                avatar={persona.avatar || '🧑'}
                imageClassName="w-12 h-12 object-cover"
              />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={async (e) => {
                const file = e.target.files?.[0]
                e.target.value = ''
                if (!file) return
                try {
                  onChange({
                    ...persona,
                    avatar: await fileToAvatarDataUrl(file),
                  })
                } catch {
                  // Undecodable image: keep the current avatar.
                }
              }}
            />
            {isImageAvatar(persona.avatar) && (
              <Button
                variant="ghost"
                size="sm"
                className="h-8 text-muted-foreground"
                onClick={() => onChange({ ...persona, avatar: '' })}
              >
                Remove
              </Button>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Description</label>
          <Textarea
            value={persona.description}
            onChange={(e) =>
              onChange({ ...persona, description: e.target.value })
            }
            placeholder="Who you are in the story: appearance, history, how others treat you. Reaches the model after the character's own sheet."
            rows={4}
          />
        </div>
      </div>
    </Card>
  )
}

function PersonasSettingsContent() {
  const { t } = useTranslation()
  const personas = usePersonas((s) => s.personas)
  const addPersona = usePersonas((s) => s.addPersona)
  const updatePersona = usePersonas((s) => s.updatePersona)
  const deletePersona = usePersonas((s) => s.deletePersona)
  const activePersonaId = useGeneralSetting((s) => s.activePersonaId)
  const setActivePersonaId = useGeneralSetting((s) => s.setActivePersonaId)

  const [pendingDelete, setPendingDelete] = useState<Persona | null>(null)

  const handleCreate = () => {
    const created = addPersona(createEmptyPersona())
    // The first persona is the one every chat will use, so adopt it rather
    // than making the user press "Make default" on a library of one.
    if (!activePersonaId) setActivePersonaId(created.id)
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
            <Card>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h1 className="text-foreground font-studio font-medium text-sm mb-1">
                    Personas
                  </h1>
                  <p className="text-xs text-muted-foreground">
                    Who you are playing. The name replaces {USER_TOKEN} in
                    prompts, and the description tells the character who they
                    are talking to. Any chat can pick a different one from the
                    persona pill.
                  </p>
                </div>
                <Button size="sm" className="shrink-0" onClick={handleCreate}>
                  <IconPlus size={14} className="mr-1" />
                  New persona
                </Button>
              </div>
            </Card>

            {personas.length === 0 ? (
              <Card>
                <p className="text-xs text-muted-foreground">
                  No personas yet. Without one the model is told nothing about
                  you, and {USER_TOKEN} stays unresolved.
                </p>
              </Card>
            ) : (
              personas.map((persona) => (
                <PersonaCard
                  key={persona.id}
                  persona={persona}
                  isActive={persona.id === activePersonaId}
                  onActivate={() => setActivePersonaId(persona.id)}
                  onChange={updatePersona}
                  onDelete={() => setPendingDelete(persona)}
                />
              ))
            )}
          </div>
        </div>
      </div>

      <Dialog
        open={!!pendingDelete}
        onOpenChange={(open) => !open && setPendingDelete(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete persona</DialogTitle>
            <DialogDescription>
              Delete &ldquo;{pendingDelete?.name}&rdquo;? Chats pinned to it
              fall back to your default persona.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="link" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (pendingDelete) deletePersona(pendingDelete.id)
                setPendingDelete(null)
              }}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

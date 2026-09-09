import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { useState } from 'react'

import { useCharacters } from '@/hooks/useCharacters'
import { useGeneralSetting } from '@/hooks/useGeneralSetting'
import { usePersonas } from '@/hooks/usePersonas'
import { useRpTextSettings } from '@/hooks/useRpTextSettings'
import { Switch } from '@/components/ui/switch'

import HeaderPage from '@/containers/HeaderPage'
import { IconCirclePlus, IconPencil, IconTrash } from '@tabler/icons-react'
import { AvatarEmoji } from '@/containers/AvatarEmoji'
import { DeleteCharacterDialog } from '@/containers/dialogs/DeleteCharacterDialog'
import { CharacterPreviewSheet } from '@/containers/CharacterPreviewSheet'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import SettingsMenu from '@/containers/SettingsMenu'
import { cn } from '@/lib/utils'
import { Card, CardItem } from '@/containers/Card'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ChevronsUpDown } from 'lucide-react'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.settings.characters as any)({
  component: CharactersSettingsContent,
})

function CharactersSettingsContent() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const {
    characters,
    deleteCharacter,
    defaultCharacterId,
    setDefaultCharacter,
  } = useCharacters()
  const personas = usePersonas((s) => s.personas)
  const activePersonaId = useGeneralSetting((s) => s.activePersonaId)
  const activePersona = personas.find((p) => p.id === activePersonaId)
  const hideOocFromPrompt = useRpTextSettings((s) => s.hideOocFromPrompt)
  const trimIncompleteSentence = useRpTextSettings(
    (s) => s.trimIncompleteSentence
  )
  const setRpText = useRpTextSettings((s) => s.set)
  const chubToken = useGeneralSetting((s) => s.chubToken)
  const setChubToken = useGeneralSetting((s) => s.setChubToken)
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [previewCharacter, setPreviewCharacter] = useState<Character | null>(
    null
  )

  const sorted = characters.slice().sort((a, b) => a.created_at - b.created_at)
  const defaultCharacter = sorted.find((c) => c.id === defaultCharacterId)

  const confirmDelete = () => {
    if (deletingId) {
      deleteCharacter(deletingId)
      setDeleteConfirmOpen(false)
      setDeletingId(null)
    }
  }

  return (
    <div className="flex flex-col h-full w-full">
      <HeaderPage className="h-auto pt-[calc(var(--spacing)*1.7)] pb-[calc(var(--spacing)*2.5)]">
        <div className={cn('flex items-center justify-between w-full mr-2 pr-3', !IS_MACOS && 'pr-30')}>
          <span className="font-medium text-base font-studio">
            {t('common:settings')}
          </span>
        </div>
      </HeaderPage>
      <div className="flex flex-1 min-h-0">
        <div className="flex size-full">
          <SettingsMenu />
          <div className="flex flex-col gap-4 p-4 pt-0 w-full overflow-y-auto">
            {/* Default character */}
            <Card>
              <CardItem
                title={t('characters:defaultSection')}
                description={t('characters:defaultDesc')}
                actions={
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm" className="justify-between">
                        <span className={cn('truncate')}>
                          {defaultCharacter?.name ?? t('characters:lastUsed')}
                        </span>
                        <ChevronsUpDown className="size-4 shrink-0 text-muted-foreground ml-2" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-40 max-h-80">
                      <DropdownMenuItem
                        key="none"
                        className={cn(
                          'cursor-pointer my-0.5',
                          !defaultCharacterId && 'bg-secondary-foreground/8'
                        )}
                        onClick={() => setDefaultCharacter('')}
                      >
                        {t('characters:lastUsed')}
                      </DropdownMenuItem>
                      {sorted.map((c) => (
                        <DropdownMenuItem
                          key={c.id}
                          className={cn(
                            'cursor-pointer my-0.5',
                            defaultCharacterId === c.id && 'bg-secondary-foreground/8'
                          )}
                          onClick={() => setDefaultCharacter(c.id)}
                        >
                          {c.name}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                }
              />

              {/* Personas own {{user}} now; this is a pointer, not a field. */}
              <CardItem
                title="Persona"
                description="Who you play, and what {{user}} resolves to. Personas live in their own library."
                actions={
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => navigate({ to: route.settings.personas })}
                  >
                    {activePersona ? activePersona.name : 'Set up personas'}
                  </Button>
                }
              />

              {/* Chub.ai API token — optional, unlocks NSFW/private listings */}
              <CardItem
                title="Chub.ai API token"
                description="Optional token for the character search in the Hub. Only needed for NSFW or private cards. Get it from chub.ai > Local Storage > URQL_TOKEN."
                actions={
                  <Input
                    type="password"
                    value={chubToken || ''}
                    onChange={(e) => setChubToken(e.target.value)}
                    placeholder="Paste token (optional)"
                    className="w-44"
                  />
                }
              />

              {/* Roleplay behaviour. Global on purpose: a preference about
                  how you read and prompt, not a property of one card. Every
                  read is gated on the chat's character being a roleplay one,
                  so assistants ignore all of it. */}
              <h1 className="text-foreground font-studio font-medium text-sm mt-4 mb-1">
                Roleplay behaviour
              </h1>
              <p className="text-xs text-muted-foreground mb-2">
                Applies to roleplay characters only. Assistants are untouched.
                Both edit the prompt, never the transcript — open the context
                visualizer in a chat to see exactly what the model received.
              </p>
              <CardItem
                title="Hide OOC from the prompt"
                description="Drops ((double-parenthesised)), [OOC: ...] and <ooc> asides from earlier turns. The turn you are sending is never stripped, so an OOC question still reaches the model."
                actions={
                  <Switch
                    checked={hideOocFromPrompt}
                    onCheckedChange={(v) => setRpText('hideOocFromPrompt', v)}
                  />
                }
              />
              <CardItem
                title="Trim incomplete last sentence"
                description="Drops a dangling fragment from earlier replies that hit the output cap mid-sentence, so the model is not asked to continue from half a word. Leaves short unpunctuated action beats alone."
                actions={
                  <Switch
                    checked={trimIncompleteSentence}
                    onCheckedChange={(v) =>
                      setRpText('trimIncompleteSentence', v)
                    }
                  />
                }
              />

              <div className="flex items-center justify-between gap-3 mt-4 mb-4">
                <h1 className="text-foreground font-studio font-medium text-sm">
                  {t('characters:allCharacters')}
                </h1>
                <Button
                  size="sm"
                  className="shrink-0"
                  onClick={() =>
                    navigate({
                      to: route.characters.detail,
                      params: { characterId: 'new' },
                    })
                  }
                >
                  <IconCirclePlus size={16} />
                  {t('common:newCharacter')}
                </Button>
              </div>
              {sorted.map((character) => (
                <div
                  className="group flex items-center gap-3 px-3 py-3 rounded-lg my-1 bg-secondary/20 hover:bg-secondary dark:hover:bg-secondary/20 transition-colors cursor-pointer"
                  key={character.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => setPreviewCharacter(character)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ')
                      setPreviewCharacter(character)
                  }}
                >
                  <div className="size-9 shrink-0 flex items-center justify-center bg-secondary dark:bg-secondary/40 rounded-lg overflow-hidden">
                    {character?.avatar && (
                      <AvatarEmoji
                        avatar={character?.avatar}
                        imageClassName="size-6 object-contain"
                        textClassName="text-2xl"
                      />
                    )}
                  </div>
                  <div className="flex flex-col min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-studio font-medium truncate">
                        {character.name}
                      </span>
                      {defaultCharacterId === character.id && (
                        <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full text-muted-foreground bg-foreground/10 leading-none shrink-0">
                          {t('characters:isDefault')}
                        </span>
                      )}
                    </div>
                    {character.description && (
                      <p className="text-xs text-muted-foreground line-clamp-1 pr-12 mt-0.5">
                        {character.description}
                      </p>
                    )}
                  </div>
                  <div
                    className="flex items-center shrink-0"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      title={t('common:edit')}
                      onClick={() =>
                        navigate({
                          to: route.characters.detail,
                          params: { characterId: character.id },
                        })
                      }
                    >
                      <IconPencil className="text-muted-foreground size-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      title={t('common:delete')}
                      onClick={() => {
                        setDeletingId(character.id)
                        setDeleteConfirmOpen(true)
                      }}
                    >
                      <IconTrash className="text-destructive size-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </Card>
          </div>
        </div>
      </div>
      <CharacterPreviewSheet
        source={
          previewCharacter
            ? { kind: 'library', character: previewCharacter }
            : null
        }
        onOpenChange={(open) => {
          if (!open) setPreviewCharacter(null)
        }}
      />
      <DeleteCharacterDialog
        open={deleteConfirmOpen}
        onOpenChange={setDeleteConfirmOpen}
        onConfirm={confirmDelete}
      />
    </div>
  )
}

export default CharactersSettingsContent


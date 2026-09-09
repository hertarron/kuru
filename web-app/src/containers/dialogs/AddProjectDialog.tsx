import { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useThreadManagement } from '@/hooks/useThreadManagement'
import { useCharacters } from '@/hooks/useCharacters'
import { AvatarEmoji } from '@/containers/AvatarEmoji'
import { toast } from 'sonner'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { ChevronDown } from 'lucide-react'

interface AddProjectDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  editingKey: string | null
  initialData?: {
    id: string
    name: string
    updated_at: number
    assistantId?: string
  }
  // Legacy field name kept: projects persist their linked character as
  // `assistantId` on disk.
  onSave: (name: string, assistantId?: string) => void
}

export default function AddProjectDialog({
  open,
  onOpenChange,
  editingKey,
  initialData,
  onSave,
}: AddProjectDialogProps) {
  const { t } = useTranslation()
  const [name, setName] = useState(initialData?.name || '')
  const [selectedCharacterId, setSelectedCharacterId] = useState<string | undefined>(initialData?.assistantId)
  const { folders } = useThreadManagement()
  const { characters } = useCharacters()

  const selectedCharacter = characters.find((c) => c.id === selectedCharacterId)

  useEffect(() => {
    if (open) {
      setName(initialData?.name || '')
      setSelectedCharacterId(initialData?.assistantId)
    }
  }, [open, initialData])

  const handleSave = () => {
    if (!name.trim()) return

    const trimmedName = name.trim()

    // Check for duplicate names (excluding current project when editing)
    const isDuplicate = folders.some(
      (folder) =>
        folder.name.toLowerCase() === trimmedName.toLowerCase() &&
        folder.id !== editingKey
    )

    if (isDuplicate) {
      toast.warning(t('projects.addProjectDialog.alreadyExists', { projectName: trimmedName }))
      return
    }

    onSave(trimmedName, selectedCharacterId)

    // Show success message
    if (editingKey) {
      toast.success(t('projects.addProjectDialog.updateSuccess', { projectName: trimmedName }))
    } else {
      toast.success(t('projects.addProjectDialog.createSuccess', { projectName: trimmedName }))
    }
    setName('')
    setSelectedCharacterId(undefined)
  }

  const handleCancel = () => {
    onOpenChange(false)
    setName('')
    setSelectedCharacterId(undefined)
  }

  // Check if the button should be disabled
  const hasChanged = editingKey
    ? name.trim() !== initialData?.name || selectedCharacterId !== initialData?.assistantId
    : true
  const isButtonDisabled = !name.trim() || (editingKey && !hasChanged)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            {editingKey ? t('projects.addProjectDialog.editTitle') : t('projects.addProjectDialog.createTitle')}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t('projects.addProjectDialog.namePlaceholder')}
              className="mt-1"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !isButtonDisabled) {
                  handleSave()
                }
              }}
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1.5 block">
              {t('characters:character')}
            </label>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-between rounded-md"
                >
                  {selectedCharacter ? (
                    <div className="flex items-center gap-2">
                      {selectedCharacter.avatar && (
                        <AvatarEmoji
                          avatar={selectedCharacter.avatar}
                          imageClassName="w-4 h-4 object-contain"
                          textClassName="text-sm"
                        />
                      )}
                      <span>{selectedCharacter.name}</span>
                    </div>
                  ) : (
                    <span className="text-muted-foreground">
                      {t('characters:selectCharacter')}
                    </span>
                  )}
                  <ChevronDown className="size-4 text-muted-foreground" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-(--radix-dropdown-menu-trigger-width)">
                <DropdownMenuItem
                  onSelect={() => setSelectedCharacterId(undefined)}
                >
                  <span className="text-muted-foreground">
                    {t('characters:noCharacter')}
                  </span>
                </DropdownMenuItem>
                {characters.map((character) => (
                  <DropdownMenuItem
                    key={character.id}
                    onSelect={() => setSelectedCharacterId(character.id)}
                  >
                    <div className="flex items-center gap-2">
                      {character.avatar && (
                        <AvatarEmoji
                          avatar={character.avatar}
                          imageClassName="w-4 h-4 object-contain"
                          textClassName="text-sm"
                        />
                      )}
                      <span>{character.name}</span>
                    </div>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        <DialogFooter>
          <Button size="sm" variant="ghost" onClick={handleCancel}>
            {t('cancel')}
          </Button>
          <Button size="sm" onClick={handleSave} disabled={Boolean(isButtonDisabled)}>
            {editingKey ? t('projects.addProjectDialog.updateButton') : t('projects.addProjectDialog.createButton')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

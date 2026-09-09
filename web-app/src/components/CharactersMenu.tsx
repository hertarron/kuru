import { DropdownMenuItem } from '@/components/ui/dropdown-menu'
import { AvatarEmoji } from '@/containers/AvatarEmoji'
import { IconUser } from '@tabler/icons-react'

type CharacterMenuProps = {
  selectedCharacter: string | undefined
  setSelectedCharacter: (character: string) => void
  currentThread: Thread | undefined
  updateCurrentThreadCharacter: (character?: Character | Assistant) => void
  characters: Character[]
}

export function CharactersMenu({
  selectedCharacter,
  setSelectedCharacter,
  currentThread,
  updateCurrentThreadCharacter,
  characters,
}: CharacterMenuProps) {
  const threadCharacter = currentThread?.assistants?.[0]
  const deletedCharacter =
    threadCharacter &&
    threadCharacter.id !== 'model-only' &&
    !characters.some((c) => c.id === threadCharacter.id)
      ? threadCharacter
      : null
  const noSelectedCharacter = currentThread
    ? !threadCharacter || threadCharacter.id === 'model-only'
    : !selectedCharacter
  return (
    <>
      {deletedCharacter && (
        <DropdownMenuItem className="bg-accent" disabled>
          <div className="flex items-center gap-2 w-full">
            {deletedCharacter.avatar ? (
              <AvatarEmoji
                avatar={deletedCharacter.avatar}
                imageClassName="w-4 h-4 object-contain"
                textClassName="text-sm"
              />
            ) : (
              <IconUser size={18} className="text-muted-foreground" />
            )}
            <span>{deletedCharacter.name || 'Unnamed character'} (deleted)</span>
            <span className="ml-auto text-xs text-muted-foreground">✓</span>
          </div>
        </DropdownMenuItem>
      )}
      <DropdownMenuItem
        className={noSelectedCharacter ? 'bg-accent' : ''}
        onClick={() => {
          setSelectedCharacter('')
          if (currentThread) {
            updateCurrentThreadCharacter(undefined)
          }
        }}
      >
        <div className="flex items-center gap-2 w-full">
          <span className="text-muted-foreground">—</span>
          <span>None</span>
          {noSelectedCharacter && (
            <span className="ml-auto text-xs text-muted-foreground">✓</span>
          )}
        </div>
      </DropdownMenuItem>
      {characters.length > 0 ? (
        characters.map((character) => {
          const isSelected = currentThread
            ? currentThread?.assistants?.some((a) => a.id === character.id)
            : selectedCharacter === character.id
          return (
            <DropdownMenuItem
              key={character.id}
              className={isSelected ? 'bg-accent' : ''}
              onClick={() => {
                if (currentThread) {
                  updateCurrentThreadCharacter(character)
                } else {
                  setSelectedCharacter(character ? character.id : '')
                }
              }}
            >
              <div className="flex items-center gap-2 w-full">
                <AvatarEmoji
                  avatar={character.avatar}
                  imageClassName="w-4 h-4 object-contain"
                  textClassName="text-sm"
                />
                <span>{character.name || 'Unnamed character'}</span>
                {isSelected && (
                  <span className="ml-auto text-xs text-muted-foreground">
                    ✓
                  </span>
                )}
              </div>
            </DropdownMenuItem>
          )
        })
      ) : (
        <DropdownMenuItem disabled>
          <span className="text-muted-foreground">No characters available</span>
        </DropdownMenuItem>
      )}
    </>
  )
}

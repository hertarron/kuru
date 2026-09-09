import React, { memo } from 'react'

/**
 * Checks if an avatar is a custom image (a bundled '/images/' path or a
 * data URI, e.g. an avatar imported from a character card PNG).
 */
const isCustomImageAvatar = (avatar: React.ReactNode): avatar is string => {
  return (
    typeof avatar === 'string' &&
    (avatar.startsWith('/images/') || avatar.startsWith('data:image/'))
  )
}

/**
 * Component for rendering assistant avatars with consistent styling
 */
interface AvatarEmojiProps {
  avatar?: React.ReactNode
  imageClassName?: string
  textClassName?: string
}

export const AvatarEmoji: React.FC<AvatarEmojiProps> = memo(({
  avatar,
  imageClassName = 'w-5 h-5 object-contain',
  textClassName = 'text-base',
}) => {
  if (!avatar) return null
  if (isCustomImageAvatar(avatar)) {
    return <img src={avatar} alt="Custom avatar" className={imageClassName} />
  }

  return <span className={textClassName}>{avatar}</span>
})

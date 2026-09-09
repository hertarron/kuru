import { describe, it, expect } from 'vitest'
import { portraitsByMessageId, isPortraitImage } from '../scene-portraits'

const IMG = 'data:image/png;base64,aaa'
const ALT = '/images/other.png'

const characters = [
  { id: 'ana', avatar: IMG },
  { id: 'bo', avatar: ALT },
  { id: 'emoji', avatar: '🐉' },
]

const charOpts = {
  stampKey: 'characterId',
  library: characters,
  fallback: IMG,
  rendersOn: (role: string) => role === 'assistant',
}

const msg = (id: string, role: string, metadata?: unknown) => ({
  id,
  role,
  metadata,
})

describe('isPortraitImage', () => {
  it('takes data urls and app image paths, not emoji', () => {
    expect(isPortraitImage(IMG)).toBe(true)
    expect(isPortraitImage(ALT)).toBe(true)
    expect(isPortraitImage('🐉')).toBe(false)
    expect(isPortraitImage(undefined)).toBe(false)
  })
})

describe('portraitsByMessageId', () => {
  it('only maps rows that render the portrait', () => {
    const map = portraitsByMessageId(
      [msg('greeting', 'assistant'), msg('u1', 'user')],
      charOpts
    )
    expect(map).toEqual({ greeting: IMG })
    expect('u1' in map).toBe(false)
  })

  it('carries a stamp forward to the replies that follow it', () => {
    const map = portraitsByMessageId(
      [
        msg('greeting', 'assistant'),
        msg('u1', 'user', { characterId: 'bo' }),
        msg('a1', 'assistant'),
        msg('u2', 'user'),
        msg('a2', 'assistant'),
      ],
      charOpts
    )
    // The greeting predates the switch and keeps the live pick.
    expect(map.greeting).toBe(IMG)
    expect(map.a1).toBe(ALT)
    // An unstamped turn stays in the era the last stamp opened.
    expect(map.a2).toBe(ALT)
  })

  it('gives no portrait for a stamp naming someone deleted', () => {
    const map = portraitsByMessageId(
      [msg('u1', 'user', { characterId: 'gone' }), msg('a1', 'assistant')],
      charOpts
    )
    expect(map.a1).toBeUndefined()
  })

  it('gives no portrait for an emoji avatar', () => {
    const map = portraitsByMessageId(
      [msg('u1', 'user', { characterId: 'emoji' }), msg('a1', 'assistant')],
      charOpts
    )
    expect(map.a1).toBeUndefined()
  })

  it('reads persona stamps off user rows', () => {
    const map = portraitsByMessageId(
      [
        msg('u1', 'user'),
        msg('a1', 'assistant'),
        msg('u2', 'user', { personaId: 'bo' }),
      ],
      {
        stampKey: 'personaId',
        library: characters,
        fallback: IMG,
        rendersOn: (role) => role === 'user',
      }
    )
    expect(map).toEqual({ u1: IMG, u2: ALT })
  })
})

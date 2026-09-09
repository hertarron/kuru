/* eslint-disable @typescript-eslint/no-explicit-any */
import { createFileRoute, useSearch } from '@tanstack/react-router'
import ChatInput from '@/containers/ChatInput'
import HeaderPage from '@/containers/HeaderPage'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { useTools } from '@/hooks/useTools'
import { cn } from '@/lib/utils'

import { useModelProvider } from '@/hooks/useModelProvider'
import SetupScreen from '@/containers/SetupScreen'
import { route } from '@/constants/routes'
import {
  CONTENT_COLUMN,
  FADE_TOP_CLEARANCE,
  FADE_TOP_MASK,
} from '@/constants/layout'
import { hasUsableProvider } from '@/lib/providerReadiness'

type ThreadModel = {
  id: string
  provider: string
}

type SearchParams = {
  threadModel?: ThreadModel
}
import { useEffect, useMemo, useState } from 'react'
import { IconChevronLeft, IconChevronRight } from '@tabler/icons-react'
import { useThreads } from '@/hooks/useThreads'
import { RenderMarkdown } from '@/containers/RenderMarkdown'
import { useCharacters } from '@/hooks/useCharacters'
import { isPortraitImage } from '@/lib/scene-portraits'

export const Route = createFileRoute(route.home as any)({
  component: Index,
  validateSearch: (search: Record<string, unknown>): SearchParams => {
    const result: SearchParams = {
      threadModel: search.threadModel as ThreadModel | undefined,
    }

    return result
  },
})

function Index() {
  const { t } = useTranslation()
  const { providers } = useModelProvider()
  const search = useSearch({ from: route.home as any })
  const threadModel = search.threadModel
  const { setCurrentThreadId } = useThreads()
  useTools()

  const hasValidProviders = hasUsableProvider(providers)

  // The selected/default character's opening message(s), previewed on the
  // home screen before the chat starts. Index 0 is first_mes, the rest are
  // alternate_greetings — the same ordering the thread seeding uses.
  const { currentCharacter } = useCharacters()
  const greetings = useMemo(() => {
    const c = currentCharacter as Character | undefined
    if (!c) return []
    return [c.first_mes ?? '', ...(c.alternate_greetings ?? [])]
      .map((g) => g.trim())
      .filter(Boolean)
  }, [currentCharacter])

  // The greeting is the character speaking, so it carries the same portrait
  // the thread page puts above an assistant turn. Without it the portrait
  // only appeared once the chat started, which read as a bug.
  const greetingPortrait = isPortraitImage(currentCharacter?.avatar)
    ? currentCharacter.avatar
    : undefined

  const [greetIndex, setGreetIndex] = useState(0)
  useEffect(() => {
    setGreetIndex(0)
  }, [currentCharacter?.id])
  const activeGreeting = Math.min(greetIndex, greetings.length - 1)

  useEffect(() => {
    setCurrentThreadId(undefined)
  }, [setCurrentThreadId])

  if (!hasValidProviders) {
    return <SetupScreen />
  }

  return (
    <div className="flex h-full flex-col">
      {/* Model, character and world info moved into the chatbox toolbar; the
          band stays for the window drag strip and the collapsed-sidebar
          controls HeaderPage renders itself. */}
      <HeaderPage />
      {/* Greeting scrolls here; the chat input stays pinned below. my-auto
          centers short content but lets long greetings scroll from the top
          (justify-center would clip overflow past the top edge). */}
      <div
        className="min-h-0 flex-1 overflow-y-auto inline-flex flex-col"
        style={{ maskImage: FADE_TOP_MASK, WebkitMaskImage: FADE_TOP_MASK }}
      >
        <div className={cn(CONTENT_COLUMN, FADE_TOP_CLEARANCE, 'my-auto pb-4')}>
          {greetings.length > 0 ? (
            <div className="mb-2 flex flex-col items-start gap-2">
              {greetingPortrait && (
                <img
                  src={greetingPortrait}
                  alt=""
                  className="size-24 rounded-xl object-cover border shrink-0"
                />
              )}
              <div
                dir="auto"
                className="select-text max-w-[85%] text-left text-foreground"
              >
                <RenderMarkdown
                  key={activeGreeting}
                  content={greetings[activeGreeting]}
                  messageId={`home-greeting-${activeGreeting}`}
                />
              </div>
              {greetings.length > 1 && (
                <div className="flex items-center gap-1 text-muted-foreground">
                  <button
                    type="button"
                    aria-label="Previous greeting"
                    className="disabled:opacity-30 hover:text-foreground transition-colors"
                    disabled={activeGreeting === 0}
                    onClick={() => setGreetIndex(activeGreeting - 1)}
                  >
                    <IconChevronLeft size={16} />
                  </button>
                  <span className="text-xs tabular-nums">
                    {activeGreeting + 1}/{greetings.length}
                  </span>
                  <button
                    type="button"
                    aria-label="Next greeting"
                    className="disabled:opacity-30 hover:text-foreground transition-colors"
                    disabled={activeGreeting === greetings.length - 1}
                    onClick={() => setGreetIndex(activeGreeting + 1)}
                  >
                    <IconChevronRight size={16} />
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className={cn('text-center')}>
              <h1
                className={cn(
                  'text-2xl font-studio font-medium',
                )}
              >
                {t('chat:description')}
              </h1>
            </div>
          )}
        </div>
      </div>
      {/* Flush with the bottom of the content column, which is the bottom
          edge of the sidebar card beside it. */}
      <div className="shrink-0 pt-2">
        <div className={CONTENT_COLUMN}>
          <ChatInput
            showSpeedToken={false}
            model={threadModel}
            initialMessage={true}
            initialGreetingIndex={activeGreeting}
          />
        </div>
      </div>
    </div>
  )
}


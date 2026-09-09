import { createFileRoute } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import HeaderPage from '@/containers/HeaderPage'
import SettingsMenu from '@/containers/SettingsMenu'
import { Card, CardItem } from '@/containers/Card'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { cn } from '@/lib/utils'
import { useMemorySettings } from '@/hooks/useMemorySettings'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.settings.memory as any)({
  component: MemorySettingsContent,
})

function MemorySettingsContent() {
  const { t } = useTranslation()
  const settings = useMemorySettings()

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
            <Card title="Memory behaviour">
              <CardItem
                title="Auto by default"
                description="Untouched chats fold on their own once a whole batch is owed. Sizes and per-chat overrides live in each chat's context view."
                actions={
                  <Switch
                    checked={settings.autoDefault}
                    onCheckedChange={(v) => settings.set('autoDefault', v)}
                  />
                }
              />
            </Card>

            <Card title="Fold prompts">
              <CardItem
                title="Chapter prompt"
                description="How a batch of messages becomes a chapter. {{maxlines}}, {{prior}} and {{messages}} are filled in per fold."
                actions={
                  <Textarea
                    value={settings.chapterPrompt}
                    rows={8}
                    className="w-full max-w-xl text-xs font-mono"
                    onChange={(e) => settings.set('chapterPrompt', e.target.value)}
                  />
                }
              />
              <CardItem
                title="Canon prompt"
                description="How discarded chapters become permanent facts. {{canon}} and {{chapters}} are filled in per pass."
                actions={
                  <Textarea
                    value={settings.canonPrompt}
                    rows={8}
                    className="w-full max-w-xl text-xs font-mono"
                    onChange={(e) => settings.set('canonPrompt', e.target.value)}
                  />
                }
              />
              <CardItem
                title="Reset prompts"
                description="Restore both templates to their defaults."
                actions={
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => settings.resetPrompts()}
                  >
                    Reset prompts
                  </Button>
                }
              />
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MemorySettingsContent

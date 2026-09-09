import { Link } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { cn } from '@/lib/utils'

/**
 * The Hub's page switcher. Models, characters and lorebooks are all
 * "browse a remote catalog and download from it", so they share one
 * sidebar destination and differ only by this row.
 *
 * Rendered directly under HeaderPage, on the gutter of the page below it,
 * so each Hub page keeps its own search/filter header untouched.
 */
export function HubTabs({ className }: { className?: string }) {
  const { t } = useTranslation()

  const tabs = [
    { to: route.hub.index, label: t('common:models') },
    { to: route.hub.characters, label: t('common:characters') },
    { to: route.hub.lorebooks, label: t('common:lorebooks') },
  ]

  return (
    <div className={cn('flex items-center gap-1 shrink-0', className)}>
      {tabs.map((tab) => (
        <Link
          key={tab.to}
          to={tab.to}
          // `activeOptions.exact` keeps /hub/ from lighting up on /hub/characters.
          activeOptions={{ exact: true }}
          className={cn(
            'px-2.5 py-1 rounded-sm text-sm font-medium text-muted-foreground',
            'hover:bg-secondary hover:dark:bg-secondary/60',
            '[&.active]:bg-secondary [&.active]:dark:bg-secondary/80 [&.active]:text-foreground'
          )}
        >
          {tab.label}
        </Link>
      ))}
    </div>
  )
}

export default HubTabs

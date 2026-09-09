import { useLeftPanel } from '@/hooks/useLeftPanel'
import { cn } from '@/lib/utils'
import {
  IconLayoutSidebar,
} from '@tabler/icons-react'
import { ReactNode, memo } from 'react'
import { Button } from "@/components/ui/button"
import { useSidebarLocked } from '@/lib/sidebar-routes'
import { useTitlebarLayout } from '@/stores/titlebar-layout-store'

type HeaderPageProps = {
  children?: ReactNode
  /** Appended last so tailwind-merge can override the default height. */
  className?: string
  /**
   * Pull the child back to the panel's left edge so a child using
   * CONTENT_COLUMN centers on the panel rather than on the space left over
   * beside the header's own padding. Only takes effect with the sidebar open;
   * collapsed, the toggle sits in this row and the child stays clear of it.
   */
  bleed?: boolean
}
const HeaderPage = memo(function HeaderPage({ children, className, bleed }: HeaderPageProps) {
  const { open, setLeftPanel } = useLeftPanel()
  // Settings and hub lock the sidebar open, so there is no collapsed strip to
  // own here — skip the toggle/download column even if a collapsed chat state
  // is persisted.
  const isLocked = useSidebarLocked()
  // Collapsed, this header owns the top-left strip — indent past left-anchored Linux
  // window controls (size-8 each at left-4); macOS uses the pl-24 class below.
  const leftButtons = useTitlebarLayout((s) => s.layout.left.length)
  const linuxControlsPad =
    !IS_MACOS && !open && leftButtons > 0 ? leftButtons * 32 + 24 : undefined
  // Right-anchored controls (Windows, and Linux DEs that place them there) sit at
  // right-4 above the header — keep header content clear of them at every width.
  const rightButtons = useTitlebarLayout((s) => s.layout.right.length)
  const rightControlsPad = rightButtons > 0 ? rightButtons * 32 + 24 : undefined

  return (
    <div
      className={cn(
        // Hug the top on the same rhythm as the sidebar's inner header row
        // (8px gutter + 1px border + 4px padding -> row center ~27px abs), so
        // every page's header lines up with the "Jan" row beside it.
        'h-15 flex items-start shrink-0 pt-1',
        (IS_MACOS && !open) ? 'pl-24' : 'pl-4',
        children === undefined && 'border-none',
        // With no content and the sidebar open there is nothing in this row
        // but the drag strip, so give the page back the height. Collapsed, the
        // row still owns the sidebar toggle.
        children === undefined && open && 'h-4',
        className
      )}
      style={{
        ...(linuxControlsPad ? { paddingLeft: linuxControlsPad } : {}),
        ...(rightControlsPad ? { paddingRight: rightControlsPad } : {}),
      }}
    >
      <div
        className={cn(
          'flex items-center w-full gap-3',
        )}
      >
        {!open && !isLocked && (
          <div className="shrink-0">
            <Button
              variant="ghost"
              size="icon-sm"
              className='h-8 w-8 rounded-full relative z-50'
              onClick={() => setLeftPanel(!open)}
              aria-label="Toggle sidebar"
            >
              <IconLayoutSidebar
                className="text-muted-foreground relative size-4.5"
              />
            </Button>
          </div>
        )}
        <div className={cn('flex-1 min-w-0', bleed && open && '-ml-4')}>
          {children}
        </div>
      </div>
    </div>
  )
})

export default HeaderPage

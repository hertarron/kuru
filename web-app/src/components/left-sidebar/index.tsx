import { DownloadManagement } from '@/containers/DownloadManegement'
import { NavChats } from './NavChats'
import { NavMain } from './NavMain'
import { NavProjects } from './NavProjects'

import {
  Sidebar,
  SidebarContent,
  SidebarTrigger,
  SidebarHeader,
  SidebarRail,
  useSidebar,
} from '@/components/ui/sidebar'
import { cn } from '@/lib/utils'
import { useSidebarLocked } from '@/lib/sidebar-routes'
import { useTitlebarLayout } from '@/stores/titlebar-layout-store'

export function LeftSidebar() {
  // Settings/hub lock the sidebar expanded; outside those, the download stack
  // only fits the expanded header (it is hidden on the collapsed icon rail).
  const locked = useSidebarLocked()
  const { state } = useSidebar()
  // Right-align the header when native controls own the top-left (macOS, or a Linux
  // DE placing buttons left); "Jan" moves into the right cluster except on macOS.
  const leftButtons = useTitlebarLayout((s) => s.layout.left.length)
  const controlsOnLeft = !IS_MACOS && leftButtons > 0
  const reserveLeft = IS_MACOS || controlsOnLeft
  return (
    <div className='relative z-50'>
      <Sidebar variant="floating" collapsible="offcanvas">
        <SidebarHeader className="flex px-1 py-1">
          <div className={cn("flex items-center w-full justify-between", reserveLeft && "justify-end")}>
            {!reserveLeft && <span className="ml-2 font-medium font-studio">Kuru</span>}
            {/* h-8 pins the row height to the trigger's size so dropping the
                trigger on locked routes doesn't shift the header rhythm. */}
            <div className="flex items-center h-8">
              {controlsOnLeft && (
                <span className="mr-2 font-medium font-studio">Kuru</span>
              )}
              {state !== 'collapsed' && <DownloadManagement />}
              {/* Locked routes (settings/hub) drop the collapse trigger, so the
                  download button sits flush at the right edge there. */}
              {!locked && (
                <SidebarTrigger className="text-muted-foreground rounded-full hover:bg-sidebar-foreground/8! -mt-0.5 relative z-50 ml-0.5" />
              )}
            </div>
          </div>
          <NavMain />
        </SidebarHeader>
        <SidebarContent className="mask-b-from-95% mask-t-from-98%">
          <NavProjects />
          <NavChats />
        </SidebarContent>
        {!locked && <SidebarRail />}
      </Sidebar>
    </div>
  )
}

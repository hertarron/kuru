import { useLocation } from '@tanstack/react-router'

/**
 * The left sidebar is only collapsible inside chat (home, threads, project,
 * character and lorebook pages). Settings and hub pages lock it expanded: no
 * collapse trigger, no rail, and a persisted collapsed state is ignored while
 * there — but left untouched so returning to chat restores it.
 */
export function isSidebarLockedPath(pathname: string): boolean {
  return pathname.startsWith('/settings') || pathname.startsWith('/hub')
}

/**
 * Reactive to client-side navigation. Don't read window.location.pathname at
 * render time: React would not re-render on history pushes, leaving the value
 * stale after navigating between chat and settings/hub.
 */
export function useSidebarLocked(): boolean {
  return isSidebarLockedPath(useLocation().pathname)
}
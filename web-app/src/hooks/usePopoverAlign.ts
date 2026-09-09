import { useCallback, useRef, useState } from 'react'

/**
 * Popover alignment that follows the pill: hug the pill's left edge when the
 * panel fits to the right of it, otherwise hug its right edge.
 *
 * Radix can't express this on its own -- align="start" plus collision
 * avoidance only slides the panel left until it clears the window edge,
 * which strands it away from the pill instead of flipping it around.
 *
 * Attach pillRef to the element the popover anchors to (the pill wrapper,
 * which doubles as the PopoverTrigger via asChild, like the model pill).
 */
export function usePopoverAlign(panelWidth: number) {
  const pillRef = useRef<HTMLDivElement | null>(null)
  const [align, setAlign] = useState<'start' | 'end'>('start')

  // Called as the popover opens, while the pill is still measurable.
  const measure = useCallback(() => {
    const el = pillRef.current
    if (!el) return
    const MARGIN = 16
    const fitsRight =
      el.getBoundingClientRect().left + panelWidth <=
      window.innerWidth - MARGIN
    setAlign(fitsRight ? 'start' : 'end')
  }, [panelWidth])

  return { pillRef, align, measure }
}

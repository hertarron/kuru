/**
 * The centered content column shared by the chat view and the Hub.
 *
 * A max width rather than a fraction of the window: narrow windows get the
 * whole width bar a small gutter, and only once there is room to spare does
 * margin appear. Everything that reads as "the column of cards" — models,
 * characters, lorebooks, messages, the chat box — uses this so they line up
 * with each other at every window size.
 */
export const CONTENT_COLUMN = 'mx-auto w-full max-w-[1360px] px-4'

/**
 * The Hub's column. Same stretch as CONTENT_COLUMN -- before they were unified
 * the chat stopped at 1024px while the card grid ran to 1360px, which read as
 * two different apps on a wide display. Kept as a named alias because the Hub
 * pages think in terms of the card grid, not the chat column.
 */
export const HUB_COLUMN = CONTENT_COLUMN

/**
 * Cards size themselves rather than being cut from a fixed column count, so a
 * row of two and a row of four hold the same card width.
 */
export const CARD_GRID = 'grid gap-4 grid-cols-[repeat(auto-fill,minmax(288px,1fr))]'

/**
 * Top-edge fade for a scrolling column: clear at the very top, solid 24px
 * down. Applied as a mask so the content's own pixels dissolve, whatever
 * colour sits behind them.
 *
 * A mask clips `position: fixed` descendants to the masked box, so anything
 * inside a masked column that has to cover the window -- the message image
 * lightbox, for one -- has to be portalled out.
 */
export const FADE_TOP_MASK = 'linear-gradient(to bottom, transparent 0, black 48px)'

/**
 * Top padding for the content inside a faded column. Scrolled to the top, the
 * first line has to clear both the fade and the window's fixed h-12 drag strip
 * -- otherwise the text you scrolled up to read arrives half dissolved and
 * unselectable. Scrolling further still carries content up through the fade.
 */
export const FADE_TOP_CLEARANCE = 'pt-10'

/**
 * Chevrons are the last thing a pill gives up, once its label has already
 * gone and only the avatar, logo or icon is left. Label widths are not a
 * breakpoint problem and are measured instead: see `PillRow`.
 */
export const PILL_CHEVRON = '@max-sm:hidden'

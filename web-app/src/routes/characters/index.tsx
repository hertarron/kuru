import { createFileRoute, redirect } from '@tanstack/react-router'
import { route } from '@/constants/routes'

/**
 * Characters discovery moved into the Hub alongside models and lorebooks.
 * Kept as a redirect so old deep links and the character pill's
 * "browse characters" action keep working.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.characters.index as any)({
  beforeLoad: () => {
    throw redirect({ to: route.hub.characters })
  },
})

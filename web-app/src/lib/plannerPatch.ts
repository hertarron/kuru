import { useModelProvider } from '@/hooks/useModelProvider'

/**
 * Writes a planner patch to one llama.cpp model, store first and then yaml.
 *
 * Both writes matter and their order does: the router preset is built from
 * model.yml, and the sheet reads the store, so a patch that reaches only one
 * of them loads with settings the UI does not show or shows settings it does
 * not load. The settings sheet does the same two writes in the same order;
 * this is that sequence for the paths with no sheet in front of them — the
 * send-time refit and the background fit-test scheduler.
 *
 * Silent on failure: every caller is a background convenience whose absence
 * leaves the stored settings untouched and the normal path working.
 */
export async function applyPlannerPatch(
  modelId: string,
  patch: Record<string, string | number | boolean>,
  updateModelSettings?: (
    id: string,
    patch: Record<string, string | number | boolean>
  ) => Promise<void>
): Promise<boolean> {
  if (Object.keys(patch).length === 0) return false

  const provider = useModelProvider.getState().getProviderByName('llamacpp')
  const index = provider?.models?.findIndex((m) => m.id === modelId) ?? -1
  if (!provider || index === -1) return false

  const target = provider.models[index] as { settings?: Record<string, unknown> }
  const settings: Record<string, unknown> = { ...(target.settings ?? {}) }
  for (const [key, value] of Object.entries(patch)) {
    const existing = settings[key]
    settings[key] = {
      ...(typeof existing === 'object' && existing !== null ? existing : {}),
      key,
      controller_props: {
        ...((existing as { controller_props?: object } | undefined)
          ?.controller_props ?? {}),
        value,
      },
    }
  }
  const models = [...provider.models]
  models[index] = { ...target, settings } as (typeof provider.models)[number]
  useModelProvider.getState().updateProvider('llamacpp', { models })

  if (typeof updateModelSettings === 'function') {
    await updateModelSettings(modelId, patch).catch(() => undefined)
  }
  return true
}

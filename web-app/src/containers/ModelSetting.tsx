import { IconChevronRight, IconSettings } from '@tabler/icons-react'
import debounce from 'lodash.debounce'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { ContextPlanPanel } from '@/containers/ContextPlanPanel'
import { DynamicControllerSetting } from '@/containers/dynamicControllerSetting'
import { SamplerDefaults } from '@/containers/SamplerDefaults'
import { ChatTemplateKwargs } from '@/containers/ChatTemplateKwargs'
import { useModelProvider } from '@/hooks/useModelProvider'
import { useServiceHub } from '@/hooks/useServiceHub'
import { cn, getModelDisplayName } from '@/lib/utils'
import { useTranslation } from '@/i18n/react-i18next-compat'
import { useAppState } from '@/hooks/useAppState'
import { paramsSettings, samplerKeysForProvider } from '@/lib/predefinedParams'

const MTP_MIN_BUILD = 9193

/** Surfaced by the planner panel, so never repeated in Advanced. */
const PLANNER_PANEL_KEYS = [
  'ctx_len',
  'cache_type_k',
  'cache_type_v',
  'batch_size',
  'ubatch_size',
  'device',
  'tensor_split',
  'split_mode',
]

/** Settings the context planner writes. Editing one by hand turns it off. */
const PLANNER_OWNED_KEYS = [
  'ngl',
  'override_tensor_buffer_t',
  'cpu_moe',
  'n_cpu_moe',
]

/** Settings llama.cpp only reads at load, so a running model must restart. */
const RESTART_ON_CHANGE_KEYS = [
  'ctx_len',
  'ngl',
  'chat_template',
  'offload_mmproj',
  'batch_size',
  'ubatch_size',
  'cpu_moe',
  'n_cpu_moe',
  'cache_type_k',
  'cache_type_v',
  'flash_attn',
  'override_tensor_buffer_t',
  'device',
  'tensor_split',
]

function parseBuildNumber(version: unknown): number | null {
  if (typeof version !== 'string') return null
  const m = version.match(/^b(\d+)$/)
  return m ? parseInt(m[1], 10) : null
}

type ModelSettingProps = {
  provider: ProviderObject
  model: Model
}

export function ModelSetting({
  model,
  provider,
}: ModelSettingProps) {
  const { updateProvider } = useModelProvider()
  const { t } = useTranslation()
  const serviceHub = useServiceHub()
  const setActiveModels = useAppState((state) => state.setActiveModels)

  // Debounced stopModel — memoized so the timer survives across renders.
  // Without useMemo a fresh debounce is created every keystroke and the
  // cleanup effect below cancels the previous timer before it can fire.
  const debouncedStopModel = useMemo(
    () =>
      debounce((modelId: string) => {
        serviceHub
          .models()
          .stopModel(modelId)
          .then(() => {
            serviceHub
              .models()
              .getActiveModels()
              .then((models) => setActiveModels(models || []))
          })
      }, 500),
    [serviceHub, setActiveModels]
  )

  // Coalesce rapid sidebar edits into a single yaml-write + router-restart.
  // Each call merges into a per-model accumulator; the debounced flush writes
  // and clears it. Avoids a flurry of slider drags rewriting model.yml + bouncing
  // the router on every keystroke.
  const pendingPatchesRef = useRef<
    Map<string, Record<string, string | number | boolean | null | undefined>>
  >(new Map())
  const rollbackSettingsRef = useRef<Map<string, Model['settings'] | undefined>>(
    new Map()
  )
  const flushPendingPatches = useMemo(
    () =>
      debounce(() => {
        const patches = pendingPatchesRef.current
        pendingPatchesRef.current = new Map()
        for (const [modelId, patch] of patches) {
          serviceHub
            .models()
            .updateModelSettings(modelId, patch)
            .then(() => {
              rollbackSettingsRef.current.delete(modelId)
            })
            .catch((e) => {
              const previousSettings = rollbackSettingsRef.current.get(modelId)
              rollbackSettingsRef.current.delete(modelId)
              const latestProvider = useModelProvider
                .getState()
                .getProviderByName(provider.provider)
              if (latestProvider) {
                const modelIndex = latestProvider.models.findIndex(
                  (m) => m.id === modelId
                )
                if (modelIndex !== -1) {
                  const revertedModels = [...latestProvider.models]
                  revertedModels[modelIndex] = {
                    ...revertedModels[modelIndex],
                    settings: previousSettings,
                  } as Model
                  useModelProvider.getState().updateProvider(provider.provider, {
                    models: revertedModels,
                  })
                }
              }
              console.error('Failed to persist model settings', e)
            })
        }
      }, 1500),
    [provider.provider, serviceHub]
  )
  const debouncedPersistModelSettings = (
    modelId: string,
    patch: Record<string, string | number | boolean | null | undefined>,
    previousSettings: Model['settings'] | undefined
  ) => {
    if (!rollbackSettingsRef.current.has(modelId)) {
      rollbackSettingsRef.current.set(modelId, previousSettings)
    }
    const existing = pendingPatchesRef.current.get(modelId) ?? {}
    pendingPatchesRef.current.set(modelId, { ...existing, ...patch })
    flushPendingPatches()
  }

  useEffect(() => {
    return () => {
      // Flush — not cancel — so a pending ctx_len/ngl edit still gets
      // written to model.yml and the router restarted after the sheet closes.
      flushPendingPatches.flush()
      debouncedStopModel.flush()
    }
  }, [debouncedStopModel, flushPendingPatches])

  const handleSettingChange = (
    key: string,
    value: string | boolean | number
  ) => handleSettingsChange({ [key]: value })

  /**
   * Applies several settings as one update. The planner writes context and
   * placement together, and one key at a time would drop all but the last:
   * each write spreads `model.settings` as it was when the handler was made.
   */
  const handleSettingsChange = (
    patch: Record<string, string | boolean | number>
  ) => {
    if (!provider) return

    const updatedSettings = { ...model.settings }
    for (const [key, value] of Object.entries(patch)) {
      updatedSettings[key] = {
        key,
        ...(model.settings?.[key] != null ? model.settings?.[key] : {}),
        controller_props: {
          ...(model.settings?.[key]?.controller_props ?? {}),
          value: value,
        },
      } as ProviderSetting
    }
    const updatedModel = { ...model, settings: updatedSettings }

    // Find the model index in the provider's models array
    const modelIndex = provider.models.findIndex((m) => m.id === model.id)

    if (modelIndex !== -1) {
      // Create a copy of the provider's models array
      const updatedModels = [...provider.models]

      // Update the specific model in the array
      updatedModels[modelIndex] = updatedModel as Model

      // Update the provider with the new models array
      updateProvider(provider.provider, {
        models: updatedModels,
      })

      // Call debounced stopModel only when updating settings that require restart,
      // and only if the model is currently running
      if (
        Object.keys(patch).some((key) =>
          RESTART_ON_CHANGE_KEYS.includes(key)
        )
      ) {
        // Check if model is running before stopping it
        serviceHub
          .models()
          .getActiveModels()
          .then((activeModels) => {
            if (activeModels.includes(model.id)) {
              debouncedStopModel(model.id)
            }
          })
      }

      // In router mode the router reads args from `router.preset.ini` (built
      // from `model.yml`) — Zustand alone has no effect on inference. Persist
      // mappable keys to disk + restart the router so the next load uses the
      // new args. Non-mappable keys are filtered inside the extension.
      if (provider.provider === 'llamacpp') {
        debouncedPersistModelSettings(model.id, patch, model.settings)
      }
    }
  }

  // Chat-template kwargs are a per-request concern (sent as chat_template_kwargs),
  // not a router preset arg — persist to the store only, like `reasoning`.
  const handleTemplateKwargsChange = (
    value: Record<string, boolean | number | string>
  ) => {
    if (!provider) return
    const modelIndex = provider.models.findIndex((m) => m.id === model.id)
    if (modelIndex === -1) return
    const existing = model.settings?.chat_template_kwargs
    const updatedModel = {
      ...model,
      settings: {
        ...model.settings,
        chat_template_kwargs: {
          key: 'chat_template_kwargs',
          title: 'Chat template options',
          description: '',
          controller_type: 'object',
          ...(existing ?? {}),
          controller_props: {
            ...(existing?.controller_props ?? {}),
            value,
          },
        },
      },
    } as unknown as Model
    const updatedModels = [...provider.models]
    updatedModels[modelIndex] = updatedModel
    updateProvider(provider.provider, { models: updatedModels })
  }

  const [advancedOpen, setAdvancedOpen] = useState(false)

  const planManual =
    model.settings?.plan_manual?.controller_props?.value === true

  const handleAdvancedChange = (
    key: string,
    value: string | boolean | number
  ) => {
    if (!planManual && PLANNER_OWNED_KEYS.includes(key)) {
      handleSettingsChange({ [key]: value, plan_manual: true })
      return
    }
    handleSettingChange(key, value)
  }

  const plannerActive =
    provider.provider === 'llamacpp' && model.embedding !== true

  // llama.cpp needs flash attention for a quantized V cache. While one is
  // selected the Off switch is unavailable, with the reason on the wrapper
  // (the same treatment `DynamicControllerSetting` gives `disabledReason`).
  const effectiveCacheTypeV =
    (model.settings?.cache_type_v?.controller_props?.value as
      | string
      | undefined) ??
    (provider.settings?.find((s) => s.key === 'cache_type_v')
      ?.controller_props?.value as string | undefined) ??
    'f16'
  const flashOffBlocked = effectiveCacheTypeV !== 'f16'
  const flashOffReason =
    'A quantized K/V cache needs Flash Attention on. Return to fp16 to turn it off.'

  // Kuru Fit is opt-out: installs predating the key behave as fitted.
  // Legacy mode restores stock Jan fitting and strips every Kuru behavior.
  const kuruMode =
    provider.provider !== 'llamacpp' ||
    provider.settings?.find((s) => s.key === 'kuru_fit')?.controller_props
      ?.value !== false

  const fitEnabled =
    provider.settings?.find((s) => s.key === 'fit')?.controller_props
      ?.value === true
  const fitCtxSetting = provider.settings?.find((s) => s.key === 'fit_ctx')

  const handleEngineSettingChange = (
    key: string,
    value: string | boolean | number
  ) => {
    if (!provider) return
    const newSettings = provider.settings.map((s) =>
      s.key === key
        ? {
            ...s,
            controller_props: { ...s.controller_props, value },
          }
        : s
    )
    serviceHub.providers().updateSettings(provider.provider, newSettings)
    updateProvider(provider.provider, { settings: newSettings })
  }

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon-xs">
          <IconSettings size={18} className="text-muted-foreground" />
        </Button>
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>
            {t('common:modelSettings.title', {
              modelId: getModelDisplayName(model),
            })}
          </SheetTitle>
          <SheetDescription className='text-xs leading-normal'>
            {t('common:modelSettings.description')}
          </SheetDescription>
        </SheetHeader>

        <div className="px-4 space-y-8 pb-4 flex-1 min-h-0 overflow-y-auto">
          {provider.provider === 'llamacpp' && (
            <MtpPanel modelId={model.id} provider={provider} />
          )}
          {provider.provider === 'llamacpp' && model.embedding !== true && (
            <ChatTemplateKwargs
              model={model}
              onChange={handleTemplateKwargsChange}
            />
          )}
          {(provider.provider === 'llamacpp' || provider.provider === 'mlx') &&
            model.embedding !== true && (
              <SamplerDefaults
                model={model}
                keys={samplerKeysForProvider(provider.provider)}
                onChange={handleSettingChange}
              />
            )}
          {provider.provider === 'llamacpp' &&
            model.embedding !== true &&
            kuruMode && (
              <ContextPlanPanel
                model={model}
                manual={planManual}
                onChange={handleSettingsChange}
                onResume={() => handleSettingChange('plan_manual', false)}
              />
            )}
          {provider.provider === 'llamacpp' &&
            model.embedding !== true &&
            !kuruMode &&
            fitEnabled &&
            fitCtxSetting && (
              <div key="fit_ctx" className="space-y-2">
                <div>
                  <span
                    title={fitCtxSetting.title}
                    className="font-medium"
                  >
                    {fitCtxSetting.title}
                  </span>
                </div>
                <DynamicControllerSetting
                  key={fitCtxSetting.key}
                  title={fitCtxSetting.title}
                  description={fitCtxSetting.description}
                  controllerType={fitCtxSetting.controller_type}
                  controllerProps={fitCtxSetting.controller_props}
                  onChange={(newValue) =>
                    handleEngineSettingChange('fit_ctx', newValue)
                  }
                />
                <p className="text-muted-foreground leading-normal text-xs">
                  {fitCtxSetting.description}
                </p>
              </div>
            )}
          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <CollapsibleTrigger className="flex w-full items-center gap-2 rounded py-1 text-left font-medium hover:bg-secondary/50">
              <IconChevronRight
                size={14}
                className={cn('transition-transform', advancedOpen && 'rotate-90')}
              />
              Advanced
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-8 pt-4">
          {provider.provider === 'llamacpp' &&
            model.embedding !== true &&
            kuruMode && (
              <div className="space-y-3">
                <div>
                  <span className="font-medium">Flash Attention</span>
                  <p className="text-muted-foreground leading-normal text-xs">
                    Faster attention kernels using less memory. Auto decides
                    per load; a measured model re-measures on change.
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-1">
                  {(
                    [
                      { value: 'auto', label: 'Auto' },
                      { value: 'on', label: 'On' },
                      { value: 'off', label: 'Off' },
                    ] as const
                  ).map((mode) =>
                    mode.value === 'off' && flashOffBlocked ? (
                      <div
                        key={mode.value}
                        className="opacity-50 pointer-events-none select-none"
                        title={flashOffReason}
                        aria-disabled
                      >
                        <Button
                          size="sm"
                          className="w-full"
                          variant={
                            (
                              (model.settings?.flash_attn?.controller_props
                                ?.value as string) || 'auto'
                            ) === mode.value
                              ? 'default'
                              : 'ghost'
                          }
                          disabled
                        >
                          {mode.label}
                        </Button>
                      </div>
                    ) : (
                      <Button
                        key={mode.value}
                        size="sm"
                        variant={
                          (
                            (model.settings?.flash_attn?.controller_props
                              ?.value as string) || 'auto'
                          ) === mode.value
                            ? 'default'
                            : 'ghost'
                        }
                        onClick={() => handleSettingChange('flash_attn', mode.value)}
                      >
                        {mode.label}
                      </Button>
                    )
                  )}
                </div>
              </div>
            )}
          {(() => {
            return Object.entries(model.settings || {})
          .reduce<[string, unknown][]>((acc, entry) => {
            if (entry[0] === 'reasoning') return acc
            // Rendered by the dedicated ChatTemplateKwargs section above.
            if (entry[0] === 'chat_template_kwargs') return acc
            // Removed in v15 migration; defend against any pre-migration
            // localStorage state that still carries the orphan entry.
            if (entry[0] === 'auto_increase_ctx_len') return acc
            // Stock Jan fitting owns context sizing and GPU layers, so the
            // request field has nowhere to live while it is on.
            if (!kuruMode && fitEnabled && entry[0] === 'ctx_len') return acc
            // Owned by the context planner panel above. The layer counts stay
            // here in every mode: editing one takes manual control (see
            // handleAdvancedChange), which is the only door into it.
            if (kuruMode && plannerActive && PLANNER_PANEL_KEYS.includes(entry[0])) return acc
            if (entry[0] === 'plan_manual') return acc
            // Planner-written keys arrive as bare {key, controller_props}
            // entries with no title or control, and would render as an
            // unlabelled switch. In manual mode they still need controls,
            // so the map below infers the obvious one from the stored value.
            if (
              !(entry[1] as ProviderSetting)?.controller_type &&
              !(
                kuruMode &&
                planManual &&
                PLANNER_OWNED_KEYS.includes(entry[0])
              )
            )
              return acc
            // Sampling params live in the composer popover, not the sidebar.
            if (entry[0] in paramsSettings) return acc
            acc.push(entry)
            return acc
          }, [])
          .filter(([key]) => {
            // MLX models only support context size setting
            if (provider.provider === 'mlx') {
              return key === 'ctx_len'
            }
            return true
          })
          .map(([key, value]) => {
            const raw = value as ProviderSetting
            // Bare planner-written keys get an inferred control in manual
            // mode (see the filter above); anywhere else they stay hidden.
            const storedValue = (
              raw as {
                controller_props?: { value?: unknown }
              }
            )?.controller_props?.value
            const config: ProviderSetting =
              raw?.controller_type
                ? raw
                : ({
                    key,
                    title: key,
                    controller_type:
                      typeof storedValue === 'boolean'
                        ? 'checkbox'
                        : 'input',
                    controller_props: {
                      ...(typeof storedValue === 'number'
                        ? { type: 'number' }
                        : {}),
                      value: storedValue,
                    },
                  } as ProviderSetting)
            const takesManual =
              kuruMode &&
              !planManual &&
              PLANNER_OWNED_KEYS.includes(key)
            return (
              <div key={key} className="space-y-2">
                <div
                  className={cn(
                    'flex items-start justify-between gap-8',
                    (key === 'chat_template' ||
                      key === 'override_tensor_buffer_t' ||
                      config.controller_type === 'dropdown') &&
                      'flex-col gap-1 w-full'
                  )}
                >
                  <div className="mb-1 truncate">
                    <span title={config.title} className="font-medium">{config.title}</span>
                  </div>
                  <DynamicControllerSetting
                    key={config.key}
                    title={config.title}
                    description={config.description}
                    controllerType={config.controller_type}
                    disabledReason={
                      !kuruMode && fitEnabled && key === 'ngl'
                        ? t('common:modelSettings.nglDisabledByFit')
                        : undefined
                    }
                    controllerProps={{
                      ...config.controller_props,
                      value: config.controller_props?.value,
                    }}
                    onChange={(newValue) => handleAdvancedChange(key, newValue)}
                  />
                </div>
                <p className="text-muted-foreground leading-normal text-xs">
                  {config.description}
                  {takesManual &&
                    ' Editing takes manual control of placement.'}
                </p>
              </div>
            )
          })
          })()}
            </CollapsibleContent>
          </Collapsible>
        </div>
      </SheetContent>
    </Sheet>
  )
}

type MtpInfo = {
  mtp_layers: number
  mtp: boolean
  spec_draft_n_max?: number
  spec_draft_n_min?: number
  spec_draft_p_min?: number
}

function MtpPanel({
  modelId,
  provider,
}: {
  modelId: string
  provider: ProviderObject
}) {
  const { t } = useTranslation()
  const serviceHub = useServiceHub()
  const [info, setInfo] = useState<MtpInfo | null>(null)

  useEffect(() => {
    let active = true
    serviceHub
      .models()
      .getMtpInfo(modelId)
      .then((v) => {
        if (active) setInfo(v)
      })
      .catch(() => {
        if (active) setInfo({ mtp_layers: 0, mtp: false })
      })
    return () => {
      active = false
    }
  }, [modelId, serviceHub])

  const llamacppVersion = provider.settings?.find(
    (s) => s.key === 'llamacpp_version'
  )?.controller_props?.value as string | undefined
  const buildNo = parseBuildNumber(llamacppVersion)
  const backendSupports = buildNo !== null && buildNo >= MTP_MIN_BUILD

  const persist = useCallback(
    async (patch: {
      mtp?: boolean
      spec_draft_n_max?: number | null
      spec_draft_n_min?: number | null
      spec_draft_p_min?: number | null
    }) => {
      try {
        await serviceHub.models().updateMtpSettings(modelId, patch)
      } catch (e) {
        console.error('Failed to update MTP settings', e)
      }
    },
    [modelId, serviceHub]
  )

  if (!info || info.mtp_layers <= 0) return null

  const enabled = info.mtp === true && backendSupports

  const updateNumber = (
    key: 'spec_draft_n_max' | 'spec_draft_n_min' | 'spec_draft_p_min',
    raw: string
  ) => {
    const trimmed = raw.trim()
    if (trimmed.length === 0) {
      setInfo({ ...info, [key]: undefined })
      void persist({ [key]: null })
      return
    }
    const n = Number(trimmed)
    if (!Number.isFinite(n)) return
    setInfo({ ...info, [key]: n })
    void persist({ [key]: n })
  }

  return (
    <div className="space-y-3">
      <div className="font-medium">
        {t('common:modelSettings.mtp.section')}
      </div>

      <div className="space-y-2">
        <div className="flex items-start justify-between gap-8">
          <div className="mb-1 truncate">
            <span className="font-medium">
              {t('common:modelSettings.mtp.enable')}
            </span>
          </div>
          <Switch
            checked={enabled}
            disabled={!backendSupports}
            onCheckedChange={(v) => {
              setInfo({ ...info, mtp: v })
              void persist({ mtp: v })
            }}
          />
        </div>
        <p className="text-muted-foreground leading-normal text-xs">
          {backendSupports
            ? t('common:modelSettings.mtp.enableDescription')
            : t('common:modelSettings.mtp.needsUpgrade')}
        </p>
      </div>

      {enabled && (
        <>
          <NumberRow
            label={t('common:modelSettings.mtp.nMax')}
            description={t('common:modelSettings.mtp.nMaxDescription')}
            placeholder="16"
            value={info.spec_draft_n_max}
            min={1}
            step={1}
            onChange={(raw) => updateNumber('spec_draft_n_max', raw)}
          />
          <NumberRow
            label={t('common:modelSettings.mtp.nMin')}
            description={t('common:modelSettings.mtp.nMinDescription')}
            placeholder="0"
            value={info.spec_draft_n_min}
            min={0}
            step={1}
            onChange={(raw) => updateNumber('spec_draft_n_min', raw)}
          />
          <NumberRow
            label={t('common:modelSettings.mtp.pMin')}
            description={t('common:modelSettings.mtp.pMinDescription')}
            placeholder="0.75"
            value={info.spec_draft_p_min}
            min={0}
            max={1}
            step={0.05}
            onChange={(raw) => updateNumber('spec_draft_p_min', raw)}
          />
        </>
      )}
    </div>
  )
}

function NumberRow({
  label,
  description,
  placeholder,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  description: string
  placeholder: string
  value: number | undefined
  min?: number
  max?: number
  step?: number
  onChange: (raw: string) => void
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-start justify-between gap-8">
        <div className="mb-1 truncate">
          <span className="font-medium">{label}</span>
        </div>
        <Input
          type="number"
          className="w-32"
          placeholder={placeholder}
          value={value ?? ''}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
      <p className="text-muted-foreground leading-normal text-xs">
        {description}
      </p>
    </div>
  )
}

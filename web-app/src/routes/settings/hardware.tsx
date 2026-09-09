import { createFileRoute } from '@tanstack/react-router'
import { route } from '@/constants/routes'
import SettingsMenu from '@/containers/SettingsMenu'
import HeaderPage from '@/containers/HeaderPage'
import { Card, CardItem } from '@/containers/Card'
import { Switch } from '@/components/ui/switch'
import { Progress } from '@/components/ui/progress'
import { useTranslation } from '@/i18n/react-i18next-compat'
import {
  gpuMemoryUsage,
  isDisplayGpu,
  resolveGpuReserveMiB,
  useHardware,
} from '@/hooks/useHardware'
import { useLlamacppDevices } from '@/hooks/useLlamacppDevices'
import { useEffect, useState } from 'react'
import { IconDeviceDesktopAnalytics } from '@tabler/icons-react'
import { useServiceHub } from '@/hooks/useServiceHub'
import type { HardwareData, SystemUsage } from '@/services/hardware/types'
import { cn, formatMegaBytes } from '@/lib/utils'
import { toNumber } from '@/utils/number'
import { useModelProvider } from '@/hooks/useModelProvider'
import { useAppState } from '@/hooks/useAppState'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { DEFAULT_VRAM_SAFETY_MARGIN_BYTES } from '@/lib/contextPlanner'

/** VRAM a model can use on one card, in MiB. Mirrors `freeVramByGpu`. */
function plannableMiB(
  device: { id: string; name: string; mem: number; free: number },
  usage: SystemUsage,
  hardware: HardwareData,
  reserves: Record<string, number>
): number {
  const used =
    gpuMemoryUsage(usage, hardware, device.name)?.used ??
    Math.max(0, device.mem - device.free)
  const margin = DEFAULT_VRAM_SAFETY_MARGIN_BYTES / (1024 * 1024)
  const reserve = resolveGpuReserveMiB(usage, hardware, device, reserves)
  return Math.max(0, device.mem - used - margin - reserve)
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Route = createFileRoute(route.settings.hardware as any)({
  component: HardwareContent,
})

function HardwareContent() {
  const { t } = useTranslation()
  const [isLoading, setIsLoading] = useState(false)
  const serviceHub = useServiceHub()
  const {
    hardwareData,
    systemUsage,
    setHardwareData,
    updateSystemUsage,
    pollingPaused,
    gpuReserveMiB,
    setGpuReserve,
  } = useHardware()
  const setActiveModels = useAppState((state) => state.setActiveModels)

  const { providers } = useModelProvider()
  const llamacpp = providers.find((p) => p.provider === 'llamacpp')

  // Llamacpp devices hook
  const llamacppDevicesResult = useLlamacppDevices()

  // Use default values on macOS since llamacpp devices are not relevant
  const {
    devices: llamacppDevices,
    loading: llamacppDevicesLoading,
    error: llamacppDevicesError,
    toggleDevice,
    fetchDevices,
  } = IS_MACOS
    ? {
        devices: [],
        loading: false,
        error: null,
        toggleDevice: () => {},
        fetchDevices: () => {},
      }
    : llamacppDevicesResult

  // Fetch llamacpp devices when component mounts
  useEffect(() => {
    fetchDevices()
  }, [fetchDevices])

  // Fetch initial hardware info and system usage
  useEffect(() => {
    setIsLoading(true)
    Promise.all([
      serviceHub
        .hardware()
        .getHardwareInfo()
        .then((data: HardwareData | null) => {
          if (data) setHardwareData(data)
        })
        .catch((error) => {
          console.error('Failed to get hardware info:', error)
        }),
      serviceHub
        .hardware()
        .getSystemUsage()
        .then((data: SystemUsage | null) => {
          if (data) updateSystemUsage(data)
        })
        .catch((error: unknown) => {
          console.error('Failed to get initial system usage:', error)
        }),
    ]).finally(() => {
      setIsLoading(false)
    })
  }, [serviceHub, setHardwareData, updateSystemUsage])

  useEffect(() => {
    if (pollingPaused) {
      return
    }
    const intervalId = setInterval(() => {
      serviceHub
        .hardware()
        .getSystemUsage()
        .then((data: SystemUsage | null) => {
          if (data) updateSystemUsage(data)
        })
        .catch((error: unknown) => {
          console.error('Failed to get system usage:', error)
        })
    }, 5000)

    return () => clearInterval(intervalId)
  }, [serviceHub, updateSystemUsage, pollingPaused])

  const handleClickSystemMonitor = async () => {
    try {
      await serviceHub.window().openSystemMonitorWindow()
    } catch (error) {
      console.error('Failed to open system monitor window:', error)
    }
  }

  const handleRefreshHardware = async () => {
    try {
      setIsLoading(true)
      await serviceHub.hardware().refreshHardwareInfo()
      const [hardwareData, systemUsage] = await Promise.all([
        serviceHub.hardware().getHardwareInfo(),
        serviceHub.hardware().getSystemUsage(),
      ])
      if (hardwareData) setHardwareData(hardwareData)
      if (systemUsage) updateSystemUsage(systemUsage)
      if (!IS_MACOS) fetchDevices()
    } catch (error) {
      console.error('Failed to refresh hardware:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full w-full">
      <HeaderPage className="h-auto pt-[calc(var(--spacing)*1.7)] pb-[calc(var(--spacing)*2.5)]">
        <div className={cn("flex items-center justify-between w-full mr-2 pr-3", !IS_MACOS && "pr-30")}>
          <span className='font-medium text-base font-studio'>{t('common:settings')}</span>
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-2 relative z-50 -my-1"
            onClick={handleClickSystemMonitor}
          >
            <IconDeviceDesktopAnalytics className="text-muted-foreground size-5" />
            <p>{t('settings:hardware.systemMonitor')}</p>
          </Button>
        </div>
      </HeaderPage>
      <div className="flex flex-1 min-h-0">
        <SettingsMenu />
        <div className="p-4 pt-0 w-full overflow-y-auto">
          {isLoading ? (
            <div className="flex items-center justify-center h-32">
              <div className="text-muted-foreground">
                Loading hardware information...
              </div>
            </div>
          ) : (
            <div className="flex flex-col justify-between gap-4 gap-y-3 w-full">
              {/* OS Information */}
              <Card title={t('settings:hardware.os')}>
                <CardItem
                  title={t('settings:hardware.name')}
                  actions={
                    <span className="text-foreground capitalize">
                      {hardwareData.os_type}
                    </span>
                  }
                />
                <CardItem
                  title={t('settings:hardware.version')}
                  actions={
                    <span className="text-foreground">
                      {hardwareData.os_name}
                    </span>
                  }
                />
              </Card>

              {/* CPU Information */}
              <Card title={t('settings:hardware.cpu')}>
                <CardItem
                  title={t('settings:hardware.model')}
                  actions={
                    <span className="text-foreground">
                      {hardwareData.cpu?.name}
                    </span>
                  }
                />
                <CardItem
                  title={t('settings:hardware.architecture')}
                  actions={
                    <span className="text-foreground">
                      {hardwareData.cpu?.arch}
                    </span>
                  }
                />
                <CardItem
                  title={t('settings:hardware.cores')}
                  actions={
                    <span className="text-foreground">
                      {hardwareData.cpu?.core_count}
                    </span>
                  }
                />
                {hardwareData.cpu?.extensions?.join(', ').length > 0 && (
                  <CardItem
                    title={t('settings:hardware.instructions')}
                    column={hardwareData.cpu?.extensions.length > 6}
                    actions={
                      <span className="text-foreground wrap-break-word">
                        {hardwareData.cpu?.extensions?.join(', ')}
                      </span>
                    }
                  />
                )}
                <CardItem
                  title={t('settings:hardware.usage')}
                  actions={
                    <div className="flex items-center gap-2">
                      {systemUsage.cpu > 0 && (
                        <>
                          <Progress
                            value={systemUsage.cpu}
                            className="h-2 w-10 border"
                          />
                          <span className="text-foreground">
                            {systemUsage.cpu?.toFixed(2)}%
                          </span>
                        </>
                      )}
                    </div>
                  }
                />
              </Card>

              {/* RAM Information */}
              <Card title={t('settings:hardware.memory')}>
                <CardItem
                  title={t('settings:hardware.totalRam')}
                  actions={
                    <span className="text-foreground">
                      {formatMegaBytes(hardwareData.total_memory)}
                    </span>
                  }
                />
                <CardItem
                  title={t('settings:hardware.availableRam')}
                  actions={
                    <span className="text-foreground">
                      {formatMegaBytes(
                        hardwareData.total_memory - systemUsage.used_memory
                      )}
                    </span>
                  }
                />
                <CardItem
                  title={t('settings:hardware.usage')}
                  actions={
                    <div className="flex items-center gap-2">
                      {hardwareData.total_memory > 0 && (
                        <>
                          <Progress
                            value={
                              toNumber(
                                systemUsage.used_memory /
                                  hardwareData.total_memory
                              ) * 100
                            }
                            className="h-2 w-10 border"
                          />
                          <span className="text-foreground">
                            {(
                              toNumber(
                                systemUsage.used_memory /
                                  hardwareData.total_memory
                              ) * 100
                            ).toFixed(2)}
                            %
                          </span>
                        </>
                      )}
                    </div>
                  }
                />
              </Card>

              {/* Llamacpp Devices Information */}
              {!IS_MACOS && llamacpp && (
                <Card
                  title="GPUs"
                  header={
                    <div className="flex items-center justify-end mb-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleRefreshHardware}
                        disabled={isLoading}
                      >
                        {isLoading ? '...' : 'Refresh'}
                      </Button>
                    </div>
                  }
                >
                  {llamacppDevicesLoading ? (
                    <CardItem title="Loading devices..." actions={<></>} />
                  ) : llamacppDevicesError ? (
                    <CardItem
                      title="Error loading devices"
                      actions={
                        <span className="text-destructive text-sm">
                          {llamacppDevicesError}
                        </span>
                      }
                    />
                  ) : llamacppDevices.length > 0 ? (
                    llamacppDevices.map((device, index) => (
                      <Card key={index}>
                        <CardItem
                          title={device.name}
                          actions={
                            <div className="flex items-center gap-4">
                              {/* <div className="flex flex-col items-end gap-1">
                            <span className="text-foreground text-sm">
                              ID: {device.id}
                            </span>
                            <span className="text-foreground text-sm">
                              Memory: {formatMegaBytes(device.mem)} /{' '}
                              {formatMegaBytes(device.free)} free
                            </span>
                          </div> */}
                              <Switch
                                checked={device.activated}
                                onCheckedChange={() => {
                                  toggleDevice(device.id)
                                  serviceHub.models().stopAllModels()

                                  // Refresh active models after stopping
                                  serviceHub
                                    .models()
                                    .getActiveModels()
                                    .then((models) =>
                                      setActiveModels(models || [])
                                    )
                                }}
                              />
                            </div>
                          }
                        />
                        <div className="mt-3">
                          <CardItem
                            title={t('settings:hardware.vram')}
                            description="What a model can use here: the card less whatever is on it, less the memory no CUDA process can reach, less the reserve below."
                            actions={
                              <span className="text-foreground">
                                {formatMegaBytes(
                                  plannableMiB(
                                    device,
                                    systemUsage,
                                    hardwareData,
                                    gpuReserveMiB
                                  )
                                                                )}{' '}
                                {t('settings:hardware.freeOf')}{' '}
                                {formatMegaBytes(device.mem)}
                              </span>
                            }
                          />
                        </div>
                        {(() => {
                          const live = gpuMemoryUsage(
                            systemUsage,
                            hardwareData,
                            device.name
                          )
                          if (!live) return null
                          const percent =
                            toNumber(live.used / live.total) * 100
                          return (
                            <div className="mt-3">
                              <CardItem
                                title={t('settings:hardware.usage')}
                                actions={
                                  <div className="flex items-center gap-2">
                                    <Progress
                                      value={percent}
                                      className="h-2 w-10 border"
                                    />
                                    <span className="text-foreground">
                                      {formatMegaBytes(live.used)} /{' '}
                                      {formatMegaBytes(live.total)}
                                    </span>
                                  </div>
                                }
                              />
                            </div>
                          )
                        })()}
                        <div className="mt-3">
                          <CardItem
                            title="Reserved memory (MB)"
                            description={
                              isDisplayGpu(systemUsage, hardwareData, device.name)
                                ? 'Kept free of models, on top of what this card already has in use. It is driving a display, so raise this if the desktop stutters while a model is loaded.'
                                : 'Kept free of models, on top of what this card already has in use. Raise it if this card also drives a display or runs anything else.'
                            }
                            actions={
                              <Input
                                className="w-24 text-right"
                                type="number"
                                min={0}
                                step={128}
                                value={String(
                                  resolveGpuReserveMiB(
                                    systemUsage,
                                    hardwareData,
                                    device,
                                    gpuReserveMiB
                                  )
                                )}
                                onChange={(e) =>
                                  setGpuReserve(
                                    device.id,
                                    toNumber(e.target.value)
                                  )
                                }
                              />
                            }
                          />
                        </div>
                      </Card>
                    ))
                  ) : (
                    <CardItem title="No devices found" actions={<></>} />
                  )}
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

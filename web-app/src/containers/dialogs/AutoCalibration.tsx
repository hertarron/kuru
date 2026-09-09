import { useEffect } from 'react'
import { listen } from '@tauri-apps/api/event'

import { useModelProvider } from '@/hooks/useModelProvider'
import { useServiceHub } from '@/hooks/useServiceHub'
import { isPlatformTauri } from '@/lib/platform/utils'
import { ensureCalibration } from '@/lib/calibrationRunner'
import { events, AppEvent } from '@janhq/core'

type UnloadEventPayload = {
  model: string
  exit_code?: number | null
}

/**
 * Background fit-test scheduler. Measuring is worth doing exactly once per
 * model and card set, and never worth interrupting anything for, so probes
 * start only from idle transitions — model arrival, model unload, and one
 * delayed sweep after launch — and only while nothing is loaded, streaming,
 * or already measuring. Every outcome lands silently in the calibration
 * store, where the download popover and the settings sheet already reflect
 * it. A chat send preempts a running probe (see custom-chat-transport).
 */
export default function AutoCalibration() {
  const serviceHub = useServiceHub()

  useEffect(() => {
    if (!isPlatformTauri()) return
    let disposed = false
    const ensure = (modelId: string | undefined) => {
      if (!modelId || disposed) return
      void ensureCalibration(serviceHub, modelId).catch(() => undefined)
    }

    // Models fitted in an earlier session (explicit device set, no matching
    // measurement) get covered without a visit to the sheet. Delayed so a
    // cold start — backend download, embedding fetch — settles first.
    const startupTimer = setTimeout(() => {
      ensure(useModelProvider.getState().selectedModel?.id)
    }, 30_000)

    const unlisten = listen<UnloadEventPayload>(
      'llamacpp-model-unloaded',
      (event) => {
        // A crash is not an idle moment; the manual button stays for that.
        if (event.payload?.exit_code) return
        ensure(event.payload?.model)
      }
    ).catch((e) => {
      console.warn('listen llamacpp-model-unloaded failed:', e)
      return () => {}
    })

    const onImported = (payload?: { modelId?: string }) => {
      ensure(payload?.modelId)
    }
    events.on(AppEvent.onModelImported, onImported)

    return () => {
      disposed = true
      clearTimeout(startupTimer)
      void unlisten.then((fn) => fn?.())
      events.off(AppEvent.onModelImported, onImported)
    }
  }, [serviceHub])

  return null
}

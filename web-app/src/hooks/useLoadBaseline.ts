import { create } from 'zustand'

/**
 * What one model's cards and RAM looked like immediately before it loaded.
 *
 * The context sheet has to plan a resident model against the machine it will
 * find on its *next* load, which is this machine minus this model. Live usage
 * cannot answer that: the model's own VRAM is inside every reading. The load
 * path already measures exactly the right thing — it evicts, waits for the
 * freed VRAM to leave the reading, then reads — so that measurement is kept
 * here for the sheet to reuse.
 *
 * Deliberately not persisted: nothing is loaded after a restart, so a stored
 * baseline could only ever describe a machine that no longer exists.
 */
export interface LoadBaseline {
  /** VRAM in use per llama.cpp device id, in MiB, with this model unloaded. */
  perDeviceUsedMiB: Record<string, number>
  /** System RAM in use, in MiB. Absent when the reading was unavailable. */
  ramUsedMiB?: number
  at: number
}

interface LoadBaselineStore {
  byModel: Record<string, LoadBaseline>
  record: (modelId: string, baseline: Omit<LoadBaseline, 'at'>) => void
}

export const useLoadBaseline = create<LoadBaselineStore>()((set) => ({
  byModel: {},
  record: (modelId, baseline) =>
    set((state) => ({
      byModel: { ...state.byModel, [modelId]: { ...baseline, at: Date.now() } },
    })),
}))

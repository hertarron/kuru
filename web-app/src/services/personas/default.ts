/**
 * Default Personas Service - one JSON file per persona under `file://personas`.
 *
 * Mirrors the lorebooks service: no extension of its own, just a flat
 * directory of `<id>.json` files reached through the core `fs` bridge.
 */

import { fs } from '@janhq/core'
import type { Persona } from '@/lib/persona'
import { normalizePersona } from '@/lib/persona'
import type { PersonasService } from './types'

const DIR = 'file://personas'

/** Path separator on either kind of backend. */
const SEP = /[\\/]/

/**
 * `readdirSync` resolves to absolute OS paths on desktop, but plain names on
 * other backends. Taking the basename covers both, and rebuilding the
 * `file://` path from it keeps every read inside the data folder.
 */
const basename = (entry: string): string => entry.split(SEP).pop() ?? entry

const pathFor = (id: string): string => `${DIR}/${encodeURIComponent(id)}.json`

export class DefaultPersonasService implements PersonasService {
  private async ensureDir(): Promise<void> {
    if (!(await fs.existsSync(DIR))) {
      await fs.mkdir(DIR)
    }
  }

  async getPersonas(): Promise<Persona[]> {
    if (!(await fs.existsSync(DIR))) return []

    const entries: string[] = (await fs.readdirSync(DIR)) ?? []
    const personas: Persona[] = []

    for (const entry of entries) {
      const name = basename(entry)
      if (!name.endsWith('.json')) continue

      const id = decodeURIComponent(name.slice(0, -'.json'.length))
      try {
        const raw = JSON.parse(await fs.readFileSync(`${DIR}/${name}`))
        const persona = normalizePersona(raw, id)
        if (persona) personas.push(persona)
      } catch (error) {
        console.error(`Failed to read persona ${name}:`, error)
      }
    }

    return personas.sort((a, b) => a.name.localeCompare(b.name))
  }

  async savePersona(persona: Persona): Promise<void> {
    await this.ensureDir()
    await fs.writeFileSync(pathFor(persona.id), JSON.stringify(persona, null, 2))
  }

  async deletePersona(id: string): Promise<void> {
    const path = pathFor(id)
    if (await fs.existsSync(path)) {
      await fs.unlinkSync(path)
    }
  }
}

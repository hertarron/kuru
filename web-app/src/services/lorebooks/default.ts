/**
 * Default Lorebooks Service - one JSON file per book under `file://lorebooks`.
 *
 * Characters go through the assistant extension, but lorebooks have no
 * extension of their own and don't need one: the library is a flat directory
 * of `<id>.json` files, so the core `fs` bridge is enough.
 */

import { fs } from '@janhq/core'
import type { Lorebook } from '@/lib/lorebook'
import { normalizeLorebook } from '@/lib/lorebook'
import type { LorebooksService } from './types'

const DIR = 'file://lorebooks'

/** Path separator on either kind of backend. */
const SEP = /[\\/]/

/**
 * `readdirSync` resolves to absolute OS paths on desktop, but plain names on
 * other backends. Taking the basename covers both, and rebuilding the
 * `file://` path from it keeps every read inside the data folder.
 */
const basename = (entry: string): string =>
  entry.split(SEP).pop() ?? entry

const pathFor = (id: string): string => `${DIR}/${encodeURIComponent(id)}.json`

export class DefaultLorebooksService implements LorebooksService {
  private async ensureDir(): Promise<void> {
    if (!(await fs.existsSync(DIR))) {
      await fs.mkdir(DIR)
    }
  }

  async getLorebooks(): Promise<Lorebook[]> {
    if (!(await fs.existsSync(DIR))) return []

    const entries: string[] = (await fs.readdirSync(DIR)) ?? []
    const books: Lorebook[] = []

    for (const entry of entries) {
      const name = basename(entry)
      if (!name.endsWith('.json')) continue

      const id = decodeURIComponent(name.slice(0, -'.json'.length))
      try {
        const raw = JSON.parse(await fs.readFileSync(`${DIR}/${name}`))
        // Re-normalizing on read means a hand-edited file, or one written by
        // an older build, still loads in the current shape.
        const book = normalizeLorebook(raw, {
          id,
          name: raw?.name,
          allowEmpty: true,
        })
        if (book) books.push(book)
      } catch (error) {
        console.error(`Failed to read lorebook ${name}:`, error)
      }
    }

    return books.sort((a, b) => a.name.localeCompare(b.name))
  }

  async saveLorebook(lorebook: Lorebook): Promise<void> {
    await this.ensureDir()
    await fs.writeFileSync(pathFor(lorebook.id), JSON.stringify(lorebook, null, 2))
  }

  async deleteLorebook(id: string): Promise<void> {
    const path = pathFor(id)
    if (await fs.existsSync(path)) {
      await fs.unlinkSync(path)
    }
  }
}

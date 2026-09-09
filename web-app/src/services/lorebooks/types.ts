/**
 * Lorebooks Service Types
 *
 * Lorebooks live in their own library on disk, independent of characters —
 * characters and threads reference them by id, they never embed a copy.
 */

import type { Lorebook } from '@/lib/lorebook'

export interface LorebooksService {
  /** Every book in the library. Missing/unreadable files are skipped. */
  getLorebooks(): Promise<Lorebook[]>

  /** Creates or overwrites the book's file. */
  saveLorebook(lorebook: Lorebook): Promise<void>

  deleteLorebook(id: string): Promise<void>
}

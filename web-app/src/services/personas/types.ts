/**
 * Personas Service Types
 *
 * Personas are a flat library on disk. Threads and settings reference one by
 * id; nothing embeds a copy, so renaming a persona reaches every chat using it.
 */

import type { Persona } from '@/lib/persona'

export interface PersonasService {
  /** Every persona in the library. Missing/unreadable files are skipped. */
  getPersonas(): Promise<Persona[]>

  /** Creates or overwrites the persona's file. */
  savePersona(persona: Persona): Promise<void>

  deletePersona(id: string): Promise<void>
}

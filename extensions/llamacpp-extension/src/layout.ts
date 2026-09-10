import { fs, joinPath, logger } from '@janhq/core'
import { basename } from '@tauri-apps/api/path'

/**
 * Breadth-first search of `rootDir` for the directory directly containing
 * a file named `serverName`. Returns the absolute path of that directory,
 * or null if not found.
 */
export async function findLlamaServerDir(
  rootDir: string,
  serverName: string
): Promise<string | null> {
  // Note: fs.readdirSync returns absolute child paths (see
  // src-tauri/src/core/filesystem/commands.rs::readdir_sync), not basenames.
  const queue: string[] = [rootDir]
  while (queue.length > 0) {
    const current = queue.shift() as string
    let entries: string[]
    try {
      entries = await fs.readdirSync(current)
    } catch {
      continue
    }
    for (const entryPath of entries) {
      let stat
      try {
        stat = await fs.fileStat(entryPath)
      } catch {
        continue
      }
      if (!stat) continue
      const entryName = await basename(entryPath)
      if (!stat.isDirectory && entryName === serverName) {
        return current
      }
      if (stat.isDirectory) {
        queue.push(entryPath)
      }
    }
  }
  return null
}

/**
 * Move a just-extracted backend so the server binary sits at
 * `<backendDir>/build/bin/`, whatever shape the archive had.
 *
 * Upstream Windows zips are flat with the binary and its DLLs at the root;
 * upstream Linux tarballs nest under `llama-bXXXX/`. Both the automatic
 * download and the manual install go through here, because the CUDA
 * redistributable is unpacked into `build/bin` and would otherwise land beside
 * nothing.
 *
 * Throws if no server binary is anywhere under `backendDir`. Leaves
 * `backendDir` removed in that case, so a retry cannot reuse a half-install.
 */
export async function normalizeBackendLayout(
  backendDir: string,
  serverName: string
): Promise<string> {
  const expectedBinDir = await joinPath([backendDir, 'build', 'bin'])
  const expectedBinPath = await joinPath([expectedBinDir, serverName])
  if (await fs.existsSync(expectedBinPath)) return expectedBinPath

  const foundDir = await findLlamaServerDir(backendDir, serverName)
  if (!foundDir) {
    await fs.rm(backendDir)
    throw new Error(
      'Not a supported backend archive! Missing llama-server binary.'
    )
  }

  if (foundDir !== expectedBinDir) {
    const staging = `${backendDir}.staging`
    try {
      // Move the binary's dir into build/bin in one rename to keep relative
      // symlinks intact (libggml.so → .so.0 → .so.0.10.0). A flat-root
      // archive can't rename into its own subtree, so stage to a sibling.
      if (foundDir === backendDir) {
        await fs.mv(backendDir, staging)
        await fs.mkdir(await joinPath([backendDir, 'build']))
        await fs.mv(staging, expectedBinDir)
      } else {
        await fs.mkdir(await joinPath([backendDir, 'build']))
        await fs.mv(foundDir, expectedBinDir)
      }
    } catch (e) {
      if (await fs.existsSync(staging)) await fs.rm(staging)
      if (await fs.existsSync(backendDir)) await fs.rm(backendDir)
      throw new Error(`Failed to normalize backend layout: ${String(e)}`)
    }
  }

  if (!(await fs.existsSync(expectedBinPath))) {
    await fs.rm(backendDir)
    throw new Error('Backend layout normalization did not produce a server binary.')
  }
  logger.info(`Backend layout normalized to ${expectedBinDir}`)
  return expectedBinPath
}

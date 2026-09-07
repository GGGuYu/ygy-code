// @ygy-code/core — Optimistic-concurrency guard for file writes.
//
// The readFile tool already fingerprints every fully-delivered file into
// LoopState.readFileCache (mtime + size). Writing tools (writeFile / edit,
// plus the intercepted sed -i path) consult that same cache before touching
// the filesystem: an existing file the agent never read, or whose
// mtime/size changed since the read, is refused with an error telling the
// model to re-read first. This turns silent lost updates — concurrent
// sub-agents, a second ygy process, a human editor, or a formatter mutating
// the file between the agent's read and its write — into loud, self-healing
// failures.
//
// Why stat fingerprints instead of a lock or a self-managed version counter:
// the kernel updates mtime/size for EVERY write path, including shell
// redirections, editors, and other agent processes, so external mutations
// that our own bookkeeping would never observe are detected for free. The
// check is also fail-closed on path-form mismatches (case variants on
// case-insensitive filesystems, symlinks): they fall into the "not read"
// branch and are refused, never silently allowed.
import fs from 'node:fs/promises'

import type { ReadFileCache } from '../tools/read-file.js'

async function statFingerprint(filePath: string): Promise<{ mtimeMs: number; size: number } | null> {
  const stat = await fs.stat(filePath).catch(() => null)
  if (!stat) return null
  return { mtimeMs: stat.mtimeMs, size: stat.size }
}

/** After a successful write, refresh the fingerprint so the next write by the
 *  same agent passes without a forced re-read (the agent authored the current
 *  content), and a follow-up readFile hits the de-dup stub instead of
 *  re-delivering the bytes the agent just wrote. */
export async function recordWriteFingerprint(cache: ReadFileCache | undefined, filePath: string): Promise<void> {
  const fingerprint = await statFingerprint(filePath)
  if (fingerprint) cache?.set(filePath, fingerprint)
  else cache?.delete(filePath)
}

/** Pre-write OCC check. Returns null when the write may proceed, otherwise an
 *  error string for the model. A missing target is allowed (file-creation
 *  path); a mismatched fingerprint means someone else wrote in between. */
export async function verifyFileUnchangedSinceRead(
  cache: ReadFileCache | undefined,
  filePath: string,
): Promise<string | null> {
  if (!cache) return null
  const current = await statFingerprint(filePath)
  if (!current) return null
  const read = cache.get(filePath)
  if (!read) {
    return (
      `${filePath} exists but has not been read in this session. ` +
      `Read it with readFile before writing so edits are based on current content.`
    )
  }
  if (read.mtimeMs !== current.mtimeMs || read.size !== current.size) {
    return (
      `${filePath} has been modified since it was last read (by the user, a concurrent agent, or another process). ` +
      `Read it again before writing.`
    )
  }
  return null
}

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { recordWriteFingerprint, verifyFileUnchangedSinceRead } from '../src/agent/file-write-guard.js'
import { executeWriteTool } from '../src/agent/tool-execution.js'
import { createReadFileTool } from '../src/tools/read-file.js'
import type { AgentCallbacks } from '../src/types/index.js'

let dir: string
let cache: ReturnType<typeof createCache>
const callbacks = { onFileEdit: vi.fn() } as unknown as AgentCallbacks
const noBeforeWrite = async (): Promise<void> => {}

function createCache() {
  return new Map<string, { mtimeMs: number; size: number }>()
}

const writeTool = (filePath: string, content: string, toolCallId = 'call-write') =>
  executeWriteTool('writeFile', { filePath, content }, toolCallId, callbacks, undefined, cache, noBeforeWrite)

const editTool = (input: Record<string, unknown>, toolCallId = 'call-edit') =>
  executeWriteTool('edit', input, toolCallId, callbacks, undefined, cache, noBeforeWrite)

/** Full read through the real readFile tool so the cache entry is recorded
 *  exactly the way production does it. */
const readViaTool = async (filePath: string): Promise<unknown> => {
  const tool = createReadFileTool(cache)
  return tool.execute!({ filePath }, { toolCallId: 'call-read' })
}

beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), 'ygy-write-guard-'))
  cache = createCache()
})

afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

describe('verifyFileUnchangedSinceRead / recordWriteFingerprint', () => {
  it('refuses an existing file the agent never read', async () => {
    const file = path.join(dir, 'unread.ts')
    await fs.writeFile(file, 'original')
    await expect(verifyFileUnchangedSinceRead(cache, file)).resolves.toMatch(/has not been read/)
  })

  it('passes after a full read and detects a later content mutation', async () => {
    const file = path.join(dir, 'mutated.ts')
    await fs.writeFile(file, 'hello')
    await readViaTool(file)
    await expect(verifyFileUnchangedSinceRead(cache, file)).resolves.toBeNull()
    await fs.writeFile(file, 'changed content')
    await expect(verifyFileUnchangedSinceRead(cache, file)).resolves.toMatch(/modified since/)
  })

  it('detects a same-size touch (mtime-only change)', async () => {
    const file = path.join(dir, 'touched.ts')
    await fs.writeFile(file, 'same length!!')
    await readViaTool(file)
    const future = new Date(Date.now() + 5000)
    await fs.utimes(file, future, future)
    await expect(verifyFileUnchangedSinceRead(cache, file)).resolves.toMatch(/modified since/)
  })

  it('allows writing when the target disappeared (create semantics)', async () => {
    const file = path.join(dir, 'gone.ts')
    await fs.writeFile(file, 'temp')
    await readViaTool(file)
    await fs.rm(file)
    await expect(verifyFileUnchangedSinceRead(cache, file)).resolves.toBeNull()
  })

  it('recordWriteFingerprint lets consecutive writes pass and still detects later drift', async () => {
    const file = path.join(dir, 'record.ts')
    await writeTool(file, 'v1')
    await expect(verifyFileUnchangedSinceRead(cache, file)).resolves.toBeNull()
    await fs.writeFile(file, 'someone else')
    await expect(verifyFileUnchangedSinceRead(cache, file)).resolves.toMatch(/modified since/)
  })

  it('recordWriteFingerprint drops the entry when the file is gone', async () => {
    const file = path.join(dir, 'deleted.ts')
    await writeTool(file, 'v1')
    expect(cache.get(file)).toBeDefined()
    await fs.rm(file)
    await recordWriteFingerprint(cache, file)
    expect(cache.has(file)).toBe(false)
  })
})

describe('executeWriteTool OCC integration', () => {
  it('writeFile refuses an existing unread file without touching it', async () => {
    const file = path.join(dir, 'w-unread.ts')
    await fs.writeFile(file, 'original')
    await expect(writeTool(file, 'clobber')).resolves.toMatch(/has not been read/)
    await expect(fs.readFile(file, 'utf-8')).resolves.toBe('original')
  })

  it('writeFile succeeds after a read and overwrites the content', async () => {
    const file = path.join(dir, 'w-ok.ts')
    await fs.writeFile(file, 'original')
    await readViaTool(file)
    await expect(writeTool(file, 'new content')).resolves.toMatch(/File written: /)
    await expect(fs.readFile(file, 'utf-8')).resolves.toBe('new content')
  })

  it('writeFile detects an external mutation between read and write', async () => {
    const file = path.join(dir, 'w-race.ts')
    await fs.writeFile(file, 'original')
    await readViaTool(file)
    await fs.writeFile(file, 'external writer')
    await expect(writeTool(file, 'stale agent view')).resolves.toMatch(/modified since/)
    await expect(fs.readFile(file, 'utf-8')).resolves.toBe('external writer')
  })

  it('writeFile creates a new file without requiring a prior read', async () => {
    const file = path.join(dir, 'w-new.ts')
    await expect(writeTool(file, 'fresh')).resolves.toMatch(/File created: /)
    await expect(fs.readFile(file, 'utf-8')).resolves.toBe('fresh')
  })

  it('consecutive writes pass without re-reading in between', async () => {
    const file = path.join(dir, 'w-twice.ts')
    await fs.writeFile(file, 'original')
    await readViaTool(file)
    await expect(writeTool(file, 'first')).resolves.toMatch(/File written/)
    await expect(writeTool(file, 'second')).resolves.toMatch(/File written/)
    await expect(fs.readFile(file, 'utf-8')).resolves.toBe('second')
  })

  it('edit refuses an unread file', async () => {
    const file = path.join(dir, 'e-unread.ts')
    await fs.writeFile(file, 'hello world')
    await expect(editTool({ filePath: file, oldString: 'hello', newString: 'hi' })).resolves.toMatch(
      /has not been read/,
    )
  })

  it('edit detects an external mutation and keeps the external content', async () => {
    const file = path.join(dir, 'e-race.ts')
    await fs.writeFile(file, 'hello world')
    await readViaTool(file)
    await fs.writeFile(file, 'external world')
    await expect(editTool({ filePath: file, oldString: 'world', newString: 'planet' })).resolves.toMatch(
      /modified since/,
    )
    await expect(fs.readFile(file, 'utf-8')).resolves.toBe('external world')
  })

  it('edit applies the replacement after a fresh read', async () => {
    const file = path.join(dir, 'e-ok.ts')
    await fs.writeFile(file, 'hello world')
    await readViaTool(file)
    await expect(editTool({ filePath: file, oldString: 'hello', newString: 'hi' })).resolves.toMatch(/File edited/)
    await expect(fs.readFile(file, 'utf-8')).resolves.toBe('hi world')
  })

  it('a post-write readFile hits the delivery de-dup stub', async () => {
    const file = path.join(dir, 'dedup.ts')
    await fs.writeFile(file, 'original content')
    await readViaTool(file)
    await writeTool(file, 'agent authored')
    const second = await readViaTool(file)
    expect(second).toMatch(/unchanged since its full content/)
  })
})

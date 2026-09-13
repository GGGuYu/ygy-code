import { afterEach, describe, expect, it } from 'vitest'

import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { WIKI_MEMORY_TOOL_NAME, createWikiMemoryTool } from '../src/tools/wiki-memory.js'

const temporaryDirectories: string[] = []

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })))
})

async function makeWikiDir(): Promise<string> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ygy-code-wiki-tool-'))
  temporaryDirectories.push(root)
  const wikiDir = path.join(root, 'wiki')
  await fs.mkdir(path.join(wikiDir, 'tools'), { recursive: true })
  await fs.writeFile(path.join(wikiDir, 'index.md'), '# Wiki Index\n', 'utf8')
  await fs.writeFile(path.join(wikiDir, 'tools', 'wiki-standards.md'), '## Wiki Standards\n', 'utf8')
  return wikiDir
}

describe('createWikiMemoryTool', () => {
  it('exposes the tool name, a non-empty description and a strict empty schema', () => {
    const wikiDir = '/tmp/some-wiki'
    const definition = createWikiMemoryTool({ path: wikiDir }) as unknown as {
      description?: string
      inputSchema?: { strict?: () => unknown }
    }
    expect(WIKI_MEMORY_TOOL_NAME).toBe('wikiMemory')
    expect(definition.description).toBeTruthy()
    expect(definition.description).toContain('long-term memory')
    expect(definition.inputSchema).toBeTruthy()
  })

  it('returns the full wiki SOP wrapped in an XML envelope on activation', async () => {
    const wikiDir = await makeWikiDir()
    const tool = createWikiMemoryTool({ path: wikiDir }) as any

    const result: string = await tool.execute({}, { toolCallId: 'tc-wiki-1' })

    expect(result).toContain('<wiki_memory_instructions>')
    expect(result).toContain('</wiki_memory_instructions>')
    // The SOP body is the former resident rules — full retrieval guidance.
    expect(result).toContain('index.md')
    expect(result).toContain('.pageindex')
    expect(result).toContain('wiki-standards.md')
    // Real path resolution keeps grep instructions usable even via symlinks.
    expect(result).toContain(wikiDir)
  })

  it('returns a repair hint instead of an empty result when the wiki path is unusable', async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ygy-code-wiki-tool-missing-'))
    temporaryDirectories.push(root)
    const tool = createWikiMemoryTool({ path: path.join(root, 'does-not-exist') }) as any

    const result: string = await tool.execute({}, { toolCallId: 'tc-wiki-2' })

    expect(result).not.toContain('<wiki_memory_instructions>')
    expect(result).toContain('not usable')
    expect(result).toContain('wiki.path')
  })
})

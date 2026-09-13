import { afterEach, describe, expect, it } from 'vitest'

import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { buildKnowledgeContext } from '../src/knowledge/loader.js'

const temporaryDirectories: string[] = []

afterEach(async () => {
  delete process.env.YGY_CODE_HOME
  await Promise.all(temporaryDirectories.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })))
})

describe('buildKnowledgeContext deterministic deduplication', () => {
  it('injects byte-identical rule content only once while preserving distinct layers', async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ygy-code-knowledge-dedupe-'))
    temporaryDirectories.push(root)
    const userDir = path.join(root, 'user')
    const projectDir = path.join(root, 'project')
    await fs.mkdir(userDir, { recursive: true })
    await fs.mkdir(path.join(projectDir, '.git'), { recursive: true })
    process.env.YGY_CODE_HOME = userDir
    await fs.writeFile(path.join(userDir, 'AGENTS.md'), 'shared exact rule\n', 'utf8')
    await fs.writeFile(path.join(projectDir, 'AGENTS.md'), 'shared exact rule\n', 'utf8')
    await fs.writeFile(path.join(projectDir, 'AGENTS.local.md'), 'distinct local rule\n', 'utf8')

    const context = await buildKnowledgeContext({ cwd: projectDir })

    expect(context.match(/shared exact rule/g)).toHaveLength(1)
    expect(context).toContain('### User Preferences')
    expect(context).not.toContain('### Project AGENTS.md')
    expect(context).toContain('### Local Preferences')
    expect(context).toContain('distinct local rule')
  })
})

describe('buildKnowledgeContext wiki memory mode', () => {
  it('injects a one-line wiki pointer instead of the full SOP (progressive disclosure)', async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ygy-code-knowledge-wiki-'))
    temporaryDirectories.push(root)
    const userDir = path.join(root, 'user')
    const projectDir = path.join(root, 'project')
    const wikiDir = path.join(root, 'wiki')
    await fs.mkdir(userDir, { recursive: true })
    await fs.mkdir(path.join(projectDir, '.git'), { recursive: true })
    await fs.mkdir(path.join(wikiDir, 'tools'), { recursive: true })
    process.env.YGY_CODE_HOME = userDir
    await fs.writeFile(path.join(wikiDir, 'index.md'), '# Wiki Index\n', 'utf8')
    await fs.writeFile(path.join(wikiDir, 'tools', 'wiki-standards.md'), '## Wiki Standards\n', 'utf8')
    await fs.writeFile(path.join(projectDir, 'AGENTS.md'), 'project rule\n', 'utf8')

    const context = await buildKnowledgeContext({
      cwd: projectDir,
      wikiMemory: { path: wikiDir },
    })

    expect(context).toContain('### Wiki 记忆（外部长期记忆）')
    expect(context).toContain(wikiDir)
    expect(context).toContain('wikiMemory')
    // The full SOP is delivered on demand by the wikiMemory tool, not resident.
    expect(context).not.toContain('guyu-feishu-llm-wiki')
    expect(context).not.toContain('.pageindex')
    expect(context).not.toContain('wiki-standards.md')
    expect(context).not.toContain('### User Auto Memory')
    expect(context).toContain('### Project AGENTS.md')
    expect(context).toContain('project rule')
  })

  it('skips the wiki section when the configured path is not a directory', async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ygy-code-knowledge-wiki-missing-'))
    temporaryDirectories.push(root)
    const userDir = path.join(root, 'user')
    const projectDir = path.join(root, 'project')
    await fs.mkdir(userDir, { recursive: true })
    await fs.mkdir(path.join(projectDir, '.git'), { recursive: true })
    process.env.YGY_CODE_HOME = userDir

    const context = await buildKnowledgeContext({
      cwd: projectDir,
      wikiMemory: { path: path.join(root, 'does-not-exist') },
    })

    expect(context).not.toContain('### Wiki 记忆')
    expect(context).not.toContain('### User Auto Memory')
  })
})

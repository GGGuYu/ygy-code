// wikiRag unit tests — all embedding goes through a deterministic fake
// Embedder keyed on the text's first alphanumeric character. Nothing here
// imports Transformers.js, downloads a model, or touches the network; the
// persisted index lands in a throwaway YGY_CODE_HOME.
import { afterEach, describe, expect, it } from 'vitest'

import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import {
  type Embedder,
  type Reranker,
  WIKI_RAG_MODEL_ID,
  buildWikiRagIndex,
  loadWikiRagIndex,
  saveWikiRagIndex,
  searchWikiRag,
  wikiRagIndexPath,
} from '../src/knowledge/wiki-rag.js'
import { WIKI_RAG_TOOL_NAME, createWikiRagTool } from '../src/tools/wiki-rag.js'

const temporaryDirectories: string[] = []

afterEach(async () => {
  delete process.env.YGY_CODE_HOME
  await Promise.all(temporaryDirectories.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })))
})

async function makeTempDir(prefix: string): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), prefix))
  temporaryDirectories.push(dir)
  return dir
}

// ── Deterministic fake embedder ──────────────────────────────────────────
// Direction = angle(code(first alphanumeric char) × 0.7 rad) on the unit
// circle. Same leading char ⇒ identical vector (cosine 1.0); different chars
// ⇒ clearly separated directions (no integer code gap is a full turn).

function fakeVector(text: string): Float32Array {
  const match = text.match(/[a-zA-Z0-9]/)
  const code = match ? match[0].charCodeAt(0) : '?'.charCodeAt(0)
  const angle = code * 0.7
  return new Float32Array([Math.cos(angle), Math.sin(angle)])
}

function countingEmbedder() {
  let calls = 0
  const embedder: Embedder = async (texts) => {
    calls += 1
    return texts.map(fakeVector)
  }
  return { embedder, callCount: () => calls }
}

// ── Deterministic fake reranker ───────────────────────────────────────────
// keywordReranker mimics a cross-encoder: texts containing the needle score
// 0.9, everything else 0.1 (ties keep vector order via stable sort).
// offlineReranker stands in for the default Transformers.js reranker in the
// tool-level tests — throwing makes them exercise the degrade-to-vector path
// instead of downloading the real model.

function keywordReranker(needle: string): Reranker {
  return async (_query, texts) => texts.map((text) => (text.includes(needle) ? 0.9 : 0.1))
}

const offlineReranker: Reranker = async () => {
  throw new Error('reranker offline')
}

// ── Fixture wiki ──────────────────────────────────────────────────────────
// personal/foo.md (23 lines, 3-level tree, one <3-line stub section) plus a
// flat tools.md. Expected foo chunks (DFS order):
//   foo 文档                       1-4
//   foo 文档 > alpha 子节点        5-8
//   foo 文档 > alpha 子节点 > C 第三层  9-14
//   foo 文档 > beta 兄弟          15-18
//   foo 文档 > 短段                19-19  ← skipped (<3 lines)
//   foo 文档 > 结尾                20-23

const FOO_MD = [
  '# foo 文档',
  'alpha 段落开头，这是顶层介绍内容。',
  '继续介绍。',
  '',
  '## alpha 子节点',
  '这是 alpha 子节点正文第一行。',
  '这是 alpha 子节点正文第二行。',
  '',
  '### C 第三层',
  'charlie 内容行 1',
  'charlie 内容行 2',
  'charlie 内容行 3',
  'charlie 内容行 4',
  '',
  '## beta 兄弟',
  'beta 内容行 1',
  'beta 内容行 2',
  '',
  '## 短段',
  '## 结尾',
  'tail line 1',
  'tail line 2',
  'tail line 3',
].join('\n')

const FOO_TREE = {
  doc_name: 'foo',
  line_count: 23,
  structure: [
    {
      title: 'foo 文档',
      node_id: '0000',
      line_num: 1,
      nodes: [
        {
          title: 'alpha 子节点',
          node_id: '0001',
          line_num: 5,
          nodes: [{ title: 'C 第三层', node_id: '0002', line_num: 9 }],
        },
        { title: 'beta 兄弟', node_id: '0003', line_num: 15 },
        { title: '短段', node_id: '0004', line_num: 19 },
        { title: '结尾', node_id: '0005', line_num: 20 },
      ],
    },
  ],
}

const TOOLS_MD = ['# zulu 工具规范', 'zulu 内容行 1', 'zulu 内容行 2', 'zulu 内容行 3'].join('\n')

const TOOLS_TREE = {
  doc_name: 'tools',
  line_count: 4,
  structure: [{ title: 'zulu 工具规范', node_id: '0000', line_num: 1 }],
}

async function makeWiki(): Promise<string> {
  const root = await makeTempDir('ygy-code-wiki-rag-')
  const wikiDir = path.join(root, 'wiki')
  await fs.mkdir(path.join(wikiDir, 'personal'), { recursive: true })
  await fs.mkdir(path.join(wikiDir, '.pageindex', 'personal'), { recursive: true })
  await fs.writeFile(path.join(wikiDir, 'personal', 'foo.md'), FOO_MD, 'utf8')
  await fs.writeFile(path.join(wikiDir, '.pageindex', 'personal', 'foo.tree.json'), JSON.stringify(FOO_TREE), 'utf8')
  await fs.writeFile(path.join(wikiDir, 'tools.md'), TOOLS_MD, 'utf8')
  await fs.writeFile(path.join(wikiDir, '.pageindex', 'tools.tree.json'), JSON.stringify(TOOLS_TREE), 'utf8')
  return wikiDir
}

async function withTempHome(): Promise<string> {
  const userDir = await makeTempDir('ygy-code-wiki-rag-home-')
  process.env.YGY_CODE_HOME = userDir
  return userDir
}

// ── Chunk extraction ──────────────────────────────────────────────────────

describe('wiki-rag chunk extraction', () => {
  it('cuts DFS line spans, joins ancestor title chains, and skips <3-line stubs', async () => {
    const wikiDir = await makeWiki()
    const realWiki = await fs.realpath(wikiDir)
    const index = await buildWikiRagIndex(realWiki, async (texts) => texts.map(fakeVector))

    const foo = index.chunks.filter((chunk) => chunk.file === 'personal/foo.md')
    expect(foo.map((chunk) => [chunk.title, chunk.startLine, chunk.endLine])).toEqual([
      ['foo 文档', 1, 4],
      ['foo 文档 > alpha 子节点', 5, 8],
      ['foo 文档 > alpha 子节点 > C 第三层', 9, 14],
      ['foo 文档 > beta 兄弟', 15, 18],
      // 短段 19-19 dropped: fewer than 3 lines.
      ['foo 文档 > 结尾', 20, 23],
    ])
    // Chunk text is the exact md slice for the span (trimmed: the trailing
    // blank line 14 is dropped).
    expect(foo[2].text.split('\n')[0]).toBe('### C 第三层')
    expect(foo[2].text.split('\n')).toHaveLength(5)
    // The flat top-level doc indexes too.
    expect(index.chunks.filter((chunk) => chunk.file === 'tools.md')).toHaveLength(1)
  })

  it('throws a rebuild-worthy error when the wiki has no PageIndex trees', async () => {
    const wikiDir = await makeTempDir('ygy-code-wiki-rag-empty-')
    await fs.mkdir(wikiDir, { recursive: true })
    await expect(buildWikiRagIndex(wikiDir, async (texts) => texts.map(fakeVector))).rejects.toThrow(
      /no usable PageIndex trees/,
    )
  })
})

// ── Index build / persistence / reuse ─────────────────────────────────────

describe('wiki-rag index persistence', () => {
  it('persists to <YGY_CODE_HOME>/wiki-rag-index.json and reloads without re-embedding', async () => {
    const userDir = await withTempHome()
    const wikiDir = await makeWiki()
    const realWiki = await fs.realpath(wikiDir)
    const { embedder, callCount } = countingEmbedder()

    const built = await buildWikiRagIndex(realWiki, embedder)
    await saveWikiRagIndex(built)
    const buildCalls = callCount()

    const indexPath = path.join(userDir, 'wiki-rag-index.json')
    expect(await fs.readFile(indexPath, 'utf8')).toBeTruthy()
    const raw = JSON.parse(await fs.readFile(indexPath, 'utf8'))
    expect(raw.modelId).toBe(WIKI_RAG_MODEL_ID)
    expect(raw.wikiRealPath).toBe(realWiki)
    expect(raw.builtAt).toBeTruthy()
    expect(raw.chunks).toHaveLength(built.chunks.length)
    expect(raw.embeddings).toHaveLength(built.chunks.length)

    const loaded = await loadWikiRagIndex(realWiki)
    expect(loaded?.builtAt).toBe(built.builtAt)
    expect(loaded?.chunks).toHaveLength(built.chunks.length)
    expect(callCount()).toBe(buildCalls) // load path never invokes the embedder
    expect(wikiRagIndexPath()).toBe(indexPath)
  })

  it('rejects a stored index built for a different wiki (forces rebuild)', async () => {
    await withTempHome()
    const wikiA = await fs.realpath(await makeWiki())
    const wikiB = await fs.realpath(await makeWiki())
    await saveWikiRagIndex(await buildWikiRagIndex(wikiA, async (texts) => texts.map(fakeVector)))
    expect(await loadWikiRagIndex(wikiB)).toBeUndefined()
  })
})

// ── Search ────────────────────────────────────────────────────────────────

describe('wiki-rag search ranking', () => {
  it('ranks the chunk sharing the query vector first and sorts descending', async () => {
    await withTempHome()
    const realWiki = await fs.realpath(await makeWiki())
    const embedder: Embedder = async (texts) => texts.map(fakeVector)
    const index = await buildWikiRagIndex(realWiki, embedder)

    const hits = await searchWikiRag(index, embedder, 'C 查询第三层内容', 5)
    expect(hits[0].chunk.title).toBe('foo 文档 > alpha 子节点 > C 第三层')
    expect(hits[0].score).toBeCloseTo(1, 5)
    expect(hits.map((hit) => hit.score)).toEqual([...hits.map((hit) => hit.score)].sort((a, b) => b - a))

    const betaHits = await searchWikiRag(index, embedder, 'beta 兄弟在哪', 3)
    expect(betaHits).toHaveLength(3)
    expect(betaHits[0].chunk.title).toBe('foo 文档 > beta 兄弟')
  })
})

// ── Tool-level behaviour ──────────────────────────────────────────────────

describe('createWikiRagTool', () => {
  it('returns ranked sections with the first-build note and a readFile hint', async () => {
    await withTempHome()
    const wikiDir = await makeWiki()
    const realWiki = await fs.realpath(wikiDir)
    const { embedder } = countingEmbedder()
    const definition = createWikiRagTool({ path: wikiDir }, embedder, offlineReranker) as any

    expect(WIKI_RAG_TOOL_NAME).toBe('wikiRag')
    expect(definition.description).toContain('long-term memory')

    const result: string = await definition.execute({ query: 'C 查询' }, { toolCallId: 'tc-rag-1' })
    expect(result).toContain('（首次构建完成，6 chunks')
    expect(result).toContain('[1] personal/foo.md :9-14 — foo 文档 > alpha 子节点 > C 第三层（score 1.00）')
    expect(result).toContain('readFile')
    expect(result).toContain('grep')
    // Index was persisted against the REAL path even though a configured
    // path was handed in.
    const stored = await loadWikiRagIndex(realWiki)
    expect(stored?.wikiRealPath).toBe(realWiki)
  })

  it('skips the rebuild when the persisted index already matches', async () => {
    await withTempHome()
    const wikiDir = await makeWiki()
    const { embedder, callCount } = countingEmbedder()
    const definition = createWikiRagTool({ path: wikiDir }, embedder, offlineReranker) as any

    await definition.execute({ query: 'beta 查询' }, { toolCallId: 'tc-rag-2a' })
    const afterFirst = callCount()
    expect(afterFirst).toBeGreaterThan(1) // build batches + one query embed

    const second: string = await definition.execute({ query: 'beta 查询' }, { toolCallId: 'tc-rag-2b' })
    expect(second).not.toContain('首次构建完成')
    expect(callCount()).toBe(afterFirst + 1) // exactly the query embed
  })

  it('degrades to a grep hint instead of throwing when embedding fails', async () => {
    await withTempHome()
    const wikiDir = await makeWiki()
    const failing: Embedder = async () => {
      throw new Error('model offline')
    }
    const definition = createWikiRagTool({ path: wikiDir }, failing, offlineReranker) as any

    const result: string = await definition.execute({ query: 'anything' }, { toolCallId: 'tc-rag-3' })
    expect(result).toContain('wiki-rag 不可用')
    expect(result).toContain('model offline')
    expect(result).toContain('grep')
    expect(result).toContain(await fs.realpath(wikiDir))
  })

  it('degrades when the configured wiki directory does not exist', async () => {
    await withTempHome()
    const root = await makeTempDir('ygy-code-wiki-rag-missing-')
    const missing = path.join(root, 'does-not-exist')
    const embedder: Embedder = async (texts) => texts.map(fakeVector)
    const definition = createWikiRagTool({ path: missing }, embedder, offlineReranker) as any

    const result: string = await definition.execute({ query: 'anything' }, { toolCallId: 'tc-rag-4' })
    expect(result).toContain('wiki-rag 不可用')
    expect(result).toContain(missing)
  })
})

// ── Rerank (two-stage retrieval) ──────────────────────────────────────────

describe('wiki-rag rerank', () => {
  it('recalls a wider pool and lets the cross-encoder invert the vector order', async () => {
    await withTempHome()
    const realWiki = await fs.realpath(await makeWiki())
    const embedder: Embedder = async (texts) => texts.map(fakeVector)
    const index = await buildWikiRagIndex(realWiki, embedder)

    // Vector-only baseline: the C-section (cosine 1.0 with the query) ranks
    // first; tools.md sits mid-pack (rank 3 of 6) behind two foo chunks.
    const allVector = await searchWikiRag(index, embedder, 'C 查询', 6)
    expect(allVector[0].chunk.title).toBe('foo 文档 > alpha 子节点 > C 第三层')
    expect(allVector.map((hit) => hit.chunk.file).indexOf('tools.md')).toBe(2)

    // The keyword cross-encoder lifts tools.md over both chunks the vector
    // stage ranked higher; its 0.1-score ties keep vector order among the rest
    // (stable sort).
    const hits = await searchWikiRag(index, embedder, 'C 查询', 5, keywordReranker('zulu'))
    expect(hits).toHaveLength(5)
    expect(hits[0].chunk.file).toBe('tools.md')
    expect(hits[0].score).toBe(0.9)
    expect(hits.slice(1).map((hit) => hit.chunk.title)).toEqual(
      allVector
        .filter((hit) => hit.chunk.file !== 'tools.md')
        .slice(0, 4)
        .map((hit) => hit.chunk.title),
    )
  })

  it('degrades to pure vector order without throwing when the reranker fails', async () => {
    await withTempHome()
    const realWiki = await fs.realpath(await makeWiki())
    const embedder: Embedder = async (texts) => texts.map(fakeVector)
    const index = await buildWikiRagIndex(realWiki, embedder)

    const hits = await searchWikiRag(index, embedder, 'C 查询', 5, offlineReranker)
    expect(hits).toHaveLength(5)
    const plain = await searchWikiRag(index, embedder, 'C 查询', 5)
    expect(hits.map((hit) => hit.chunk.title)).toEqual(plain.map((hit) => hit.chunk.title))
    expect(hits[0].chunk.title).toBe('foo 文档 > alpha 子节点 > C 第三层')
  })

  it('marks hits reranked only when the cross-encoder actually scored them', async () => {
    await withTempHome()
    const realWiki = await fs.realpath(await makeWiki())
    const embedder: Embedder = async (texts) => texts.map(fakeVector)
    const index = await buildWikiRagIndex(realWiki, embedder)

    const plain = await searchWikiRag(index, embedder, 'C 查询', 5)
    expect(plain.every((hit) => hit.reranked)).toBe(false)
    const reranked = await searchWikiRag(index, embedder, 'C 查询', 5, keywordReranker('zulu'))
    expect(reranked.every((hit) => hit.reranked)).toBe(true)
  })

  it('notes two-stage retrieval in the tool output only when reranking succeeded', async () => {
    await withTempHome()
    const wikiDir = await makeWiki()
    const { embedder } = countingEmbedder()
    const rerankedTool = createWikiRagTool({ path: wikiDir }, embedder, keywordReranker('zulu')) as any

    const reranked: string = await rerankedTool.execute({ query: 'C 查询' }, { toolCallId: 'tc-rag-5a' })
    expect(reranked).toContain('[1] tools.md')
    expect(reranked).toContain('（两阶段检索：向量召回 20 → bge-reranker-base 精排 top 5）')

    // Second tool with a failing reranker reuses the persisted index and must
    // not claim two-stage precision it did not have.
    const plainTool = createWikiRagTool({ path: wikiDir }, embedder, offlineReranker) as any
    const plain: string = await plainTool.execute({ query: 'C 查询' }, { toolCallId: 'tc-rag-5b' })
    expect(plain).not.toContain('两阶段检索')
    expect(plain).toContain('[1] personal/foo.md :9-14')
  })
})

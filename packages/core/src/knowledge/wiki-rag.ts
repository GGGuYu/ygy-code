// @ygy-code/core — wiki-rag local vector index
//
// Backing store for the wikiRag tool: semantic (embedding) search over the
// external Markdown wiki. Chunking reuses the PageIndex structure trees —
// DFS-flatten each tree, one chunk per node spanning [node.line_num, next
// DFS node's line_num - 1] of the source md (last node runs to line_count).
// Spans under 3 lines are dropped (heading-only stubs); md files without a
// tree are skipped entirely — only tree-backed docs are indexed, mirroring
// the wiki SOP's "tree exists → pre-filter with it" rule.
//
// Embeddings run fully locally via Transformers.js (Xenova/bge-small-zh-v1.5,
// q8-quantized ONNX; first use downloads once into ~/.cache/huggingface).
// The library is imported lazily inside the embedder so loading this module
// — or the CLI bundle — never pulls in onnxruntime.
//
// The index persists at <userYgyDir>/wiki-rag-index.json. If it exists and
// wikiRealPath matches, it is reused as-is: NO incremental updates and NO
// mtime validation in this MVP (delete the file to force a rebuild) — both
// are left for a later iteration. Search is a plain dot product over all
// chunk vectors (they are L2-normalized, so dot = cosine), top-K.
import fs from 'node:fs/promises'
import path from 'node:path'

import { debugLog, fileExists, userYgyDir } from '../utils.js'

/** Local embedding model (Transformers.js, q8 = the old `quantized` flag). */
export const WIKI_RAG_MODEL_ID = 'Xenova/bge-small-zh-v1.5'

/** Chunks whose line span is smaller than this are dropped (empty stubs). */
const MIN_CHUNK_LINES = 3

/** How many chunk texts to embed per pipeline call. */
const EMBED_BATCH_SIZE = 16

/** Embeds a batch of texts into L2-normalized vectors. Injectable so tests
 *  never touch Transformers.js or download a model. */
export type Embedder = (texts: string[]) => Promise<Float32Array[]>

export interface WikiRagChunk {
  /** Wiki-relative md path, e.g. `personal/foo.md`. */
  file: string
  /** Heading chain incl. ancestors, e.g. `工具 > 搜索 > grep 用法`. */
  title: string
  startLine: number
  endLine: number
  text: string
}

export interface WikiRagIndex {
  modelId: string
  wikiRealPath: string
  builtAt: string
  chunks: WikiRagChunk[]
  /** One L2-normalized vector per chunk, aligned with `chunks`. */
  embeddings: number[][]
}

interface PageIndexNode {
  title: string
  node_id: string
  line_num: number
  nodes?: PageIndexNode[]
}

interface PageIndexTree {
  doc_name: string
  line_count: number
  structure: PageIndexNode[]
}

/** Where the built index lives (honours YGY_CODE_HOME via userYgyDir). */
export function wikiRagIndexPath(): string {
  return path.join(userYgyDir(), 'wiki-rag-index.json')
}

// ── Production embedder (lazy Transformers.js) ───────────────────────────

/** Load the feature-extraction pipeline. Typed via the library's own return
 *  type; the `await import` inside is the ONLY reference to the package, so
 *  nothing loads until the first embed call. */
async function loadExtractor() {
  const { pipeline } = await import('@huggingface/transformers')
  // v4 replaced the old `quantized: true` flag with dtype selection;
  // q8 maps to the same *_quantized.onnx weights.
  return pipeline('feature-extraction', WIKI_RAG_MODEL_ID, { dtype: 'q8' })
}

type Extractor = Awaited<ReturnType<typeof loadExtractor>>

let extractorPromise: Promise<Extractor> | undefined

/** Embedder backed by local Transformers.js inference. The first call lazily
 *  imports the package and loads the model (downloading it on first ever
 *  use); subsequent calls reuse the cached pipeline. */
export function createTransformersEmbedder(): Embedder {
  return async (texts) => {
    if (texts.length === 0) return []
    if (!extractorPromise) {
      extractorPromise = loadExtractor().catch((error) => {
        extractorPromise = undefined // don't cache a failed load
        throw error
      })
    }
    const extractor = await extractorPromise
    const output = await extractor(texts, { pooling: 'mean', normalize: true })
    const vectors = output.tolist() as number[][]
    return vectors.map((vector) => Float32Array.from(vector))
  }
}

// ── Chunk extraction ──────────────────────────────────────────────────────

/** DFS-flatten tree nodes with their ancestor title chains. Preorder on a
 *  well-formed tree is already the increasing line_num sequence the chunk
 *  spans are cut against. */
function flattenTree(nodes: PageIndexNode[], ancestors: string[], out: { title: string; line_num: number }[]) {
  for (const node of nodes) {
    const chain = [...ancestors, node.title]
    out.push({ title: chain.join(' > '), line_num: node.line_num })
    if (node.nodes?.length) flattenTree(node.nodes, chain, out)
  }
}

/** Extract chunks for one doc: tree (line spans) + md lines (text). */
function extractChunks(mdRelPath: string, tree: PageIndexTree, mdLines: string[]): WikiRagChunk[] {
  const flat: { title: string; line_num: number }[] = []
  flattenTree(tree.structure ?? [], [], flat)
  const lastLine = Math.min(tree.line_count ?? 0, mdLines.length)
  const chunks: WikiRagChunk[] = []
  for (let i = 0; i < flat.length; i++) {
    const startLine = flat[i].line_num
    const endLine = i + 1 < flat.length ? flat[i + 1].line_num - 1 : tree.line_count
    if (endLine - startLine + 1 < MIN_CHUNK_LINES) continue
    const clampedEnd = Math.min(endLine, lastLine)
    const text = mdLines
      .slice(startLine - 1, clampedEnd)
      .join('\n')
      .trim()
    if (!text) continue
    chunks.push({ file: mdRelPath, title: flat[i].title, startLine, endLine: clampedEnd, text })
  }
  return chunks
}

/** Recursively collect `*.tree.json` files under a directory. */
async function collectTreeFiles(dir: string, out: string[]) {
  let entries
  try {
    entries = await fs.readdir(dir, { withFileTypes: true })
  } catch {
    return
  }
  for (const entry of entries) {
    const entryPath = path.join(dir, entry.name)
    if (entry.isDirectory()) await collectTreeFiles(entryPath, out)
    else if (entry.name.endsWith('.tree.json')) out.push(entryPath)
  }
}

// ── Index build / load / search ───────────────────────────────────────────

/** Build a full index over every tree-backed md in the wiki. Throws when the
 *  wiki has no PageIndex trees at all — the tool degrades to grep there. */
export async function buildWikiRagIndex(wikiRealPath: string, embedder: Embedder): Promise<WikiRagIndex> {
  const pageIndexDir = path.join(wikiRealPath, '.pageindex')
  const treeFiles: string[] = []
  await collectTreeFiles(pageIndexDir, treeFiles)
  const chunks: WikiRagChunk[] = []
  for (const treeFile of treeFiles.sort()) {
    const mdRelPath = path.relative(pageIndexDir, treeFile).replace(/\.tree\.json$/, '.md')
    const mdAbsPath = path.join(wikiRealPath, mdRelPath)
    let tree: PageIndexTree
    let mdContent: string
    try {
      tree = JSON.parse(await fs.readFile(treeFile, 'utf8')) as PageIndexTree
      mdContent = await fs.readFile(mdAbsPath, 'utf8')
    } catch (error) {
      debugLog('wiki-rag.skip-doc', `${mdRelPath}: ${(error as Error).message}`)
      continue
    }
    chunks.push(...extractChunks(mdRelPath, tree, mdContent.split('\n')))
  }
  if (chunks.length === 0) {
    throw new Error(`no usable PageIndex trees under ${pageIndexDir}`)
  }
  const embeddings: number[][] = []
  for (let i = 0; i < chunks.length; i += EMBED_BATCH_SIZE) {
    const batch = chunks.slice(i, i + EMBED_BATCH_SIZE).map((chunk) => chunk.text)
    const vectors = await embedder(batch)
    for (const vector of vectors) embeddings.push(Array.from(vector))
  }
  debugLog('wiki-rag.build', `${chunks.length} chunks from ${treeFiles.length} trees`)
  return { modelId: WIKI_RAG_MODEL_ID, wikiRealPath, builtAt: new Date().toISOString(), chunks, embeddings }
}

/** Load the persisted index. Returns undefined when it is absent, corrupt,
 *  built for another wiki path, or built with another model — callers then
 *  rebuild from scratch (MVP: no incremental path). */
export async function loadWikiRagIndex(wikiRealPath: string): Promise<WikiRagIndex | undefined> {
  const file = wikiRagIndexPath()
  if (!(await fileExists(file))) return undefined
  let index: WikiRagIndex
  try {
    index = JSON.parse(await fs.readFile(file, 'utf8')) as WikiRagIndex
  } catch {
    debugLog('wiki-rag.load-corrupt', `discarding unreadable index at ${file}`)
    return undefined
  }
  if (index.wikiRealPath !== wikiRealPath || index.modelId !== WIKI_RAG_MODEL_ID) return undefined
  if (!Array.isArray(index.chunks) || index.chunks.length === 0) return undefined
  if (index.embeddings?.length !== index.chunks.length) return undefined
  return index
}

export async function saveWikiRagIndex(index: WikiRagIndex): Promise<void> {
  await fs.mkdir(userYgyDir(), { recursive: true })
  await fs.writeFile(wikiRagIndexPath(), JSON.stringify(index), 'utf8')
}

function dotProduct(a: Float32Array | number[], b: Float32Array | number[]): number {
  const length = Math.min(a.length, b.length)
  let sum = 0
  for (let i = 0; i < length; i++) sum += a[i] * b[i]
  return sum
}

export interface WikiRagHit {
  chunk: WikiRagChunk
  /** Cosine similarity — vectors are normalized, so this is a dot product. */
  score: number
}

/** Embed the query and score it against every chunk vector, top-K desc. */
export async function searchWikiRag(
  index: WikiRagIndex,
  embedder: Embedder,
  query: string,
  topK = 5,
): Promise<WikiRagHit[]> {
  const [queryVector] = await embedder([query])
  const hits = index.chunks.map((chunk, i) => ({
    chunk,
    score: dotProduct(queryVector, index.embeddings[i] ?? []),
  }))
  hits.sort((a, b) => b.score - a.score)
  return hits.slice(0, topK)
}

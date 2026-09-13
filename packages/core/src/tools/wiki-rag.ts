// @ygy-code/core — wikiRag tool (semantic search over the external wiki)
//
// Sibling of wikiMemory: wikiMemory delivers the wiki usage SOP, this tool
// does local embedding-vector retrieval over the same wiki so the model can
// find passages whose wording differs from the query (synonyms, paraphrase,
// cross-language) where grep fails. Registered in the same conditional block
// (root agent, `wiki` config set) — see loop.ts.
//
// All heavy lifting lives in knowledge/wiki-rag.ts (chunking, index build /
// load, search). The embedder is injected; production uses Transformers.js,
// tests use deterministic fakes so no test ever loads a model or touches the
// network. Like wikiMemory this maps to `content-read` (it only reads the
// wiki and the local index cache).
//
// Execute NEVER throws across the tool boundary — any failure (model load,
// index build, unreadable wiki) degrades to a short message pointing the
// model back at grep with the wiki's real path.
import fs from 'node:fs/promises'

import { tool } from 'ai'

import { z } from 'zod'

import type { WikiMemory } from '../knowledge/wiki-memory.js'
import {
  type Embedder,
  buildWikiRagIndex,
  createTransformersEmbedder,
  loadWikiRagIndex,
  saveWikiRagIndex,
  searchWikiRag,
} from '../knowledge/wiki-rag.js'

export const WIKI_RAG_TOOL_NAME = 'wikiRag'

const TOP_K = 5
/** Preview length per hit; the model reads full context via readFile. */
const PREVIEW_CHARS = 400

export function createWikiRagTool(wiki: WikiMemory, embedder: Embedder = createTransformersEmbedder()) {
  return tool({
    description:
      "Semantic vector search over the user's external Markdown wiki (their long-term memory). " +
      'Returns the 5 most relevant wiki sections as file path + heading chain + line range + text preview. ' +
      'Prefer it as the FIRST retrieval step when the task touches past experience, personal preferences, ' +
      'or project archives and you do not already know the exact file or keyword — it matches by MEANING, ' +
      'so paraphrases, vague recollections, and differently-worded notes still hit. ' +
      'Falls back to grep when you know the exact keyword, or to index.md for the topic map. ' +
      'After the results, read the exact source lines with readFile. ' +
      'The first call builds the local index and may take a while.',
    inputSchema: z.object({
      query: z
        .string()
        .min(1)
        .describe(
          'Semantic query describing the kind of past note, experience, preference, or project record to find (same language as the task is fine).',
        ),
    }),
    execute: async ({ query }) => {
      let realPath = wiki.path
      try {
        realPath = await fs.realpath(wiki.path)
      } catch {
        // Keep the configured path for the degrade message below.
      }
      try {
        let index = await loadWikiRagIndex(realPath)
        let buildNote = ''
        if (!index) {
          const startedAt = Date.now()
          index = await buildWikiRagIndex(realPath, embedder)
          await saveWikiRagIndex(index)
          const seconds = ((Date.now() - startedAt) / 1000).toFixed(1)
          buildNote = `（首次构建完成，${index.chunks.length} chunks，耗时 ${seconds}s）\n\n`
        }
        const hits = await searchWikiRag(index, embedder, query, TOP_K)
        const sections = hits.map((hit, i) => {
          const preview =
            hit.chunk.text.length > PREVIEW_CHARS ? `${hit.chunk.text.slice(0, PREVIEW_CHARS)}…` : hit.chunk.text
          const indented = preview
            .split('\n')
            .map((line) => `    ${line}`)
            .join('\n')
          return `[${i + 1}] ${hit.chunk.file} :${hit.chunk.startLine}-${hit.chunk.endLine} — ${hit.chunk.title}（score ${hit.score.toFixed(2)}）\n${indented}`
        })
        return (
          `wiki 语义检索结果（query: ${query}）\n\n` +
          buildNote +
          sections.join('\n\n') +
          `\n\n提示：对相关段落用 readFile 按行号区间精读原文；需要精确关键词检索时改用 grep。`
        )
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error)
        return `wiki-rag 不可用：${reason}。请改用 grep 检索 wiki：${realPath}`
      }
    },
  })
}

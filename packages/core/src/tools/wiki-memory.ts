// @ygy-code/core — wikiMemory tool (progressive-disclosure entry point)
//
// Registered only when config `wiki` is set (same conditional-registration
// pattern as activateSkill). The full wiki usage SOP (~700 tokens) is NOT
// resident in every system prompt — the knowledge section carries a one-line
// pointer instead, and this tool delivers the SOP on demand as a tool result.
// Mirrors the activateSkill dual-channel pattern: short resident hint + tool
// that injects the full body.
//
// No new tools are mounted on activation — the SOP directs the model to the
// existing grep/readFile tools. The tool table therefore stays byte-stable
// for the whole session, keeping provider prompt/KV prefix caches valid
// (cf. dsh-vision-router docs/progressive-tools-cache.md, which keeps its
// tool table stable for exactly this reason).
//
// execute only reads the filesystem (realpath resolution) — no mutation,
// no network — so it maps to the same `content-read` capability as
// activateSkill.
import { tool } from 'ai'

import { z } from 'zod'

import { type WikiMemory, buildWikiMemoryContext } from '../knowledge/wiki-memory.js'

export const WIKI_MEMORY_TOOL_NAME = 'wikiMemory'

export function createWikiMemoryTool(wiki: WikiMemory) {
  return tool({
    description:
      "Load usage instructions for the external Markdown wiki that serves as this user's long-term memory. " +
      'Call it ONCE when a task needs prior context: past decisions, personal preferences, recurring workflows, ' +
      'project archives, or anything about "what we did / agreed / recorded before". ' +
      'After the call, follow the returned instructions (grep the wiki, read PageIndex trees, then read files by line range). ' +
      'Do NOT call it for ordinary coding tasks with no reference to past history.',
    inputSchema: z.object({}).strict(),
    execute: async () => {
      const instructions = await buildWikiMemoryContext(wiki)
      if (!instructions) {
        return (
          'The configured wiki memory path is not usable (missing or not a directory). ' +
          'Tell the user to fix their `wiki.path` configuration; proceed without long-term memory for now.'
        )
      }
      return `<wiki_memory_instructions>\n${instructions}\n</wiki_memory_instructions>`
    },
  })
}

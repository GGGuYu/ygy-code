// @ygy-code/core — External wiki memory context
//
// When the user enables `wiki` in config, the agent's long-term memory is an
// external Markdown wiki (Echo-style: index.md navigation + grep retrieval +
// .pageindex pre-filter trees + user-gated writes) instead of Memory v2's
// user-scope auto memory. This module builds the always-injected knowledge
// section that tells the model how to use that wiki.
//
// AGENTS.md / CLAUDE.md loading is intentionally untouched: the wiki layer
// replaces only the "User Auto Memory" profile in the knowledge chain.
import fs from 'node:fs/promises'
import path from 'node:path'

import { debugLog, fileExists, isDir } from '../utils.js'

export interface WikiMemory {
  /** Configured wiki root (symlink allowed). */
  path: string
}

const WIKI_MEMORY_RULES = `长期记忆系统 = 这个外部 Markdown wiki（Memory v2 自动记忆不启用）。AGENTS.md 仍按原知识链注入；本段负责“wiki 怎么用”。

工作目录（配置来源）：
- 配置路径：{path}
- 真实路径：{realPath}（grep/脚本请用真实路径；配置路径是符号链接时，grep -r 默认不跟随）
- 主索引：index.md（约 200 行；新会话/新任务先整读，含底部人格与偏好）

使用规则：
1. 新任务先读 {realPath}/index.md 全文，再按关键词 grep 检索是否已有经验；复用优先。
   grep 命令示例：grep -rn "关键词" {realPath}
1b. 语义检索（wikiRag 工具）：问题涉及过往经验/偏好/项目档案、但你拿不准精确关键词或记不清措辞时，
   调用 wikiRag 工具做本地向量检索——按语义匹配返回最相关的 5 个段落（文件+行号区间+预览），
   同义改写、模糊回忆也能命中；拿到结果后按行号 readFile 精读。掌握精确关键词时 grep 更快。
2. 检索定位到正文 md 后，先检查 PageIndex 树：{realPath}/.pageindex/<正文相对路径去掉 .md>.tree.json。
   树存在 → 先整读该树（标题/层级/行号），再按行号只读需要的原文段落；
   树不存在 ≠ 文章不存在（白名单/短文没有树），直接读原文。
3. index 是导航不是仓库：摘要够用就不读全文；长文先读树，不整篇吞。
4. 写作与维护规范遵循 {realPath}/tools/wiki-standards.md（若存在）：头部状态块分级、动机→目标→进度→额外点、相对链接、死链为零、不做自动删改。
5. 查询口径参考 guyu-feishu-llm-wiki skill（Karpathy LLM Wiki 模式）：先读 index 定位 → 读相关文章 → 综合回答并注明来源；默认只输出到对话，不写文件。
   技能原文：~/.agents/skills/guyu-feishu-llm-wiki/SKILL.md（本机存在时优先遵循其 Query/维护约定）。
6. 写入是用户把关动作：只有用户明确要求“记一下/记住/写进 wiki/更新某档案”时才写；写前先读相关现有文章与 index，判断新增还是更新；不自动写、不自动删；删除必须用户明确确认。
7. 回答引用 wiki 内容时给出相对来源路径（如 personal/xxx.md）；wiki 没有相关记录时明说“我查了 wiki 没有相关记录”，再给通用理解。`

/** Build the markdown body injected under `### Wiki 记忆`. Returns '' when
 *  the configured path is unusable so the loader can fall back to no-op. */
export async function buildWikiMemoryContext(wiki: WikiMemory): Promise<string> {
  if (!wiki.path) return ''
  if (!(await isDir(wiki.path))) {
    debugLog('wiki-memory.invalid-path', `wiki path is not a directory: ${wiki.path}`)
    return ''
  }
  let realPath = wiki.path
  try {
    realPath = await fs.realpath(wiki.path)
  } catch {
    debugLog('wiki-memory.realpath-failed', `could not resolve real path for ${wiki.path}; using configured path`)
  }
  const standardsPath = path.join(realPath, 'tools', 'wiki-standards.md')
  const standardsAvailable = await fileExists(standardsPath)
  const rules = WIKI_MEMORY_RULES.replace(/\{path\}/g, () => wiki.path).replace(/\{realPath\}/g, () => realPath)
  return `${rules}${standardsAvailable ? '' : '\n\n（注意：当前 wiki 没有 tools/wiki-standards.md，仍遵循上述通用规范。）'}`
}

/** One-line resident pointer for the knowledge section. Keeps the wiki
 *  discoverable without paying the full SOP on every turn; the SOP itself
 *  arrives via the wikiMemory tool when the model decides it needs long-term
 *  memory. Same dual-channel shape as the Available Skills block +
 *  activateSkill. Returns '' when the path is unusable (loader skips the
 *  section entirely). */
export async function buildWikiMemoryPointer(wiki: WikiMemory): Promise<string> {
  if (!wiki.path) return ''
  if (!(await isDir(wiki.path))) return ''
  return `外部 Markdown wiki 长期记忆已配置（${wiki.path}）。任务涉及过往经验、个人偏好、项目档案等历史上下文时，先调用 wikiMemory 工具获取使用说明，再按说明检索 wiki。普通编码任务无需调用。`
}

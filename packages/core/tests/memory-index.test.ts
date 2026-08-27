import path from 'node:path'

import {
  MemoryIndex,
  extractMemoryIdentifiers,
  normalizeMemoryText,
  tokenizeMemoryText,
} from '../src/knowledge/memory/search-index.js'
import { parseMemoryTopic } from '../src/knowledge/memory/store.js'
import { topicMarkdown } from './memory-test-helpers.js'

describe('MemoryIndex', () => {
  it('normalizes Unicode, identifiers, Windows paths, and script-independent n-grams', () => {
    expect(normalizeMemoryText('D:\\Res\\ygy-code')).toBe('d:/res/x code cli')
    expect(tokenizeMemoryText('MemoryService αβγδ')).toEqual(
      expect.arrayContaining(['memoryservice', 'memory', 'service', 'αβ', 'βγ', 'αβγ']),
    )
    expect(tokenizeMemoryText('ygy-code')).toEqual(expect.arrayContaining(['ygy-code', 'x', 'code', 'cli']))
    expect(tokenizeMemoryText('αβTypeScriptγδ')).toEqual(expect.arrayContaining(['αβ', 'γδ']))
    expect(extractMemoryIdentifiers('react-dom failed in @scope/tool with TS2322')).toEqual(
      expect.arrayContaining(['react-dom', '@scope/tool', 'TS2322']),
    )
  })

  it('retrieves aliases and repository paths exactly and ranks lexical content', () => {
    const topic = parseMemoryTopic(
      topicMarkdown({
        id: 'product-portfolio',
        type: 'portfolio',
        aliases: ['ygy-code'],
        keywords: ['MemoryService', 'TypeScript'],
        appliesTo: ['D:/res/ygy-code'],
        facts: [{ id: 'portfolio.ygy-code.stack', content: '- Uses TypeScript and pnpm workspace.' }],
      }),
      path.join('C:/memory/topics', 'product-portfolio.md'),
    )
    const index = new MemoryIndex()
    index.rebuild([topic], 7)

    expect([...index.exactHits('ygy-code details', [], [])].map(([id]) => id)).toEqual(['product-portfolio'])
    expect(index.exactHits('repo', ['D:/res/ygy-code'], []).has('product-portfolio')).toBe(true)
    expect(index.bm25('TypeScript pnpm')[0]?.topicId).toBe('product-portfolio')
    expect(index.facts.get('portfolio.ygy-code.stack')?.topicId).toBe('product-portfolio')
  })

  it('does not index stale or expired fact bodies as current knowledge', () => {
    const topic = parseMemoryTopic(
      topicMarkdown({
        id: 'profile',
        type: 'reference',
        facts: [
          { id: 'user.old', content: '- obsoletequux preference.', status: 'stale' },
          { id: 'user.expired', content: '- expiredquux preference.', expiresAt: '2020-01-01T00:00:00.000Z' },
          { id: 'user.current', content: '- currentquux preference.' },
        ],
      }),
      path.join('C:/memory/topics', 'profile.md'),
    )
    const index = new MemoryIndex()
    index.rebuild([topic], 1)

    expect(index.bm25('obsoletequux')).toEqual([])
    expect(index.bm25('expiredquux')).toEqual([])
    expect(index.bm25('currentquux')[0]?.topicId).toBe('profile')
    expect(index.facts.has('user.expired')).toBe(true)
  })
})

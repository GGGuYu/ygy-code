import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import path from 'node:path'

import {
  installedPluginsPath,
  knownMarketplacesPath,
  marketplaceDir,
  pluginCacheDir,
  pluginDataDir,
  pluginsRoot,
} from '../src/plugins/paths.js'

describe('plugins/paths honors YGY_CODE_HOME', () => {
  const originalHome = process.env.YGY_CODE_HOME
  const originalPluginsDir = process.env.YGY_PLUGINS_DIR

  beforeEach(() => {
    delete process.env.YGY_CODE_HOME
    delete process.env.YGY_PLUGINS_DIR
  })

  afterEach(() => {
    if (originalHome === undefined) delete process.env.YGY_CODE_HOME
    else process.env.YGY_CODE_HOME = originalHome
    if (originalPluginsDir === undefined) delete process.env.YGY_PLUGINS_DIR
    else process.env.YGY_PLUGINS_DIR = originalPluginsDir
  })

  it('routes pluginsRoot through YGY_CODE_HOME when set', () => {
    process.env.YGY_CODE_HOME = '/tmp/ygy-sandbox'
    expect(pluginsRoot()).toBe(path.join('/tmp/ygy-sandbox', 'plugins'))
  })

  it('YGY_PLUGINS_DIR still wins over YGY_CODE_HOME (plugin-specific is more specific)', () => {
    process.env.YGY_CODE_HOME = '/tmp/home'
    process.env.YGY_PLUGINS_DIR = '/tmp/just-plugins'
    expect(pluginsRoot()).toBe('/tmp/just-plugins')
  })

  it('downstream path helpers inherit the YGY_CODE_HOME redirect', () => {
    // The actual side-finding: an `ygy plugin list` under YGY_CODE_HOME=tmp
    // was still reading from ~/.ygy-code/plugins/installed_plugins.json
    // because pluginsRoot() called the frozen USER_YGY_DIR constant.
    process.env.YGY_CODE_HOME = '/tmp/ygy-sandbox'
    const root = '/tmp/ygy-sandbox/plugins'
    expect(knownMarketplacesPath()).toBe(path.join(root, 'known_marketplaces.json'))
    expect(installedPluginsPath()).toBe(path.join(root, 'installed_plugins.json'))
    expect(marketplaceDir('anthropic')).toBe(path.join(root, 'marketplaces', 'anthropic'))
    expect(pluginCacheDir('m', 'p', '1.0.0')).toBe(path.join(root, 'cache', 'm', 'p', '1.0.0'))
    expect(pluginDataDir('foo@bar')).toBe(path.join(root, 'data', 'foo@bar'))
  })

  it('falls through to ~/.ygy-code/plugins when no override is set', () => {
    // Don't pin an exact string (HOME varies across machines); just assert
    // the suffix and that we didn't pick up a stray override.
    expect(pluginsRoot().endsWith(path.join('.ygy-code', 'plugins'))).toBe(true)
  })

  it.each(['../outside', '..\\outside', '/absolute', 'C:\\absolute', 'nul', 'name.'])(
    'rejects unsafe cache path component %s',
    (component) => {
      expect(() => pluginCacheDir('market', 'plugin', component)).toThrow(/safe filesystem path component/)
      expect(() => marketplaceDir(component)).toThrow(/safe filesystem path component/)
    },
  )
})

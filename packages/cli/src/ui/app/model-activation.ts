import { createModelRegistry } from '@ygy-code/core'
import type { AgentOptions, LanguageModel } from '@ygy-code/core'

export function replaceActiveModelProvider(
  modelId: string,
  options: Pick<AgentOptions, 'modelRegistry'>,
  switchModel: (modelId: string, model: LanguageModel) => void,
): void {
  const registry = createModelRegistry()
  const model = registry.languageModel(modelId as `${string}:${string}`)
  options.modelRegistry = registry
  switchModel(modelId, model)
}

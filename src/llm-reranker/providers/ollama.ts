import ollama, { ChatResponse, GenerateRequest, GenerateResponse } from "ollama";
import { ModelProvider, ModelUsage } from ".";
import { performance } from "perf_hooks";

export class ProviderOllama implements ModelProvider {
  model: string;
  apiKey?: string;

  name = "ollama";
  //Add a pull system to check what models have ollama
  validModels = ["sam860/qwen3-reranker:0.6b-F16","dengcao/Qwen3-Reranker-0.6B:F16","gemma3:1b","qwen3:4b"];

  constructor(model: string, apiKey: string) {
    this.model = model;
    this.apiKey = apiKey;
  }

  /**
   * Creates a completion request using the Groq SDK.
   * @param input Prompt to infer the completion.
   * @returns Text completion from the language model.
   */
  public async infer(
    input: string
  ): Promise<{ output: string; usage: ModelUsage }> {
    //const openai = new OpenAI({ apiKey: this.apiKey });

    const startTime = performance.now();
    const completion: GenerateResponse = await ollama.generate({
      model: this.model,
      prompt: input,
    })
    const completionTime = performance.now() - startTime;

    return {
      output: completion.response || "",
      usage: {
        completionTokens: completion.prompt_eval_count,
        promptTokens: completion.eval_count,
        completionTime,
      },
    };
  }
}

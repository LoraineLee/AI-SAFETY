import { LanguageModel, LanguageModelStreamPart } from '../language-model';
import { LanguageModelPrompt } from '../prompt';
import { StreamTextHttpResponse } from './stream-text-http-response';

/**
 * Stream text generated by a language model.
 */
export async function streamText({
  model,
  prompt,
}: {
  model: LanguageModel;
  prompt: LanguageModelPrompt;
}): Promise<StreamTextResult> {
  const modelStream = await model.doStream({
    prompt,
  });

  return new StreamTextResult(modelStream);
}

export class StreamTextResult {
  private readonly modelStream: ReadableStream<LanguageModelStreamPart>;

  readonly textStream: AsyncIterable<string>;

  constructor(modelStream: ReadableStream<LanguageModelStreamPart>) {
    this.modelStream = modelStream;

    this.textStream = {
      [Symbol.asyncIterator](): AsyncIterator<string> {
        const reader = modelStream.getReader();
        return {
          next: async () => {
            // loops until a text delta is found or the stream is finished:
            while (true) {
              const { done, value } = await reader.read();

              if (done) {
                return { value: null, done: true };
              }

              if (value.type === 'text-delta') {
                return { value: value.textDelta, done: false };
              }
            }
          },
        };
      },
    };
  }

  toResponse() {
    return new StreamTextHttpResponse(this.modelStream);
  }
}

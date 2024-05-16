import {
  AIStreamCallbacksAndOptions,
  createCallbacksTransformer,
} from './ai-stream';
import { createStreamDataTransformer } from './stream-data';

type LangChainImageDetail = 'auto' | 'low' | 'high';

type LangChainMessageContentText = {
  type: 'text';
  text: string;
};

type LangChainMessageContentImageUrl = {
  type: 'image_url';
  image_url:
    | string
    | {
        url: string;
        detail?: LangChainImageDetail;
      };
};

type LangChainMessageContentComplex =
  | LangChainMessageContentText
  | LangChainMessageContentImageUrl
  | (Record<string, any> & {
      type?: 'text' | 'image_url' | string;
    })
  | (Record<string, any> & {
      type?: never;
    });

type LangChainMessageContent = string | LangChainMessageContentComplex[];

type LangChainAIMessageChunk = {
  content: LangChainMessageContent;
};

/**
Converts the result of a LangChain Expression Language stream invocation to an AIStream.
 */
export function toAIStream(
  stream: ReadableStream<LangChainAIMessageChunk>,
  callbacks?: AIStreamCallbacksAndOptions,
) {
  return stream
    .pipeThrough(
      new TransformStream({
        transform: async (chunk, controller) => {
          if (typeof chunk.content === 'string') {
            controller.enqueue(chunk.content);
          } else {
            const content: LangChainMessageContentComplex[] = chunk.content;
            for (const item of content) {
              if (item.type === 'text') {
                controller.enqueue(item.text);
              }
            }
          }
        },
      }),
    )
    .pipeThrough(createCallbacksTransformer(callbacks))
    .pipeThrough(createStreamDataTransformer());
}

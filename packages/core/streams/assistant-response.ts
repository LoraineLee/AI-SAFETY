import { AssistantMessage } from '../shared/types';

export function experimental_AssistantResponse(
  { threadId, messageId }: { threadId: string; messageId: string },
  process: (stream: {
    threadId: string;
    messageId: string;
    sendMessage: (message: AssistantMessage) => void;
  }) => Promise<void>,
): Response {
  const stream = new ReadableStream({
    async start(controller) {
      const textEncoder = new TextEncoder();

      const sendMessage = (message: AssistantMessage) => {
        controller.enqueue(
          textEncoder.encode(`0: ${JSON.stringify(message)}\n\n`),
        );
      };

      const sendError = (errorMessage: string) => {
        controller.enqueue(
          textEncoder.encode(`3: ${JSON.stringify(errorMessage)}\n\n`),
        );
      };

      // send the threadId and messageId as the first message:
      controller.enqueue(
        textEncoder.encode(
          `4: ${JSON.stringify({
            threadId,
            messageId,
          })}\n\n`,
        ),
      );

      try {
        await process({
          threadId,
          messageId,
          sendMessage,
        });
      } catch (error) {
        sendError((error as any).message ?? `${error}`);
      } finally {
        controller.close();
      }
    },
    pull(controller) {},
    cancel() {},
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  });
}

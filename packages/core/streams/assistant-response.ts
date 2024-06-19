import {
  AssistantMessage,
  AssistantThreadStatus,
  DataMessage,
  formatStreamPart,
} from '@ai-sdk/ui-utils';
import { type AssistantStream } from 'openai/lib/AssistantStream';
import { ErrorObject } from 'openai/resources';
import { Run, RunStatus } from 'openai/resources/beta/threads/runs/runs';
import {
  MessageDeltaEvent,
  Message as ThreadMessage,
} from 'openai/resources/beta/threads/messages';
import {
  RunStep,
  RunStepDeltaEvent,
} from 'openai/resources/beta/threads/runs/steps';
import { Thread } from 'openai/resources/beta/threads/threads';

/**
You can pass the thread and the latest message into the `AssistantResponse`. This establishes the context for the response.
 */
type AssistantResponseSettings = {
  /**
The thread ID that the response is associated with.
   */
  threadId: string;

  /**
The ID of the latest message that the response is associated with.
 */
  messageId: string;
};

/**
The process parameter is a callback in which you can run the assistant on threads, and send messages and data messages to the client.
 */
type AssistantResponseCallback = (options: {
  /**
@deprecated use variable from outer scope instead.
   */
  threadId: string;

  /**
@deprecated use variable from outer scope instead.
   */
  messageId: string;

  /**
Forwards an assistant message (non-streaming) to the client.
   */
  sendMessage: (message: AssistantMessage) => void;

  /**
Send a data message to the client. You can use this to provide information for rendering custom UIs while the assistant is processing the thread.
 */
  sendDataMessage: (message: DataMessage) => void;

  /**
Forwards the assistant response stream to the client. Returns the `Run` object after it completes, or when it requires an action.
   */
  forwardStream: (stream: AssistantStream) => Promise<Run | undefined>;
}) => Promise<void>;

/**
 * The `StreamDataPart` represents the possible types of data parts present in an assistant stream event.
 */
type StreamDataPart =
  | null
  | {
      id: string;
      threadId: string;
      requiredAction: Run.RequiredAction;
      status: RunStatus;
    }
  | { id: string }
  | { id: string; delta: { content: [{ text: { value: string } }] } };

/**
The `AssistantResponse` allows you to send a stream of assistant update to `useAssistant`.
It is designed to facilitate streaming assistant responses to the `useAssistant` hook.
It receives an assistant thread and a current message, and can send messages and data messages to the client.
 */
export function AssistantResponse(
  { threadId, messageId }: AssistantResponseSettings,
  process: AssistantResponseCallback,
): Response {
  const stream = new ReadableStream({
    async start(controller) {
      const textEncoder = new TextEncoder();

      const sendMessage = (message: AssistantMessage) => {
        controller.enqueue(
          textEncoder.encode(formatStreamPart('assistant_message', message)),
        );
      };

      const sendDataMessage = (message: DataMessage) => {
        controller.enqueue(
          textEncoder.encode(formatStreamPart('data_message', message)),
        );
      };

      const sendError = (errorMessage: string) => {
        controller.enqueue(
          textEncoder.encode(formatStreamPart('error', errorMessage)),
        );
      };

      const forwardStream = async (stream: AssistantStream) => {
        let result: Run | undefined = undefined;

        function sanitizeData(
          data:
            | Run
            | RunStep
            | RunStepDeltaEvent
            | Thread
            | ThreadMessage
            | MessageDeltaEvent
            | ErrorObject,
        ) {
          if (!('object' in data)) {
            return null;
          }

          const { object } = data;

          switch (object) {
            case 'thread.run': {
              result = data as Run;

              return {
                id: data.id,
                threadId: data.thread_id,
                requiredAction: data.required_action,
                status: data.status,
              };
            }

            case 'thread.message': {
              return {
                id: data.id,
              };
            }

            case 'thread.message.delta': {
              return {
                id: data.id,
                delta: data.delta,
              };
            }

            default:
              return null;
          }
        }

        for await (const { event, data } of stream) {
          let streamPart: {
            event: AssistantThreadStatus;
            data?: StreamDataPart;
          } = {
            event,
          };

          const streamDataPart = sanitizeData(data);

          if (streamDataPart) {
            streamPart['data'] = streamDataPart;
          }

          controller.enqueue(
            textEncoder.encode(formatStreamPart('assistant_event', streamPart)),
          );
        }

        return result;
      };

      try {
        await process({
          threadId,
          messageId,
          sendMessage,
          sendDataMessage,
          forwardStream,
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

/**
@deprecated Use `AssistantResponse` instead.
 */
export const experimental_AssistantResponse = AssistantResponse;

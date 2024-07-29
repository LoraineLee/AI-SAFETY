import type { RequestHandler } from './$types';
import { z } from 'zod';
import { convertToCoreMessages, StreamData, StreamingTextResponse, streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

export const POST = (async ({ request }) => {
    const r = await request.json();
    const { messages } = r;

    const result = await streamText({
        model: openai('gpt-4-turbo'),
        messages: convertToCoreMessages(messages),
        tools: {
          // server-side tool with execute function:
          getWeatherInformation: {
            description: 'show the weather in a given city to the user',
            parameters: z.object({ city: z.string() }),
            execute: async ({}: { city: string }) => {
              const weatherOptions = ['sunny', 'cloudy', 'rainy', 'snowy', 'windy'];
              return weatherOptions[
                Math.floor(Math.random() * weatherOptions.length)
              ];
            },
          },
          // client-side tool that starts user interaction:
          askForConfirmation: {
            description: 'Ask the user for confirmation.',
            parameters: z.object({
              message: z.string().describe('The message to ask for confirmation.'),
            }),
          },
          // client-side tool that is automatically executed on the client:
          getLocation: {
            description:
              'Get the user location. Always ask for confirmation before using this tool.',
            parameters: z.object({}),
          },
        },
      });
    

    return result.toAIStreamResponse();
}) satisfies RequestHandler;

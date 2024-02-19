import {
  OpenAIStream,
  StreamingTextResponse,
  experimental_StreamData,
  experimental_forwardLmntSpeechStream,
} from 'ai';
import Speech from 'lmnt-node';
import OpenAI from 'openai';

// Create an OpenAI API client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || 'no key',
});

// Create an LMNT API client
const speech = new Speech(process.env.LMNT_API_KEY || 'no key');

// Note: The LMNT SDK does not work on edge (v1.1.2)
// export const runtime = 'edge';

export async function POST(req: Request) {
  // Extract the `prompt` from the body of the request
  const { prompt } = await req.json();

  // Ask OpenAI for a streaming completion given the prompt
  const response = await openai.completions.create({
    model: 'gpt-3.5-turbo-instruct',
    max_tokens: 2000,
    stream: true,
    prompt,
  });

  const speechStream = speech.synthesizeStreaming('lily', {});

  const data = new experimental_StreamData();

  // note: no await here, we want to run this in parallel:
  experimental_forwardLmntSpeechStream(speechStream, data, {
    onFinal() {
      data.close();
    },
  });

  // Convert the response into a friendly text-stream
  const stream = OpenAIStream(response, {
    onToken(token) {
      speechStream.appendText(token);

      // only flush after each sentence to make it sound more natural:
      if (token.includes('.')) {
        speechStream.flush();
      }
    },
    onFinal(completion) {
      speechStream.finish(); // flush any remaining tokens and close the stream
    },
    experimental_streamData: true,
  });

  // Respond with the stream
  return new StreamingTextResponse(stream, {}, data);
}

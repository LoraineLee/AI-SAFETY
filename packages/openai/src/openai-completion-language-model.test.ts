import { LanguageModelV1Prompt } from '@ai-sdk/provider';
import {
  JsonTestServer,
  StreamingTestServer,
  convertStreamToArray,
} from '@ai-sdk/provider-utils/test';
import { createOpenAI } from './openai-provider';

const TEST_PROMPT: LanguageModelV1Prompt = [
  { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
];

const provider = createOpenAI({
  apiKey: 'test-api-key',
});

describe('doGenerate', () => {
  const server = new JsonTestServer('https://api.openai.com/v1/completions');

  server.setupTestEnvironment();

  function prepareJsonResponse({
    content = '',
    usage = {
      prompt_tokens: 4,
      total_tokens: 34,
      completion_tokens: 30,
    },
  }: {
    content?: string;
    usage?: {
      prompt_tokens: number;
      total_tokens: number;
      completion_tokens: number;
    };
  }) {
    server.responseBodyJson = {
      id: 'cmpl-96cAM1v77r4jXa4qb2NSmRREV5oWB',
      object: 'text_completion',
      created: 1711363706,
      model: 'gpt-3.5-turbo-instruct',
      choices: [
        {
          text: content,
          index: 0,
          logprobs: null,
          finish_reason: 'stop',
        },
      ],
      usage,
    };
  }

  it('should extract text response', async () => {
    prepareJsonResponse({ content: 'Hello, World!' });

    const { text } = await provider
      .completion('gpt-3.5-turbo-instruct')
      .doGenerate({
        inputFormat: 'prompt',
        mode: { type: 'regular' },
        prompt: TEST_PROMPT,
      });

    expect(text).toStrictEqual('Hello, World!');
  });

  it('should extract usage', async () => {
    prepareJsonResponse({
      content: '',
      usage: { prompt_tokens: 20, total_tokens: 25, completion_tokens: 5 },
    });

    const { usage } = await provider
      .completion('gpt-3.5-turbo-instruct')
      .doGenerate({
        inputFormat: 'prompt',
        mode: { type: 'regular' },
        prompt: TEST_PROMPT,
      });

    expect(usage).toStrictEqual({
      promptTokens: 20,
      completionTokens: 5,
    });
  });

  it('should pass the model and the prompt', async () => {
    prepareJsonResponse({ content: '' });

    await provider.completion('gpt-3.5-turbo-instruct').doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(await server.getRequestBodyJson()).toStrictEqual({
      model: 'gpt-3.5-turbo-instruct',
      prompt: 'Hello',
    });
  });

  it('should pass custom headers', async () => {
    prepareJsonResponse({ content: '' });

    const provider = createOpenAI({
      apiKey: 'test-api-key',
      organization: 'test-organization',
      project: 'test-project',
      headers: {
        'Custom-Header': 'test-header',
      },
    });

    await provider.completion('gpt-3.5-turbo-instruct').doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    const requestHeaders = await server.getRequestHeaders();

    expect(requestHeaders.get('OpenAI-Organization')).toStrictEqual(
      'test-organization',
    );
    expect(requestHeaders.get('OpenAI-Project')).toStrictEqual('test-project');
    expect(requestHeaders.get('Custom-Header')).toStrictEqual('test-header');
  });

  it('should pass the api key as Authorization header', async () => {
    prepareJsonResponse({ content: '' });

    const provider = createOpenAI({ apiKey: 'test-api-key' });

    await provider.completion('gpt-3.5-turbo-instruct').doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(
      (await server.getRequestHeaders()).get('Authorization'),
    ).toStrictEqual('Bearer test-api-key');
  });
});

describe('doStream', () => {
  const server = new StreamingTestServer(
    'https://api.openai.com/v1/completions',
  );

  server.setupTestEnvironment();

  function prepareStreamResponse({ content }: { content: string[] }) {
    server.responseChunks = [
      ...content.map(text => {
        return (
          `data: {"id":"cmpl-96c64EdfhOw8pjFFgVpLuT8k2MtdT","object":"text_completion","created":1711363440,` +
          `"choices":[{"text":"${text}","index":0,"logprobs":null,"finish_reason":null}],"model":"gpt-3.5-turbo-instruct"}\n\n`
        );
      }),
      `data: {"id":"cmpl-96c3yLQE1TtZCd6n6OILVmzev8M8H","object":"text_completion","created":1711363310,` +
        `"choices":[{"text":"","index":0,"logprobs":null,"finish_reason":"stop"}],"model":"gpt-3.5-turbo-instruct"}\n\n`,
      `data: {"id":"cmpl-96c3yLQE1TtZCd6n6OILVmzev8M8H","object":"text_completion","created":1711363310,` +
        `"model":"gpt-3.5-turbo-instruct","usage":{"prompt_tokens":10,"completion_tokens":362,"total_tokens":372},"choices":[]}\n\n`,
      'data: [DONE]\n\n',
    ];
  }

  it('should stream text deltas', async () => {
    prepareStreamResponse({ content: ['Hello', ', ', 'World!'] });

    const { stream } = await provider
      .completion('gpt-3.5-turbo-instruct')
      .doStream({
        inputFormat: 'prompt',
        mode: { type: 'regular' },
        prompt: TEST_PROMPT,
      });

    // note: space moved to last chunk bc of trimming
    expect(await convertStreamToArray(stream)).toStrictEqual([
      { type: 'text-delta', textDelta: 'Hello' },
      { type: 'text-delta', textDelta: ', ' },
      { type: 'text-delta', textDelta: 'World!' },
      { type: 'text-delta', textDelta: '' },
      {
        type: 'finish',
        finishReason: 'stop',
        usage: { promptTokens: 10, completionTokens: 362 },
      },
    ]);
  });

  it('should pass the model and the prompt', async () => {
    prepareStreamResponse({ content: [] });

    await provider.completion('gpt-3.5-turbo-instruct').doStream({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(await server.getRequestBodyJson()).toStrictEqual({
      stream: true,
      model: 'gpt-3.5-turbo-instruct',
      prompt: 'Hello',
    });
  });

  it('should pass custom headers', async () => {
    prepareStreamResponse({ content: [] });

    const provider = createOpenAI({
      apiKey: 'test-api-key',
      organization: 'test-organization',
      project: 'test-project',
      headers: {
        'Custom-Header': 'test-header',
      },
    });

    await provider.completion('gpt-3.5-turbo-instruct').doStream({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    const requestHeaders = await server.getRequestHeaders();

    expect(requestHeaders.get('OpenAI-Organization')).toStrictEqual(
      'test-organization',
    );
    expect(requestHeaders.get('OpenAI-Project')).toStrictEqual('test-project');
    expect(requestHeaders.get('Custom-Header')).toStrictEqual('test-header');
  });

  it('should pass the api key as Authorization header', async () => {
    prepareStreamResponse({ content: [] });

    const provider = createOpenAI({ apiKey: 'test-api-key' });

    await provider.completion('gpt-3.5-turbo-instruct').doStream({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(
      (await server.getRequestHeaders()).get('Authorization'),
    ).toStrictEqual('Bearer test-api-key');
  });
});

import { LanguageModelV1Prompt } from '@ai-sdk/provider';
import {
  JsonTestServer,
  StreamingTestServer,
  convertReadableStreamToArray,
} from '@ai-sdk/provider-utils/test';
import { createCohere } from './cohere-provider';

const TEST_PROMPT: LanguageModelV1Prompt = [
  {
    role: 'system',
    content: 'you are a friendly bot!',
  },
  { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
];

const provider = createCohere({ apiKey: 'test-api-key' });
const model = provider('command-r-plus');

describe('doGenerate', () => {
  const server = new JsonTestServer('https://api.cohere.com/v1/chat');

  server.setupTestEnvironment();

  function prepareJsonResponse({
    input = '',
    text = '',
    tool_calls,
    finish_reason = 'COMPLETE',
    tokens = {
      input_tokens: 4,
      output_tokens: 30,
    },
  }: {
    input?: string;
    text?: string;
    tool_calls?: any;
    finish_reason?: string;
    tokens?: {
      input_tokens: number;
      output_tokens: number;
    };
  }) {
    server.responseBodyJson = {
      response_id: '0cf61ae0-1f60-4c18-9802-be7be809e712',
      text,
      generation_id: 'dad0c7cd-7982-42a7-acfb-706ccf598291',
      chat_history: [
        { role: 'USER', message: input },
        { role: 'CHATBOT', message: text },
      ],
      ...(tool_calls ? { tool_calls } : {}),
      finish_reason,
      meta: {
        api_version: { version: '1' },
        billed_units: { input_tokens: 9, output_tokens: 415 },
        tokens,
      },
    };
  }

  it('should extract text response', async () => {
    prepareJsonResponse({ text: 'Hello, World!' });

    const { text } = await model.doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(text).toStrictEqual('Hello, World!');
  });

  it('should extract tool calls', async () => {
    prepareJsonResponse({
      text: 'Hello, World!',
      tool_calls: [
        {
          name: 'test-tool',
          parameters: { value: 'example value' },
        },
      ],
    });

    const { text, toolCalls, finishReason } = await model.doGenerate({
      inputFormat: 'prompt',
      mode: {
        type: 'regular',
        tools: [
          {
            type: 'function',
            name: 'test-tool',
            parameters: {
              type: 'object',
              properties: { value: { type: 'string' } },
              required: ['value'],
              additionalProperties: false,
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
        ],
      },
      prompt: TEST_PROMPT,
    });

    expect(toolCalls).toStrictEqual([
      expect.objectContaining({
        toolCallId: expect.any(String),
        toolCallType: 'function',
        toolName: 'test-tool',
        args: '{"value":"example value"}',
      }),
    ]);
    expect(text).toStrictEqual('Hello, World!');
    expect(finishReason).toStrictEqual('stop');
  });

  it('should extract usage', async () => {
    prepareJsonResponse({
      tokens: { input_tokens: 20, output_tokens: 5 },
    });

    const { usage } = await model.doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(usage).toStrictEqual({
      promptTokens: 20,
      completionTokens: 5,
    });
  });

  it('should extract finish reason', async () => {
    prepareJsonResponse({
      finish_reason: 'MAX_TOKENS',
    });

    const { finishReason } = await model.doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(finishReason).toStrictEqual('length');
  });

  it('should expose the raw response headers', async () => {
    prepareJsonResponse({});

    server.responseHeaders = {
      'test-header': 'test-value',
    };

    const { rawResponse } = await model.doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(rawResponse?.headers).toStrictEqual({
      // default headers:
      'content-length': '364',
      'content-type': 'application/json',

      // custom header
      'test-header': 'test-value',
    });
  });

  it('should pass model, message, and chat history', async () => {
    prepareJsonResponse({});

    await model.doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(await server.getRequestBodyJson()).toStrictEqual({
      model: 'command-r-plus',
      message: 'Hello',
      chat_history: [{ role: 'SYSTEM', message: 'you are a friendly bot!' }],
    });
  });

  it('should pass headers', async () => {
    prepareJsonResponse({});

    const provider = createCohere({
      apiKey: 'test-api-key',
      headers: {
        'Custom-Provider-Header': 'provider-header-value',
      },
    });

    await provider('command-r-plus').doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
      headers: {
        'Custom-Request-Header': 'request-header-value',
      },
    });

    const requestHeaders = await server.getRequestHeaders();

    expect(requestHeaders).toStrictEqual({
      authorization: 'Bearer test-api-key',
      'content-type': 'application/json',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
  });

  it('should pass response format', async () => {
    prepareJsonResponse({});

    await model.doGenerate({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
      responseFormat: {
        type: 'json',
        schema: {
          type: 'object',
          properties: {
            text: { type: 'string' },
          },
          required: ['text'],
        },
      },
    });

    expect(await server.getRequestBodyJson()).toStrictEqual({
      model: 'command-r-plus',
      message: 'Hello',
      chat_history: [
        {
          role: 'SYSTEM',
          message: 'you are a friendly bot!',
        },
      ],
      response_format: {
        type: 'json_object',
        schema: {
          type: 'object',
          properties: {
            text: { type: 'string' },
          },
          required: ['text'],
        },
      },
    });
  });
});

describe('doStream', () => {
  const server = new StreamingTestServer('https://api.cohere.com/v1/chat');

  server.setupTestEnvironment();

  function prepareStreamResponse({
    content,
    usage = {
      input_tokens: 17,
      output_tokens: 244,
    },
    finish_reason = 'COMPLETE',
  }: {
    content: string[];
    usage?: {
      input_tokens: number;
      output_tokens: number;
    };
    finish_reason?: string;
  }) {
    server.responseChunks = [
      `{"is_finished":false,"event_type":"stream-start","generation_id":"586ac33f-9c64-452c-8f8d-e5890e73b6fb"}\n`,
      ...content.map(
        text =>
          `{"is_finished":false,"event_type":"text-generation","text":"${text}"}\n`,
      ),
      `{"is_finished":true,"event_type":"stream-end","response":` +
        `{"response_id":"ac6d5f86-f5a7-4db9-bacf-f01b98697a5b",` +
        `"text":"${content.join('')}",` +
        `"generation_id":"586ac33f-9c64-452c-8f8d-e5890e73b6fb",` +
        `"chat_history":[{"role":"USER","message":"Invent a new holiday and describe its traditions."},` +
        `{"role":"CHATBOT","message":"${content.join('')}"}],` +
        `"finish_reason":"${finish_reason}","meta":{"api_version":{"version":"1"},` +
        `"billed_units":{"input_tokens":9,"output_tokens":20},` +
        `"tokens":${JSON.stringify(
          usage,
        )}}},"finish_reason":"${finish_reason}"}\n`,
    ];
  }

  it('should stream text deltas', async () => {
    prepareStreamResponse({
      content: ['Hello', ', ', 'World!'],
      finish_reason: 'COMPLETE',
      usage: {
        input_tokens: 34,
        output_tokens: 12,
      },
    });

    const { stream } = await model.doStream({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    // note: space moved to last chunk bc of trimming
    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      { type: 'text-delta', textDelta: 'Hello' },
      { type: 'text-delta', textDelta: ', ' },
      { type: 'text-delta', textDelta: 'World!' },
      {
        type: 'finish',
        finishReason: 'stop',
        usage: { promptTokens: 34, completionTokens: 12 },
      },
    ]);
  });

  it('should stream tool deltas', async () => {
    server.responseChunks = [
      `{"event_type":"stream-start"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":"I"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" will"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" use"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" the"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" get"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":"Stock"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":"Price"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" tool"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" to"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" find"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" the"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" price"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" of"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" AAPL"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":" stock"}\n\n`,
      `{"event_type":"tool-calls-chunk","text":"."}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"name":"getStockPrice"}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":"{\\n    \\""}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":"ticker"}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":"_"}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":"symbol"}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":"\\":"}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":" \\""}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":"AAPL"}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":"\\""}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":"\\n"}}\n\n`,
      `{"event_type":"tool-calls-chunk","tool_call_delta":{"index":0,"parameters":"}"}}\n\n`,
      `{"event_type":"tool-calls-generation","tool_calls":[{"name":"getStockPrice","parameters":{"ticker_symbol":"AAPL"}}]}\n\n`,
      `{"event_type":"stream-end","finish_reason":"COMPLETE","response":{"meta":{"tokens":{"input_tokens":893,"output_tokens":62}}}}\n\n`,
    ];

    const { stream } = await model.doStream({
      inputFormat: 'prompt',
      prompt: TEST_PROMPT,
      mode: {
        type: 'regular',
        tools: [
          {
            type: 'function',
            name: 'test-tool',
            parameters: {
              type: 'object',
              properties: { value: { type: 'string' } },
              required: ['value'],
              additionalProperties: false,
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
        ],
      },
    });

    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        type: 'tool-call',
        toolCallId: expect.any(String),
        toolCallType: 'function',
        toolName: 'getStockPrice',
        args: '{"ticker_symbol":"AAPL"}',
      },
      {
        finishReason: 'stop',
        type: 'finish',
        usage: {
          completionTokens: 62,
          promptTokens: 893,
        },
      },
    ]);
  });

  it('should handle unparsable stream parts', async () => {
    server.responseChunks = [`{unparsable}\n`];

    const { stream } = await model.doStream({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    const elements = await convertReadableStreamToArray(stream);

    expect(elements.length).toBe(2);
    expect(elements[0].type).toBe('error');
    expect(elements[1]).toStrictEqual({
      finishReason: 'error',
      type: 'finish',
      usage: {
        completionTokens: NaN,
        promptTokens: NaN,
      },
    });
  });

  it('should expose the raw response headers', async () => {
    prepareStreamResponse({ content: [] });

    server.responseHeaders = {
      'test-header': 'test-value',
    };

    const { rawResponse } = await model.doStream({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(rawResponse?.headers).toStrictEqual({
      // default headers:
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',

      // custom header
      'test-header': 'test-value',
    });
  });

  it('should pass the messages and the model', async () => {
    prepareStreamResponse({ content: [] });

    await model.doStream({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
    });

    expect(await server.getRequestBodyJson()).toStrictEqual({
      stream: true,
      model: 'command-r-plus',
      message: 'Hello',
      chat_history: [
        {
          role: 'SYSTEM',
          message: 'you are a friendly bot!',
        },
      ],
    });
  });

  it('should pass headers', async () => {
    prepareStreamResponse({ content: [] });

    const provider = createCohere({
      apiKey: 'test-api-key',
      headers: {
        'Custom-Provider-Header': 'provider-header-value',
      },
    });

    await provider('command-r-plus').doStream({
      inputFormat: 'prompt',
      mode: { type: 'regular' },
      prompt: TEST_PROMPT,
      headers: {
        'Custom-Request-Header': 'request-header-value',
      },
    });

    const requestHeaders = await server.getRequestHeaders();

    expect(requestHeaders).toStrictEqual({
      authorization: 'Bearer test-api-key',
      'content-type': 'application/json',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
  });
});

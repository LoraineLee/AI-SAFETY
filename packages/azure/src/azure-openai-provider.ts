import {
  OpenAIChatLanguageModel,
  OpenAIChatSettings,
  OpenAICompletionLanguageModel,
  OpenAICompletionSettings,
  OpenAIEmbeddingModel,
  OpenAIEmbeddingSettings,
} from '@ai-sdk/openai/internal';
import {
  EmbeddingModelV1,
  LanguageModelV1,
  ProviderV1,
} from '@ai-sdk/provider';
import { FetchFunction, loadApiKey, loadSetting } from '@ai-sdk/provider-utils';

export interface AzureOpenAIProvider extends ProviderV1 {
  (deploymentId: string, settings?: OpenAIChatSettings): LanguageModelV1;

  /**
Creates an Azure OpenAI chat model for text generation.
   */
  languageModel(
    deploymentId: string,
    settings?: OpenAIChatSettings,
  ): LanguageModelV1;

  /**
Creates an Azure OpenAI chat model for text generation.
   */
  chat(deploymentId: string, settings?: OpenAIChatSettings): LanguageModelV1;

  /**
Creates an Azure OpenAI completion model for text generation.
   */
  completion(
    deploymentId: string,
    settings?: OpenAICompletionSettings,
  ): LanguageModelV1;

  /**
@deprecated Use `textEmbeddingModel` instead.
   */
  embedding(
    deploymentId: string,
    settings?: OpenAIEmbeddingSettings,
  ): EmbeddingModelV1<string>;

  /**
@deprecated Use `textEmbeddingModel` instead.
   */
  textEmbedding(
    deploymentId: string,
    settings?: OpenAIEmbeddingSettings,
  ): EmbeddingModelV1<string>;

  /**
Creates an Azure OpenAI model for text embeddings.
   */
  textEmbeddingModel(
    deploymentId: string,
    settings?: OpenAIEmbeddingSettings,
  ): EmbeddingModelV1<string>;
}

export interface AzureOpenAIProviderSettings {
  /**
Name of the Azure OpenAI resource. Either this or `baseURL` can be used.

The resource name is used in the assembled URL: `https://{resourceName}.openai.azure.com/openai/deployments/{modelId}{path}`.
     */
  resourceName?: string;

  /**
Use a different URL prefix for API calls, e.g. to use proxy servers. Either this or `resourceName` can be used.
When a baseURL is provided, the resourceName is ignored.

With a baseURL, the resolved URL is `{baseURL}/{modelId}{path}`.
   */
  baseURL?: string;

  /**
API key for authenticating requests.
     */
  apiKey?: string;

  /**
Custom headers to include in the requests.
     */
  headers?: Record<string, string>;

  /**
Custom fetch implementation. You can use it as a middleware to intercept requests,
or to provide a custom fetch implementation for e.g. testing.
    */
  fetch?: FetchFunction;

  /**
   * Function for retrieving a bearer token via @azure/identity. This is used to authenticate requests to Azure OpenAI through Managed Identity.
   */
  identityTokenProvider?: () => Promise<string>;
}

/**
Create an Azure OpenAI provider instance.
 */
export function createAzure(
  options: AzureOpenAIProviderSettings = {},
): AzureOpenAIProvider {
  if (typeof options.identityTokenProvider === 'function' && options.apiKey) {
    throw new Error(
      'identityTokenProvider and apiKey are mutually exclusive, please use only one of them to authenticate requests to Azure OpenAI',
    );
  }

  const getHeaders = () => {
    const baseHeaders = { ...options.headers };

    if (typeof options.identityTokenProvider !== 'function') {
      baseHeaders['api-key'] = loadApiKey({
        apiKey: options.apiKey,
        environmentVariableName: 'AZURE_API_KEY',
        description: 'Azure OpenAI',
      });
    }

    return baseHeaders;
  };

  const getResourceName = () =>
    loadSetting({
      settingValue: options.resourceName,
      settingName: 'resourceName',
      environmentVariableName: 'AZURE_RESOURCE_NAME',
      description: 'Azure OpenAI resource name',
    });

  const url = ({ path, modelId }: { path: string; modelId: string }) =>
    options.baseURL
      ? `${options.baseURL}/${modelId}${path}?api-version=2024-08-01-preview`
      : `https://${getResourceName()}.openai.azure.com/openai/deployments/${modelId}${path}?api-version=2024-08-01-preview`;

  // Fetch wrapper to inject Authorization header if using managed identity
  const fetchTokenWrapper: FetchFunction = async (input, init) => {
    const baseFetch = options.fetch || globalThis.fetch;

    if (typeof options.identityTokenProvider !== 'function') {
      return baseFetch(input as RequestInfo, init);
    }

    try {
      const token = await options.identityTokenProvider();
      if (!token || typeof token !== 'string') {
        throw new Error(
          `Invalid token received from identityTokenProvider: ${token}`,
        );
      }

      const modifiedInit: RequestInit = {
        ...init,
        headers: new Headers(init?.headers),
      };
      (modifiedInit.headers as Headers).set('Authorization', `Bearer ${token}`);

      return baseFetch(input as RequestInfo, modifiedInit);
    } catch (error) {
      throw new Error(`Error getting Azure identity token: ${error}`);
    }
  };

  const createChatModel = (
    deploymentName: string,
    settings: OpenAIChatSettings = {},
  ) =>
    new OpenAIChatLanguageModel(deploymentName, settings, {
      provider: 'azure-openai.chat',
      url,
      headers: getHeaders,
      compatibility: 'strict',
      fetch: fetchTokenWrapper,
    });

  const createCompletionModel = (
    modelId: string,
    settings: OpenAICompletionSettings = {},
  ) =>
    new OpenAICompletionLanguageModel(modelId, settings, {
      provider: 'azure-openai.completion',
      url,
      compatibility: 'strict',
      headers: getHeaders,
      fetch: fetchTokenWrapper,
    });

  const createEmbeddingModel = (
    modelId: string,
    settings: OpenAIEmbeddingSettings = {},
  ) =>
    new OpenAIEmbeddingModel(modelId, settings, {
      provider: 'azure-openai.embeddings',
      headers: getHeaders,
      url,
      fetch: fetchTokenWrapper,
    });

  const provider = function (
    deploymentId: string,
    settings?: OpenAIChatSettings | OpenAICompletionSettings,
  ) {
    if (new.target) {
      throw new Error(
        'The Azure OpenAI model function cannot be called with the new keyword.',
      );
    }

    return createChatModel(deploymentId, settings as OpenAIChatSettings);
  };

  provider.languageModel = createChatModel;
  provider.chat = createChatModel;
  provider.completion = createCompletionModel;
  provider.embedding = createEmbeddingModel;
  provider.textEmbedding = createEmbeddingModel;
  provider.textEmbeddingModel = createEmbeddingModel;

  return provider as AzureOpenAIProvider;
}

/**
Default Azure OpenAI provider instance.
 */
export const azure = createAzure();

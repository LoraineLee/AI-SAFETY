import {
  EmbeddingModelV1,
  TooManyEmbeddingValuesForCallError,
} from '@ai-sdk/provider';
import {
  combineHeaders,
  createJsonResponseHandler,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import { z } from 'zod';
import { googleFailedResponseHandler } from './google-error';
import {
  GoogleVertexEmbeddingModelId,
  GoogleVertexEmbeddingSettings,
} from './google-vertex-embedding-settings';

type GoogleVertexEmbeddingConfig = {
  provider: string;
  region: string;
  project: string;
  generateAuthToken: () => Promise<string | null | undefined>;
};

export class GoogleVertexEmbeddingModel implements EmbeddingModelV1<string> {
  readonly specificationVersion = 'v1';
  readonly modelId: GoogleVertexEmbeddingModelId;

  private readonly config: GoogleVertexEmbeddingConfig;
  private readonly settings: GoogleVertexEmbeddingSettings;

  get provider(): string {
    return this.config.provider;
  }

  get maxEmbeddingsPerCall(): number {
    return 2048;
  }

  get supportsParallelCalls(): boolean {
    return true;
  }

  constructor(
    modelId: GoogleVertexEmbeddingModelId,
    settings: GoogleVertexEmbeddingSettings,
    config: GoogleVertexEmbeddingConfig,
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  async doEmbed({
    values,
    headers,
    abortSignal,
  }: Parameters<EmbeddingModelV1<string>['doEmbed']>[0]): Promise<
    Awaited<ReturnType<EmbeddingModelV1<string>['doEmbed']>>
  > {
    if (values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values,
      });
    }

    const { responseHeaders, value: response } = await postJsonToApi({
      url:
        `https://${this.config.region}-aiplatform.googleapis.com/v1/` +
        `projects/${this.config.project}/locations/${this.config.region}/` +
        `publishers/google/models/${this.modelId}:predict`,
      headers: combineHeaders(
        { Authorization: `Bearer ${await this.config.generateAuthToken()}` },
        headers,
      ),
      body: {
        instances: values.map(value => ({ content: value })),
        parameters: {
          outputDimensionality: this.settings.outputDimensionality,
        },
      },
      failedResponseHandler: googleFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        googleVertexTextEmbeddingResponseSchema,
      ),
      abortSignal,
    });

    return {
      embeddings: response.predictions.map(
        prediction => prediction.embeddings.values,
      ),
      usage: {
        tokens: response.predictions.reduce(
          (tokenCount, prediction) =>
            tokenCount + prediction.embeddings.statistics.token_count,
          0,
        ),
      },
      rawResponse: { headers: responseHeaders },
    };
  }
}

// minimal version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const googleVertexTextEmbeddingResponseSchema = z.object({
  predictions: z.array(
    z.object({
      embeddings: z.object({
        values: z.array(z.number()),
        statistics: z.object({
          token_count: z.number(),
        }),
      }),
    }),
  ),
});

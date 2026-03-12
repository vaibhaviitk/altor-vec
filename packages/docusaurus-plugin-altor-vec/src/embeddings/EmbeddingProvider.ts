/**
 * Copyright (c) altor-lab
 * SPDX-License-Identifier: MIT
 */

import { IEmbeddingProvider, Logger } from '../types';
import { PluginError } from '../utils/PluginError';
import { ErrorCode } from '../types';

/**
 * Transformers.js embedding provider (default, runs locally).
 */
export class TransformersEmbeddingProvider implements IEmbeddingProvider {
  private pipeline: any;

  constructor(
    private readonly modelName: string,
    private readonly dimensions: number,
    private readonly cachePath: string,
    private readonly logger: Logger
  ) {}

  async initialize(): Promise<void> {
    try {
      this.logger.info(`Initializing embedding model: ${this.modelName}`);
      const { pipeline, env } = await import('@huggingface/transformers');

      // Set cache directory
      env.cacheDir = this.cachePath;

      this.pipeline = await pipeline('feature-extraction', this.modelName, {
        dtype: 'fp32',
      });

      this.logger.info('Embedding model initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize embedding model', error as Error);
      throw new PluginError(
        `Failed to initialize model: ${this.modelName}`,
        ErrorCode.MODEL_INIT_FAILED,
        'Check your internet connection and model name'
      );
    }
  }

  async generateEmbedding(text: string): Promise<Float32Array> {
    try {
      if (!text || text.trim().length === 0) {
        this.logger.warn('Empty text provided for embedding');
        return new Float32Array(this.dimensions).fill(0);
      }

      const output = await this.pipeline(text, {
        pooling: 'mean',
        normalize: true,
      });
      return new Float32Array(output.data);
    } catch (error) {
      this.logger.error('Failed to generate embedding', error as Error, {
        textLength: text.length,
      });
      throw new PluginError(
        'Failed to generate embedding',
        ErrorCode.EMBEDDING_FAILED,
        'Try reducing document length or check model compatibility'
      );
    }
  }

  async generateBatch(texts: string[]): Promise<Float32Array[]> {
    this.logger.info(`Generating embeddings for ${texts.length} documents`);
    const embeddings: Float32Array[] = [];

    for (let i = 0; i < texts.length; i++) {
      embeddings.push(await this.generateEmbedding(texts[i]));

      if ((i + 1) % 10 === 0) {
        this.logger.debug(`Progress: ${i + 1}/${texts.length} embeddings generated`);
      }
    }

    return embeddings;
  }

  getDimensions(): number {
    return this.dimensions;
  }
}

/**
 * OpenAI embedding provider (alternative, requires API key).
 */
export class OpenAIEmbeddingProvider implements IEmbeddingProvider {
  private readonly baseUrl = 'https://api.openai.com/v1/embeddings';

  constructor(
    private readonly apiKey: string,
    private readonly model: string,
    private readonly dimensions: number,
    private readonly logger: Logger
  ) {}

  async initialize(): Promise<void> {
    this.logger.info(`Initializing OpenAI provider with model: ${this.model}`);
    // Test API key
    try {
      await this.generateEmbedding('test');
      this.logger.info('OpenAI API key validated');
    } catch (error) {
      throw new PluginError(
        'Invalid OpenAI API key',
        ErrorCode.INVALID_API_KEY,
        'Check your API key and ensure it has embedding permissions'
      );
    }
  }

  async generateEmbedding(text: string): Promise<Float32Array> {
    try {
      const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model: this.model,
          input: text,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error?.message || 'API request failed');
      }

      const data = await response.json();
      return new Float32Array(data.data[0].embedding);
    } catch (error) {
      this.logger.error('OpenAI API request failed', error as Error);
      throw new PluginError(
        'Failed to generate embedding via OpenAI',
        ErrorCode.API_REQUEST_FAILED,
        'Check your API key and rate limits'
      );
    }
  }

  async generateBatch(texts: string[]): Promise<Float32Array[]> {
    this.logger.info(`Generating embeddings for ${texts.length} documents via OpenAI`);
    const embeddings: Float32Array[] = [];

    // OpenAI allows batch requests, but we'll do sequential for rate limiting
    for (let i = 0; i < texts.length; i++) {
      embeddings.push(await this.generateEmbedding(texts[i]));

      if ((i + 1) % 10 === 0) {
        this.logger.debug(`Progress: ${i + 1}/${texts.length} embeddings generated`);
      }

      // Rate limiting: wait 100ms between requests
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    return embeddings;
  }

  getDimensions(): number {
    return this.dimensions;
  }
}

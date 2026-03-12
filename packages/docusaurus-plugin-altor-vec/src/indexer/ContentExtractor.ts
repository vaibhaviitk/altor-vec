/**
 * Copyright (c) altor-lab
 * SPDX-License-Identifier: MIT
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import matter from 'gray-matter';
import { glob } from 'glob';
import { Document, Logger } from '../types';
import { PluginError } from '../utils/PluginError';
import { ErrorCode } from '../types';

export interface ExtractionOptions {
  maxDocumentLength: number;
  chunkSize: number;
  chunkOverlap: number;
  includePatterns: string[];
  excludePatterns: string[];
}

export interface IContentExtractor {
  extract(filePath: string): Promise<Document>;
  extractBatch(filePaths: string[]): Promise<Document[]>;
  findFiles(): Promise<string[]>;
}

export class MarkdownContentExtractor implements IContentExtractor {
  constructor(
    private readonly siteDir: string,
    private readonly baseUrl: string,
    private readonly options: ExtractionOptions,
    private readonly logger: Logger
  ) {}

  /**
   * Find all markdown files matching include/exclude patterns.
   */
  async findFiles(): Promise<string[]> {
    const files: string[] = [];

    for (const pattern of this.options.includePatterns) {
      const matches = await glob(pattern, {
        cwd: this.siteDir,
        absolute: true,
        ignore: this.options.excludePatterns,
      });
      files.push(...matches);
    }

    // Remove duplicates
    return [...new Set(files)];
  }

  /**
   * Extract a single document from a markdown file.
   */
  async extract(filePath: string): Promise<Document> {
    try {
      // 1. Validate file path (prevent path traversal)
      const resolvedPath = path.resolve(filePath);
      const resolvedSiteDir = path.resolve(this.siteDir);
      if (!resolvedPath.startsWith(resolvedSiteDir)) {
        throw new PluginError(
          `File path outside site directory: ${filePath}`,
          ErrorCode.INVALID_INPUT,
          'Ensure file paths are within the site directory'
        );
      }

      // 2. Check file size (prevent memory exhaustion)
      const fileStats = await fs.stat(filePath);
      const maxFileSize = 10 * 1024 * 1024; // 10MB
      if (fileStats.size > maxFileSize) {
        this.logger.warn(`Skipping large file (${(fileStats.size / 1024 / 1024).toFixed(2)}MB): ${filePath}`);
        throw new PluginError(
          `File too large: ${filePath}`,
          ErrorCode.INVALID_INPUT,
          'Files must be smaller than 10MB'
        );
      }

      // 3. Read file
      const content = await fs.readFile(filePath, 'utf-8');

      // 4. Parse frontmatter
      const { data: frontmatter, content: markdown } = matter(content);

      // 5. Strip markdown syntax
      const plainText = this.stripMarkdown(markdown);

      // 6. Chunk if needed
      const chunkedText = this.chunkDocument(plainText, this.options.maxDocumentLength);

      // 7. Generate URL from file path
      const url = this.generateUrl(filePath);

      // 8. Extract title
      const title =
        frontmatter.title ||
        this.extractFirstHeading(markdown) ||
        path.basename(filePath, '.md');

      this.logger.debug(`Extracted document: ${title}`, { filePath, url });

      // 9. Return structured document
      return {
        id: this.generateId(filePath),
        title,
        content: chunkedText,
        url,
        metadata: {
          section: frontmatter.section,
          tags: frontmatter.tags || [],
          lastModified: fileStats.mtime,
        },
      };
    } catch (error) {
      this.logger.error(`Failed to extract document: ${filePath}`, error as Error);
      throw new PluginError(
        `Failed to extract ${filePath}`,
        ErrorCode.EXTRACTION_FAILED,
        'Check if the file is valid markdown'
      );
    }
  }

  /**
   * Extract multiple documents in batch.
   */
  async extractBatch(filePaths: string[]): Promise<Document[]> {
    this.logger.info(`Extracting ${filePaths.length} documents`);
    const documents: Document[] = [];

    for (let i = 0; i < filePaths.length; i++) {
      try {
        const doc = await this.extract(filePaths[i]);
        documents.push(doc);

        if ((i + 1) % 10 === 0) {
          this.logger.debug(`Progress: ${i + 1}/${filePaths.length} documents extracted`);
        }
      } catch (error) {
        this.logger.warn(`Skipping file due to error: ${filePaths[i]}`);
        // Continue with other files
      }
    }

    this.logger.info(`Successfully extracted ${documents.length}/${filePaths.length} documents`);
    return documents;
  }

  /**
   * Generate a unique ID for a document based on file path.
   * Note: MD5 is used for non-cryptographic purposes (document IDs only).
   * This is safe as we're not using it for security/authentication.
   */
  private generateId(filePath: string): string {
    return crypto.createHash('md5').update(filePath).digest('hex');
  }

  /**
   * Generate URL from file path.
   */
  private generateUrl(filePath: string): string {
    const relativePath = path.relative(this.siteDir, filePath);
    const urlPath = relativePath
      .replace(/\.mdx?$/, '')
      .replace(/\\/g, '/')
      .replace(/\/index$/, '');
    return path.posix.join(this.baseUrl, urlPath);
  }

  /**
   * Extract first heading from markdown content.
   */
  private extractFirstHeading(markdown: string): string | null {
    const match = markdown.match(/^#\s+(.+)$/m);
    return match ? match[1].trim() : null;
  }

  /**
   * Strip markdown syntax to get plain text.
   */
  private stripMarkdown(content: string): string {
    return (
      content
        // Remove code blocks
        .replace(/```[\s\S]*?```/g, '')
        .replace(/`[^`]+`/g, '')
        // Remove images
        .replace(/!\[([^\]]+)\]\([^)]+\)/g, '$1')
        // Remove links but keep text
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        // Remove HTML tags
        .replace(/<[^>]+>/g, '')
        // Remove bold/italic
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/\*([^*]+)\*/g, '$1')
        .replace(/__([^_]+)__/g, '$1')
        .replace(/_([^_]+)_/g, '$1')
        // Remove heading markers
        .replace(/^#{1,6}\s+/gm, '')
        // Remove horizontal rules
        .replace(/^[-*_]{3,}$/gm, '')
        // Remove blockquotes
        .replace(/^>\s+/gm, '')
        // Normalize whitespace
        .replace(/\s+/g, ' ')
        .trim()
    );
  }

  /**
   * Chunk document if it exceeds max length.
   */
  private chunkDocument(content: string, maxLength: number): string {
    if (content.length <= maxLength) {
      return content;
    }

    // For now, simple truncation
    // TODO: Implement smart chunking by sentences/paragraphs
    this.logger.warn(`Document truncated from ${content.length} to ${maxLength} chars`);
    return content.substring(0, maxLength);
  }
}

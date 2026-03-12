/**
 * Copyright (c) altor-lab
 * SPDX-License-Identifier: MIT
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';

export interface SearchResult {
  id: string;
  title: string;
  url: string;
  preview: string;
  score: number;
}

export interface SearchTiming {
  embedMs: string;
  searchMs: string;
  totalMs: string;
}

export interface SearchBarProps {
  placeholder?: string;
  maxResults?: number;
  debounceMs?: number;
  showTiming?: boolean;
  onResultClick?: (result: SearchResult) => void;
  renderResult?: (result: SearchResult) => React.ReactNode;
  className?: string;
  indexPath?: string;
  embeddingModel?: string;
  embeddingDimensions?: number;
  i18n?: {
    searchPlaceholder: string;
    noResults: string;
    loading: string;
    error: string;
  };
}

function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

export const SearchBar: React.FC<SearchBarProps> = ({
  placeholder = 'Search documentation...',
  maxResults = 5,
  debounceMs = 300,
  showTiming = false,
  onResultClick,
  renderResult,
  className = '',
  indexPath = '__altor-vec__',
  embeddingModel = 'Xenova/all-MiniLM-L6-v2',
  embeddingDimensions = 384,
  i18n = {
    searchPlaceholder: 'Search documentation...',
    noResults: 'No results found',
    loading: 'Loading...',
    error: 'Search error',
  },
}) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [timing, setTiming] = useState<SearchTiming | null>(null);
  const [error, setError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    // Initialize worker
    try {
      // @ts-ignore - Worker URL resolution
      workerRef.current = new Worker(
        new URL('../worker/searchWorker.ts', import.meta.url as any),
        { type: 'module' }
      );

      // Send configuration to worker
      workerRef.current.postMessage({
        type: 'init',
        config: {
          indexPath,
          embeddingModel,
          embeddingDimensions,
        },
      });

      workerRef.current.onmessage = (e) => {
        const { type, results: searchResults, timing: searchTiming, message } = e.data;

        if (type === 'ready') {
          setError(null);
        } else if (type === 'results') {
          setResults(searchResults);
          setTiming(searchTiming);
          setIsLoading(false);
          setError(null);
        } else if (type === 'error') {
          setError(message);
          setIsLoading(false);
        }
      };

      return () => workerRef.current?.terminate();
    } catch (err) {
      setError('Failed to initialize search worker');
      console.error('Worker initialization error:', err);
    }
  }, [indexPath, embeddingModel, embeddingDimensions]);

  const handleSearch = useMemo(
    () =>
      debounce((searchQuery: string) => {
        if (!searchQuery.trim()) {
          setResults([]);
          setTiming(null);
          return;
        }

        setIsLoading(true);
        setError(null);
        workerRef.current?.postMessage({
          type: 'search',
          query: searchQuery,
          topK: maxResults,
        });
      }, debounceMs),
    [maxResults, debounceMs]
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    handleSearch(value);
  };

  const handleResultClickInternal = (result: SearchResult) => {
    if (onResultClick) {
      onResultClick(result);
    } else {
      window.location.href = result.url;
    }
  };

  return (
    <div className={`altor-search-container ${className}`}>
      <div className="altor-search-input-wrapper">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder={placeholder}
          className="altor-search-input"
          aria-label="Search"
        />
        {isLoading && <span className="altor-search-loading">{i18n.loading}</span>}
      </div>

      {error && <div className="altor-search-error">{error}</div>}

      {!isLoading && query && results.length === 0 && !error && (
        <div className="altor-search-no-results">{i18n.noResults}</div>
      )}

      {results.length > 0 && (
        <div className="altor-search-results">
          {results.map((result) => (
            <div
              key={result.id}
              className="altor-search-result"
              onClick={() => handleResultClickInternal(result)}
              role="button"
              tabIndex={0}
              onKeyPress={(e) => {
                if (e.key === 'Enter') handleResultClickInternal(result);
              }}
            >
              {renderResult ? (
                renderResult(result)
              ) : (
                <>
                  <div className="altor-search-result-title">{result.title}</div>
                  <div className="altor-search-result-preview">{result.preview}</div>
                  {showTiming && (
                    <div className="altor-search-result-score">
                      Score: {result.score.toFixed(3)}
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      )}

      {showTiming && timing && (
        <div className="altor-search-timing">
          Embed: {timing.embedMs}ms | Search: {timing.searchMs}ms | Total: {timing.totalMs}ms
        </div>
      )}
    </div>
  );
};

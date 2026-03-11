# altor-vec

[![npm](https://img.shields.io/npm/v/altor-vec?color=blue)](https://www.npmjs.com/package/altor-vec)
[![downloads](https://img.shields.io/npm/dm/altor-vec?color=green)](https://www.npmjs.com/package/altor-vec)
[![GitHub stars](https://img.shields.io/github/stars/altor-lab/altor-vec?style=social)](https://github.com/altor-lab/altor-vec)

**Client-side vector search. 54KB. Sub-millisecond.**

HNSW-powered semantic search that runs entirely in the browser via WebAssembly. No server, no API keys, no per-query billing. Your users' data never leaves their device.

GitHub: **[altor-lab/altor-vec](https://github.com/altor-lab/altor-vec)** | Built by [altor-lab](https://github.com/altor-lab)

---

### Why altor-vec?

| | altor-vec | Algolia | Voy | Orama |
|---|-----------|---------|-----|-------|
| **Client-side** | Yes | No | Yes | Yes |
| **Size (gzip)** | **54KB** | N/A | 75KB | ~2KB* |
| **Algorithm** | **HNSW** | BM25 | k-d tree | Brute-force |
| **p95 latency** | **0.6ms** | ~50ms | ~2ms | ~5ms |
| **Per-query cost** | **$0** | $0.50/1K | $0 | Free tier |

<sub>*Orama 2KB = keyword only; vector search adds significant size.</sub>

---

## Install

```bash
npm install altor-vec
```

## Quick Start

```js
import init, { WasmSearchEngine } from 'altor-vec';

await init();

// Load a pre-built index
const resp = await fetch('/index.bin');
const engine = new WasmSearchEngine(new Uint8Array(await resp.arrayBuffer()));

// Search — returns in <1ms
const results = JSON.parse(engine.search(queryEmbedding, 5));
// => [[nodeId, distance], ...]
```

### Build an index in the browser

```js
const engine = WasmSearchEngine.from_vectors(
  flatVectors,  // Float32Array (vectors concatenated)
  384,           // dimensions
  16,            // M (connections per node)
  200,           // ef_construction
  50             // ef_search
);

const indexBytes = engine.to_bytes(); // save for later
```

### Web Worker (recommended)

```js
// worker.js — all WASM work off the main thread
import init, { WasmSearchEngine } from 'altor-vec';

let engine;
self.onmessage = async (e) => {
  if (e.data.type === 'init') {
    await init();
    const resp = await fetch(e.data.indexUrl);
    engine = new WasmSearchEngine(new Uint8Array(await resp.arrayBuffer()));
    postMessage({ type: 'ready', count: engine.len() });
  }
  if (e.data.type === 'search') {
    postMessage({ type: 'results',
      results: JSON.parse(engine.search(new Float32Array(e.data.query), e.data.topK))
    });
  }
};
```

## API

| Method | Description |
|--------|-------------|
| `new WasmSearchEngine(bytes: Uint8Array)` | Load a serialized index |
| `.from_vectors(flat, dims, m, ef_construction, ef_search)` | Build index from vectors |
| `.search(query: Float32Array, topK): string` | Search → JSON `[[id, dist], ...]` |
| `.add_vectors(flat: Float32Array, dims)` | Add vectors to existing index |
| `.to_bytes(): Uint8Array` | Serialize index |
| `.len(): number` | Vector count |
| `.free()` | Free WASM memory |

## Benchmarks

10K vectors, 384 dimensions (all-MiniLM-L6-v2):

| Environment | p95 Latency | Index Load |
|-------------|-------------|------------|
| Chrome | **0.60ms** | 19ms |
| Node.js | **0.50ms** | 38ms |
| Native Rust | **0.26ms** | — |

| `.wasm` raw | `.wasm` gzip | Index (10K/384d) |
|-------------|-------------|------------------|
| 117KB | **54KB** | 17MB |

## Embedding Models

Works with any embedding model:

| Model | Dims | Runs in |
|-------|------|---------|
| all-MiniLM-L6-v2 | 384 | Browser ([Transformers.js](https://huggingface.co/docs/transformers.js)) |
| text-embedding-3-small | 1536 | OpenAI API |
| embed-english-v3 | 1024 | Cohere API |

**Fully client-side** with Transformers.js:

```js
import { pipeline } from '@huggingface/transformers';

const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
const output = await embed('your query', { pooling: 'mean', normalize: true });
const results = JSON.parse(engine.search(new Float32Array(output.data), 5));
```

## License

MIT — [altor-lab](https://github.com/altor-lab)

---

**Need managed semantic search?** Embedding pipeline + index building + CDN delivery → [anshul@altorlab.dev](mailto:anshul@altorlab.dev)

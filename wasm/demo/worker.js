// Web Worker: handles WASM search + Transformers.js embedding off the main thread.
// Main thread sends { type, ... } messages; worker responds with { type, ... }.

import init, { WasmSearchEngine } from '../pkg/altor_vec_wasm.js';

let engine = null;
let metadata = null;
let embedPipeline = null;

async function handleInit() {
  postMessage({ type: 'status', msg: 'Initializing WASM...', pct: 10 });
  await init();

  postMessage({ type: 'status', msg: 'Fetching index...', pct: 30 });
  const [indexBuf, metaResp] = await Promise.all([
    fetch('../../data/index.bin').then(r => {
      if (!r.ok) throw new Error('index.bin not found — run the data pipeline first');
      return r.arrayBuffer();
    }),
    fetch('../../data/metadata.json').then(r => {
      if (!r.ok) throw new Error('metadata.json not found — run the data pipeline first');
      return r.json();
    }),
  ]);

  postMessage({ type: 'status', msg: 'Loading index into WASM...', pct: 50 });
  const t0 = performance.now();
  engine = new WasmSearchEngine(new Uint8Array(indexBuf));
  const loadMs = (performance.now() - t0).toFixed(0);
  metadata = metaResp;

  postMessage({ type: 'status', msg: `Index loaded: ${engine.len()} vectors in ${loadMs}ms. Loading embedding model...`, pct: 60 });

  const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3');
  postMessage({ type: 'status', msg: 'Downloading embedding model (first load ~33MB, cached after)...', pct: 75 });

  embedPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    dtype: 'fp32',
  });

  postMessage({ type: 'ready' });
}

async function handleSearch(query, requestId) {
  if (!engine || !embedPipeline || !query.trim()) {
    postMessage({ type: 'results', requestId, results: [], timing: null });
    return;
  }

  const totalStart = performance.now();

  const embedStart = performance.now();
  const output = await embedPipeline(query, { pooling: 'mean', normalize: true });
  const embedding = Array.from(output.data);
  const embedMs = performance.now() - embedStart;

  const searchStart = performance.now();
  const rawResults = JSON.parse(engine.search(new Float32Array(embedding), 5));
  const searchMs = performance.now() - searchStart;

  const totalMs = performance.now() - totalStart;

  const results = rawResults.map(([nodeId, distance]) => ({
    id: nodeId,
    score: 1 - distance,
    text: metadata[nodeId]?.text || `[passage ${nodeId}]`,
  }));

  postMessage({
    type: 'results',
    requestId,
    results,
    timing: {
      embedMs: embedMs.toFixed(0),
      searchMs: searchMs.toFixed(1),
      totalMs: totalMs.toFixed(0),
    },
  });
}

self.onmessage = async (e) => {
  const { type, query, requestId } = e.data;
  try {
    if (type === 'init') await handleInit();
    else if (type === 'search') await handleSearch(query, requestId);
  } catch (err) {
    postMessage({ type: 'error', msg: err.message });
  }
};

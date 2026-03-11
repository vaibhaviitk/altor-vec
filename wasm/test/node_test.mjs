import { readFile } from "fs/promises";
import { initSync, WasmSearchEngine } from "../pkg/altor_vec_wasm.js";

// Load and init WASM synchronously
const wasmBytes = await readFile(
  new URL("../pkg/altor_vec_wasm_bg.wasm", import.meta.url),
);
initSync({ module: wasmBytes });

// Load pre-built index
const indexBytes = await readFile(
  new URL("../../test_index.bin", import.meta.url),
);
const t0 = performance.now();
const engine = new WasmSearchEngine(indexBytes);
const loadTime = performance.now() - t0;
console.log(`Index loaded: ${engine.len()} vectors in ${loadTime.toFixed(1)}ms`);

// Generate random query
function randomQuery(dims) {
  const q = new Float32Array(dims);
  for (let i = 0; i < dims; i++) q[i] = Math.random() * 2 - 1;
  // normalize
  let norm = 0;
  for (let i = 0; i < dims; i++) norm += q[i] * q[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < dims; i++) q[i] /= norm;
  return q;
}

// Run queries and measure latency
const dims = 384;
const numQueries = 100;
const topK = 10;
const latencies = [];

for (let i = 0; i < numQueries; i++) {
  const query = randomQuery(dims);
  const start = performance.now();
  const results = engine.search(query, topK);
  const elapsed = performance.now() - start;
  latencies.push(elapsed);
}

latencies.sort((a, b) => a - b);
const avg = latencies.reduce((a, b) => a + b) / latencies.length;
const p50 = latencies[Math.floor(latencies.length * 0.5)];
const p95 = latencies[Math.floor(latencies.length * 0.95)];

console.log(`\nQuery latency (${numQueries} queries, top-${topK}):`);
console.log(`  avg: ${avg.toFixed(2)}ms`);
console.log(`  p50: ${p50.toFixed(2)}ms`);
console.log(`  p95: ${p95.toFixed(2)}ms`);

// Verify results are valid
const query = randomQuery(dims);
const results = JSON.parse(engine.search(query, topK));
console.log(`\nSample results (first 3 of ${results.length}):`);
for (const [id, dist] of results.slice(0, 3)) {
  console.log(`  node=${id}, distance=${dist.toFixed(6)}`);
}

engine.free();

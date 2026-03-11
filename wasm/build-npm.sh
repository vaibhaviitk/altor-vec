#!/bin/bash
# Build WASM and prepare npm package as "altor-vec"
set -e

cd "$(dirname "$0")"

echo "Building WASM..."
wasm-pack build --target web --release

echo "Patching package.json..."
node -e "
const pkg = require('./pkg/package.json');
pkg.name = 'altor-vec';
pkg.description = 'Client-side vector search powered by HNSW. 54KB gzipped WASM. Sub-millisecond latency. By altor-lab.';
pkg.keywords = ['vector', 'search', 'hnsw', 'wasm', 'semantic-search', 'embeddings', 'nearest-neighbor', 'client-side', 'altor-lab'];
pkg.homepage = 'https://github.com/altor-lab/altor-vec';
pkg.repository = { type: 'git', url: 'https://github.com/altor-lab/altor-vec' };
pkg.license = 'MIT';
pkg.author = 'altor-lab';
require('fs').writeFileSync('./pkg/package.json', JSON.stringify(pkg, null, 2) + '\n');
"

# Copy README into pkg
cp ../npm-README.md ./pkg/README.md

echo "Done! Package ready in wasm/pkg/"
echo "  npm publish ./wasm/pkg"

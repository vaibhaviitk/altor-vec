use altor_vec::HnswIndex;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmSearchEngine {
    index: HnswIndex,
    rng: SmallRng,
}

#[wasm_bindgen]
impl WasmSearchEngine {
    /// Load a pre-built index from serialized bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> WasmSearchEngine {
        WasmSearchEngine {
            index: HnswIndex::from_bytes(bytes),
            rng: SmallRng::from_entropy(),
        }
    }

    /// Build a new index from a flat f32 array of vectors.
    /// `flat` is a contiguous array of vectors, each of length `dims`.
    pub fn from_vectors(
        flat: &[f32],
        dims: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> WasmSearchEngine {
        let mut index = HnswIndex::new(dims, m, ef_construction, ef_search);
        let mut rng = SmallRng::from_entropy();
        for chunk in flat.chunks_exact(dims) {
            index.insert(chunk.to_vec(), &mut rng);
        }
        WasmSearchEngine { index, rng }
    }

    /// Add vectors to an existing index.
    /// `flat` is a contiguous array of vectors, each of length `dims`.
    pub fn add_vectors(&mut self, flat: &[f32], dims: usize) {
        for chunk in flat.chunks_exact(dims) {
            self.index.insert(chunk.to_vec(), &mut self.rng);
        }
    }

    /// Search for the `top_k` nearest neighbors of a query vector.
    /// Returns JSON: `[[node_id, distance], ...]`
    pub fn search(&self, query: &[f32], top_k: usize) -> String {
        let results = self.index.search(query, top_k);
        serde_json::to_string(&results).unwrap()
    }

    /// Export the index as bytes for later loading.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.index.to_bytes()
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.index.len()
    }
}

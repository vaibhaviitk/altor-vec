pub mod graph;
pub mod search;
pub mod construction;

use rand::Rng;

use crate::distance::normalize;
use self::graph::Graph;
use self::construction::insert;
use self::search::knn_search;

/// HNSW index with configurable parameters.
#[cfg_attr(feature = "serialization", derive(serde::Serialize, serde::Deserialize))]
pub struct HnswIndex {
    pub graph: Graph,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub dims: usize,
}

/// A search result: (node_id, distance).
pub type SearchResult = (usize, f32);

impl HnswIndex {
    /// Create a new HNSW index.
    /// - `dims`: vector dimensionality
    /// - `m`: max connections per node per layer (default: 16)
    /// - `ef_construction`: beam width during construction (default: 200)
    /// - `ef_search`: beam width during search (default: 50)
    pub fn new(dims: usize, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        HnswIndex {
            graph: Graph::new(m),
            ef_construction,
            ef_search,
            dims,
        }
    }

    /// Insert a vector into the index. The vector will be L2-normalized.
    /// Returns the assigned node ID.
    pub fn insert(&mut self, mut vector: Vec<f32>, rng: &mut impl Rng) -> usize {
        assert_eq!(vector.len(), self.dims, "Vector dimension mismatch");
        normalize(&mut vector);
        insert(&mut self.graph, vector, self.ef_construction, rng)
    }

    /// Search for the k nearest neighbors of a query vector.
    /// The query will be L2-normalized before search.
    /// Returns Vec<(node_id, distance)> sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dims, "Query dimension mismatch");
        let mut q = query.to_vec();
        normalize(&mut q);
        knn_search(&self.graph, &q, k, self.ef_search)
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.graph.nodes.is_empty()
    }

    /// Serialize the index to bytes.
    #[cfg(feature = "serialization")]
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("serialization failed")
    }

    /// Deserialize an index from bytes.
    #[cfg(feature = "serialization")]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        bincode::deserialize(bytes).expect("deserialization failed")
    }
}

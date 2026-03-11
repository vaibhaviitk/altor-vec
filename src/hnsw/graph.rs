//! Layered graph structure for HNSW.
//!
//! Each node stores its vector and a list of neighbor lists (one per layer it exists in).
//! Neighbors are stored as (node_id, distance) pairs sorted by distance.

/// A single node in the HNSW graph.
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Node {
    /// The vector data (L2-normalized at insert time).
    pub vector: Vec<f32>,
    /// Neighbors at each layer this node participates in.
    /// `neighbors[0]` = layer 0, `neighbors[1]` = layer 1, etc.
    pub neighbors: Vec<Vec<usize>>,
    /// The maximum layer this node exists on.
    pub max_layer: usize,
}

/// The HNSW layered graph.
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Graph {
    /// All nodes in insertion order. Node ID = index.
    pub nodes: Vec<Node>,
    /// The current entry point node ID.
    pub entry_point: Option<usize>,
    /// The current maximum layer in the graph.
    pub max_layer: usize,
    /// Max connections per node per layer (M).
    pub m: usize,
    /// Max connections for layer 0 (typically 2*M).
    pub m_max0: usize,
}

impl Graph {
    pub fn new(m: usize) -> Self {
        Graph {
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            m_max0: m * 2,
        }
    }

    /// Add a node with the given vector and layer assignment.
    /// Returns the new node's ID.
    pub fn add_node(&mut self, vector: Vec<f32>, max_layer: usize) -> usize {
        let id = self.nodes.len();
        let neighbors = (0..=max_layer).map(|_| Vec::new()).collect();
        self.nodes.push(Node {
            vector,
            neighbors,
            max_layer,
        });
        id
    }

    /// Get the max number of connections for a given layer.
    pub fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max0
        } else {
            self.m
        }
    }

    /// Set neighbors for a node at a specific layer.
    pub fn set_neighbors(&mut self, node_id: usize, layer: usize, neighbors: Vec<usize>) {
        self.nodes[node_id].neighbors[layer] = neighbors;
    }

    /// Get neighbors for a node at a specific layer.
    pub fn get_neighbors(&self, node_id: usize, layer: usize) -> &[usize] {
        if layer < self.nodes[node_id].neighbors.len() {
            &self.nodes[node_id].neighbors[layer]
        } else {
            &[]
        }
    }
}

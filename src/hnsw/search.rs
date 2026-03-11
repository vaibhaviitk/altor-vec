use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::distance::dot_product;
use super::graph::Graph;

#[derive(Clone, Copy)]
struct Candidate {
    id: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

// Max-heap: largest distance on top.
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Min-heap candidate: smallest distance on top.
#[derive(Clone, Copy)]
struct NearCandidate {
    id: usize,
    distance: f32,
}

impl PartialEq for NearCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for NearCandidate {}

impl PartialOrd for NearCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NearCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

#[inline]
fn dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b)
}

/// Search a single layer for the ef closest neighbors to the query.
/// Returns a list of (node_id, distance) sorted by distance ascending.
pub fn search_layer(
    graph: &Graph,
    query: &[f32],
    entry_points: &[usize],
    ef: usize,
    layer: usize,
) -> Vec<(usize, f32)> {
    search_layer_counted(graph, query, entry_points, ef, layer).0
}

/// Like search_layer but also returns the number of distance computations.
pub fn search_layer_counted(
    graph: &Graph,
    query: &[f32],
    entry_points: &[usize],
    ef: usize,
    layer: usize,
) -> (Vec<(usize, f32)>, usize) {
    let num_nodes = graph.nodes.len();
    // Use a flat bitvec instead of HashSet — much faster for sequential IDs
    let mut visited = vec![false; num_nodes];
    let mut dist_computations = 0usize;

    let mut result_heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);
    let mut candidates: BinaryHeap<NearCandidate> = BinaryHeap::new();

    for &ep in entry_points {
        let d = dist(query, &graph.nodes[ep].vector);
        dist_computations += 1;
        visited[ep] = true;
        candidates.push(NearCandidate { id: ep, distance: d });
        result_heap.push(Candidate { id: ep, distance: d });
    }

    while let Some(nearest) = candidates.pop() {
        let farthest_dist = result_heap.peek().map(|c| c.distance).unwrap_or(f32::MAX);
        if nearest.distance > farthest_dist {
            break;
        }

        for &neighbor_id in graph.get_neighbors(nearest.id, layer) {
            if visited[neighbor_id] {
                continue;
            }
            visited[neighbor_id] = true;

            let d = dist(query, &graph.nodes[neighbor_id].vector);
            dist_computations += 1;
            let farthest_dist = result_heap.peek().map(|c| c.distance).unwrap_or(f32::MAX);

            if result_heap.len() < ef || d < farthest_dist {
                candidates.push(NearCandidate {
                    id: neighbor_id,
                    distance: d,
                });
                result_heap.push(Candidate {
                    id: neighbor_id,
                    distance: d,
                });
                if result_heap.len() > ef {
                    result_heap.pop();
                }
            }
        }
    }

    let mut results: Vec<(usize, f32)> = result_heap
        .into_iter()
        .map(|c| (c.id, c.distance))
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    (results, dist_computations)
}

/// Top-level KNN search: traverse from top layer to layer 0.
pub fn knn_search(graph: &Graph, query: &[f32], k: usize, ef_search: usize) -> Vec<(usize, f32)> {
    let entry_point = match graph.entry_point {
        Some(ep) => ep,
        None => return vec![],
    };

    let mut current_entry_points = vec![entry_point];

    // Traverse from top layer down to layer 1, using ef=1 (greedy)
    if graph.max_layer > 0 {
        for layer in (1..=graph.max_layer).rev() {
            let results = search_layer(graph, query, &current_entry_points, 1, layer);
            current_entry_points = results.into_iter().map(|(id, _)| id).collect();
        }
    }

    // Search layer 0 with full ef_search
    let mut results = search_layer(graph, query, &current_entry_points, ef_search, 0);
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dist_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((dist(&a, &a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dist_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((dist(&a, &b) - 1.0).abs() < 1e-6);
    }
}

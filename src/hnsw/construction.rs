use rand::Rng;

use super::graph::Graph;
use super::search::search_layer;
use crate::distance::dot_product;

/// Distance: 1 - dot_product (for normalized vectors).
#[inline]
fn dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b)
}

/// Select a random layer for a new node using the HNSW exponential distribution.
pub fn random_layer(m: usize, rng: &mut impl Rng) -> usize {
    let m_l = 1.0 / (m as f32).ln();
    let r: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0f64);
    (-r.ln() * m_l as f64).floor().min(32.0) as usize
}

/// Insert a new vector into the HNSW graph.
/// The vector should already be L2-normalized.
pub fn insert(
    graph: &mut Graph,
    vector: Vec<f32>,
    ef_construction: usize,
    rng: &mut impl Rng,
) -> usize {
    let new_layer = random_layer(graph.m, rng);
    let new_id = graph.add_node(vector, new_layer);

    // First node — set as entry point and return
    if graph.nodes.len() == 1 {
        graph.entry_point = Some(new_id);
        graph.max_layer = new_layer;
        return new_id;
    }

    let entry_point = graph.entry_point.unwrap();
    let query = graph.nodes[new_id].vector.clone();

    let mut current_entry_points = vec![entry_point];

    // Phase 1: Traverse from top layer down to new_layer + 1 (greedy, ef=1)
    if graph.max_layer > new_layer {
        for layer in (new_layer + 1..=graph.max_layer).rev() {
            let results = search_layer(graph, &query, &current_entry_points, 1, layer);
            current_entry_points = results.into_iter().map(|(id, _)| id).collect();
        }
    }

    // Phase 2: From min(new_layer, max_layer) down to 0, search and connect
    let start_layer = std::cmp::min(new_layer, graph.max_layer);
    for layer in (0..=start_layer).rev() {
        let results = search_layer(graph, &query, &current_entry_points, ef_construction, layer);

        // Select M neighbors (use M, not Mmax — paper Algorithm 2)
        let neighbors: Vec<usize> = results.iter().take(graph.m).map(|&(id, _)| id).collect();

        // Connect new node to neighbors
        graph.set_neighbors(new_id, layer, neighbors.clone());

        // Add bidirectional connections
        for &neighbor_id in &neighbors {
            let mut neighbor_connections: Vec<usize> =
                graph.get_neighbors(neighbor_id, layer).to_vec();
            neighbor_connections.push(new_id);

            // Prune if exceeds max connections for this layer
            let max_conn = graph.max_connections(layer);
            if neighbor_connections.len() > max_conn {
                let neighbor_vec = &graph.nodes[neighbor_id].vector.clone();
                let mut scored: Vec<(usize, f32)> = neighbor_connections
                    .iter()
                    .map(|&id| {
                        let d = dist(neighbor_vec, &graph.nodes[id].vector);
                        (id, d)
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                neighbor_connections = scored
                    .into_iter()
                    .take(max_conn)
                    .map(|(id, _)| id)
                    .collect();
            }

            graph.set_neighbors(neighbor_id, layer, neighbor_connections);
        }

        // Use found results as entry points for next layer down
        current_entry_points = results.into_iter().map(|(id, _)| id).collect();
    }

    // Update entry point if new node has higher layer
    if new_layer > graph.max_layer {
        graph.entry_point = Some(new_id);
        graph.max_layer = new_layer;
    }

    new_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_layer_distribution() {
        let mut rng = rand::thread_rng();
        let m = 16;
        let mut layers = [0usize; 10];
        for _ in 0..10000 {
            let l = random_layer(m, &mut rng);
            if l < layers.len() {
                layers[l] += 1;
            }
        }
        assert!(layers[0] > layers[1]);
        assert!(layers[1] > layers[2]);
    }
}

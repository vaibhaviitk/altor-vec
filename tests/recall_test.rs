use altor_vec::distance::{dot_product, normalize};
use altor_vec::HnswIndex;
use rand::Rng;

fn random_unit_vector(dims: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
    normalize(&mut v);
    v
}

fn brute_force_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, 1.0 - dot_product(v, query)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).map(|(id, _)| id).collect()
}

fn measure_recall(
    index: &HnswIndex,
    vectors: &[Vec<f32>],
    dims: usize,
    k: usize,
    num_queries: usize,
    rng: &mut impl Rng,
) -> f32 {
    let mut total_recall = 0.0f32;
    for _ in 0..num_queries {
        let query = random_unit_vector(dims, rng);
        let hnsw_results: Vec<usize> = index.search(&query, k).iter().map(|&(id, _)| id).collect();
        let brute_results = brute_force_knn(vectors, &query, k);
        let overlap = hnsw_results
            .iter()
            .filter(|id| brute_results.contains(id))
            .count();
        total_recall += overlap as f32 / k as f32;
    }
    total_recall / num_queries as f32
}

/// Test recall on 10K random 384d vectors.
///
/// NOTE: Random unit vectors in 384 dimensions are a pathological case for ANN search
/// because all pairwise distances concentrate around 1.0 ± 0.05. Even the reference
/// hnswlib implementation only achieves ~0.36 recall@10 with ef_search=50 on this data.
///
/// For random data, we use ef_search=500 to achieve >0.93 recall.
/// For real embedding data (e.g., sentence-transformers), ef_search=50 suffices.
#[test]
fn test_recall_at_10() {
    let dims = 384;
    let n = 10_000;
    let num_queries = 100;
    let k = 10;
    let m = 16;
    let ef_construction = 200;
    // ef_search=500 needed for random data; real embeddings would use ef_search=50
    let ef_search = 500;

    let mut rng = rand::thread_rng();

    // Build index
    let mut index = HnswIndex::new(dims, m, ef_construction, ef_search);
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);

    for _ in 0..n {
        let v = random_unit_vector(dims, &mut rng);
        vectors.push(v.clone());
        index.insert(v, &mut rng);
    }

    assert_eq!(index.len(), n);

    let avg_recall = measure_recall(&index, &vectors, dims, k, num_queries, &mut rng);
    println!("Recall@10 (ef_search={ef_search}) = {avg_recall:.4}");
    assert!(
        avg_recall > 0.93,
        "Recall@10 = {avg_recall:.4}, expected > 0.93"
    );
}

/// Verify recall matches reference (hnswlib) behavior with standard params.
/// hnswlib gets ~0.36 on random 384d data with ef_search=50.
#[test]
fn test_recall_matches_reference() {
    let dims = 384;
    let n = 10_000;
    let num_queries = 50;
    let k = 10;
    let m = 16;
    let ef_construction = 200;
    let ef_search = 50;

    let mut rng = rand::thread_rng();
    let mut index = HnswIndex::new(dims, m, ef_construction, ef_search);
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);

    for _ in 0..n {
        let v = random_unit_vector(dims, &mut rng);
        vectors.push(v.clone());
        index.insert(v, &mut rng);
    }

    let avg_recall = measure_recall(&index, &vectors, dims, k, num_queries, &mut rng);
    println!("Recall@10 (ef_search={ef_search}) = {avg_recall:.4}");
    // hnswlib gets ~0.36 on this exact setup. We should be in the same ballpark.
    assert!(
        avg_recall > 0.25,
        "Recall@10 = {avg_recall:.4}, too low — possible algorithm bug"
    );
}

#[test]
fn test_basic_search() {
    let dims = 4;
    let mut rng = rand::thread_rng();
    let mut index = HnswIndex::new(dims, 4, 20, 10);

    for _ in 0..100 {
        let v = random_unit_vector(dims, &mut rng);
        index.insert(v, &mut rng);
    }

    let query = random_unit_vector(dims, &mut rng);
    let results = index.search(&query, 5);
    assert_eq!(results.len(), 5);

    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1);
    }
}

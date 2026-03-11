use altor_vec::HnswIndex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn random_vector(dims: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

#[test]
fn test_empty_index_search() {
    let index = HnswIndex::new(4, 16, 200, 50);
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = index.search(&query, 5);
    assert!(results.is_empty());
}

#[test]
fn test_single_vector_index() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(4, 16, 200, 50);
    index.insert(vec![1.0, 0.0, 0.0, 0.0], &mut rng);

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].1 < 1e-5,
        "Distance to identical vector should be ~0"
    );
}

#[test]
fn test_duplicate_vectors() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(3, 16, 200, 50);

    let v = vec![1.0, 0.0, 0.0];
    for _ in 0..10 {
        index.insert(v.clone(), &mut rng);
    }
    assert_eq!(index.len(), 10);

    let results = index.search(&v, 10);
    assert_eq!(results.len(), 10);
    // All distances should be essentially 0
    for (_, dist) in &results {
        assert!(
            *dist < 1e-5,
            "Distance to duplicate vector should be ~0, got {dist}"
        );
    }
}

#[test]
#[should_panic(expected = "Vector dimension mismatch")]
fn test_insert_dimension_mismatch() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(4, 16, 200, 50);
    index.insert(vec![1.0, 0.0], &mut rng); // wrong dims
}

#[test]
#[should_panic(expected = "Query dimension mismatch")]
fn test_search_dimension_mismatch() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(4, 16, 200, 50);
    index.insert(vec![1.0, 0.0, 0.0, 0.0], &mut rng);
    index.search(&[1.0, 0.0], 5); // wrong dims
}

#[test]
fn test_zero_vector_insert() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(3, 16, 200, 50);
    // Zero vector — normalize should leave it as zeros
    index.insert(vec![0.0, 0.0, 0.0], &mut rng);
    assert_eq!(index.len(), 1);

    // Search should still work without panicking
    let results = index.search(&[1.0, 0.0, 0.0], 1);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_topk_greater_than_index_size() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(4, 16, 200, 50);

    for _ in 0..3 {
        index.insert(random_vector(4, &mut rng), &mut rng);
    }

    // Ask for more results than vectors in the index
    let results = index.search(&random_vector(4, &mut rng), 100);
    assert_eq!(
        results.len(),
        3,
        "Should return all vectors when topK > index size"
    );
}

#[test]
fn test_search_results_sorted_by_distance() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(8, 16, 200, 50);

    for _ in 0..200 {
        index.insert(random_vector(8, &mut rng), &mut rng);
    }

    let results = index.search(&random_vector(8, &mut rng), 20);
    for i in 1..results.len() {
        assert!(
            results[i].1 >= results[i - 1].1,
            "Results not sorted: {} > {} at position {}",
            results[i - 1].1,
            results[i].1,
            i
        );
    }
}

#[test]
fn test_is_empty() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(4, 16, 200, 50);
    assert!(index.is_empty());

    index.insert(vec![1.0, 0.0, 0.0, 0.0], &mut rng);
    assert!(!index.is_empty());
}

#[test]
fn test_high_dimensional_vectors() {
    let dims = 1536; // OpenAI text-embedding-3-small dimension
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(dims, 16, 200, 50);

    for _ in 0..50 {
        index.insert(random_vector(dims, &mut rng), &mut rng);
    }

    let results = index.search(&random_vector(dims, &mut rng), 5);
    assert_eq!(results.len(), 5);
}

#[test]
fn test_search_with_k_zero() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(4, 16, 200, 50);
    index.insert(vec![1.0, 0.0, 0.0, 0.0], &mut rng);

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 0);
    assert!(results.is_empty());
}

#[test]
fn test_nearest_neighbor_is_self() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(4, 16, 200, 50);

    // Insert a known vector along with noise
    let target = vec![1.0, 0.0, 0.0, 0.0];
    let target_id = index.insert(target.clone(), &mut rng);

    for _ in 0..50 {
        index.insert(random_vector(4, &mut rng), &mut rng);
    }

    let results = index.search(&target, 1);
    assert_eq!(
        results[0].0, target_id,
        "Nearest neighbor of a vector should be itself"
    );
}

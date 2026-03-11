#![cfg(feature = "serialization")]

use altor_vec::HnswIndex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn random_vector(dims: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

#[test]
fn test_serialization_round_trip() {
    let dims = 32;
    let n = 1_000;
    let m = 16;
    let ef_construction = 200;
    let ef_search = 50;

    let mut rng = SmallRng::seed_from_u64(42);

    // Build index
    let mut index = HnswIndex::new(dims, m, ef_construction, ef_search);
    for _ in 0..n {
        let v = random_vector(dims, &mut rng);
        index.insert(v, &mut rng);
    }
    assert_eq!(index.len(), n);

    // Serialize and deserialize
    let bytes = index.to_bytes();
    let restored = HnswIndex::from_bytes(&bytes);

    assert_eq!(restored.len(), n);
    assert_eq!(restored.dims, dims);
    assert_eq!(restored.ef_search, ef_search);
    assert_eq!(restored.ef_construction, ef_construction);

    // Run queries and verify identical results
    let num_queries = 50;
    let k = 10;
    for _ in 0..num_queries {
        let query = random_vector(dims, &mut rng);
        let original_results = index.search(&query, k);
        let restored_results = restored.search(&query, k);

        assert_eq!(original_results.len(), restored_results.len());
        for (orig, rest) in original_results.iter().zip(restored_results.iter()) {
            assert_eq!(orig.0, rest.0, "Node IDs differ");
            assert!(
                (orig.1 - rest.1).abs() < 1e-6,
                "Distances differ: {} vs {}",
                orig.1,
                rest.1
            );
        }
    }
}

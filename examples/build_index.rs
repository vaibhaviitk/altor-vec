use altor_vec::HnswIndex;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fs;

fn main() {
    // Read embeddings.bin: u32 LE count + u32 LE dims + flat f32s
    let emb_bytes = fs::read("data/embeddings.bin").expect("failed to read data/embeddings.bin");
    assert!(emb_bytes.len() >= 8, "embeddings.bin too small");

    let count = u32::from_le_bytes(emb_bytes[0..4].try_into().unwrap()) as usize;
    let dims = u32::from_le_bytes(emb_bytes[4..8].try_into().unwrap()) as usize;
    println!("Embeddings: {count} vectors of {dims} dimensions");

    let float_bytes = &emb_bytes[8..];
    assert_eq!(
        float_bytes.len(),
        count * dims * 4,
        "embeddings.bin size mismatch"
    );

    let vectors: Vec<f32> = float_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // Build HNSW index: ef_search=50 (real embeddings, not random)
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(dims, 16, 200, 50);

    println!("Building index...");
    for (i, chunk) in vectors.chunks_exact(dims).enumerate() {
        index.insert(chunk.to_vec(), &mut rng);
        if (i + 1) % 1000 == 0 {
            println!("  inserted {}/{count}", i + 1);
        }
    }

    // Serialize index
    let bytes = index.to_bytes();
    fs::write("data/index.bin", &bytes).expect("failed to write data/index.bin");
    println!(
        "Wrote data/index.bin ({:.2} MB)",
        bytes.len() as f64 / (1024.0 * 1024.0)
    );

    // Copy docs.json → metadata.json
    fs::copy("data/docs.json", "data/metadata.json")
        .expect("failed to copy docs.json to metadata.json");
    println!("Copied data/docs.json → data/metadata.json");
}

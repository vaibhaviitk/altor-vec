use altor_vec::HnswIndex;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fs;

fn main() {
    let dims = 384;
    let n = 10_000;
    let m = 16;
    let ef_construction = 200;
    let ef_search = 50;

    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::new(dims, m, ef_construction, ef_search);

    println!("Building {n} vectors of {dims} dimensions...");
    for i in 0..n {
        let v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
        index.insert(v, &mut rng);
        if (i + 1) % 1000 == 0 {
            println!("  inserted {}/{n}", i + 1);
        }
    }

    let bytes = index.to_bytes();
    let path = "test_index.bin";
    fs::write(path, &bytes).expect("failed to write test_index.bin");
    println!(
        "Wrote {} ({:.2} MB)",
        path,
        bytes.len() as f64 / (1024.0 * 1024.0)
    );
}

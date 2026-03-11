use criterion::{criterion_group, criterion_main, Criterion};
use altor_vec::distance::normalize;
use altor_vec::HnswIndex;
use rand::Rng;

fn random_unit_vector(dims: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0..1.0)).collect();
    normalize(&mut v);
    v
}

fn build_test_index(n: usize, dims: usize, ef_search: usize) -> HnswIndex {
    let mut rng = rand::thread_rng();
    let mut index = HnswIndex::new(dims, 16, 200, ef_search);
    for _ in 0..n {
        let v = random_unit_vector(dims, &mut rng);
        index.insert(v, &mut rng);
    }
    index
}

fn bench_search_ef50(c: &mut Criterion) {
    let index = build_test_index(10_000, 384, 50);
    let mut rng = rand::thread_rng();
    let query = random_unit_vector(384, &mut rng);

    c.bench_function("search_10k_384d_ef50", |b| {
        b.iter(|| index.search(&query, 10))
    });
}

fn bench_search_ef500(c: &mut Criterion) {
    let index = build_test_index(10_000, 384, 500);
    let mut rng = rand::thread_rng();
    let query = random_unit_vector(384, &mut rng);

    c.bench_function("search_10k_384d_ef500", |b| {
        b.iter(|| index.search(&query, 10))
    });
}

fn bench_search_1k(c: &mut Criterion) {
    let index = build_test_index(1_000, 384, 50);
    let mut rng = rand::thread_rng();
    let query = random_unit_vector(384, &mut rng);

    c.bench_function("search_1k_384d_ef50", |b| {
        b.iter(|| index.search(&query, 10))
    });
}

criterion_group!(benches, bench_search_ef50, bench_search_ef500, bench_search_1k);
criterion_main!(benches);

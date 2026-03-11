/// Compute dot product of two f32 slices.
/// When vectors are L2-normalized, this equals cosine similarity.
/// This will auto-vectorize with SIMD in release mode.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalize a vector in place. Returns the original norm.
pub fn normalize(v: &mut [f32]) -> f32 {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_identity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((dot_product(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((dot_product(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        assert!((dot_product(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_identical_vectors() {
        let a = vec![0.6, 0.8];
        // dot(a, a) = 0.36 + 0.64 = 1.0
        assert!((dot_product(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_high_dimensions() {
        let dims = 1536;
        let a: Vec<f32> = (0..dims).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..dims).map(|i| (i as f32).cos()).collect();
        // Just verify it runs without panicking and produces a finite result
        let result = dot_product(&a, &b);
        assert!(result.is_finite());
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        let norm = normalize(&mut v);
        assert!((norm - 0.0).abs() < 1e-6);
        // Vector should remain zero
        for x in &v {
            assert!((*x).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize_already_unit() {
        let mut v = vec![1.0, 0.0, 0.0];
        let norm = normalize(&mut v);
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_returns_original_norm() {
        let mut v = vec![3.0, 4.0];
        let norm = normalize(&mut v);
        assert!((norm - 5.0).abs() < 1e-6);
    }
}

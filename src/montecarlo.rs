//! Reproducible Monte-Carlo ensembles (T4).
//!
//! Realizations are embarrassingly parallel over rayon, but reproducibility
//! must not depend on thread count or scheduling. The contract:
//!
//! - the closure derives all randomness from its realization index (e.g.
//!   `ChaCha12Rng::seed_from_u64(master).set_stream(i)`), never from thread
//!   state or entropy;
//! - results come back as a `Vec` in realization order, so the caller's
//!   reduction runs in a fixed sequential order regardless of how the work
//!   was scheduled.

use rayon::prelude::*;

/// Run `n` realizations of `f` in parallel; `f(i)` receives the realization
/// index and must be deterministic in it. Results are in index order.
pub fn seeded_ensemble<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(u64) -> T + Sync + Send,
{
    (0..n as u64).into_par_iter().map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_come_back_in_index_order() {
        let v = seeded_ensemble(64, |i| i * i);
        assert_eq!(v.len(), 64);
        assert!(v.iter().enumerate().all(|(i, &x)| x == (i * i) as u64));
    }

    #[test]
    fn deterministic_across_pool_sizes() {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| seeded_ensemble(32, |i| (i as f64 * 0.1).sin()))
        };
        assert_eq!(run(1), run(4));
    }
}

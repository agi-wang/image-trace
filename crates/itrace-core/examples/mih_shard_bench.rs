//! Cheap microbench for `ShardedMihIndex` (no criterion).
//!
//! ```bash
//! cargo run -p itrace-core --release --example mih_shard_bench
//! ```

use std::time::Instant;

use itrace_core::index::ShardedMihIndex;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn main() {
    const N: usize = 50_000;
    const SHARD_BITS: u32 = 6; // 64 shards
    const QUERY_N: usize = 1_000;
    const RADIUS: u32 = 7;

    let mut rng = 0xA5A5_5A5A_DEAD_BEEFu64;
    let keys: Vec<u64> = (0..N).map(|_| xorshift(&mut rng)).collect();

    let t0 = Instant::now();
    let mut idx = ShardedMihIndex::new(SHARD_BITS);
    for (i, &k) in keys.iter().enumerate() {
        idx.insert(k, (i % 10_000) as u32);
    }
    let build = t0.elapsed();

    let probes: Vec<u64> = (0..QUERY_N)
        .map(|i| if i % 2 == 0 { keys[i * 17 % N] } else { xorshift(&mut rng) })
        .collect();

    let t1 = Instant::now();
    let mut hit_owners = 0usize;
    for &q in &probes {
        hit_owners += idx.query(q, RADIUS).len();
    }
    let query = t1.elapsed();

    // ≈44 B/key + ≈48 KiB fixed tables per shard
    let est_bytes = N * 44 + idx.shard_count() * 48 * 1024;
    let est_disk = N * 12; // u64 key + u32 owner per entry

    println!("ShardedMihIndex microbench");
    println!("  keys          = {N}");
    println!("  shard_bits    = {SHARD_BITS} ({} shards)", idx.shard_count());
    println!("  build         = {:.3} ms", build.as_secs_f64() * 1e3);
    println!(
        "  query×{QUERY_N}   = {:.3} ms  (radius={RADIUS}, total owner hits={hit_owners})",
        query.as_secs_f64() * 1e3
    );
    println!(
        "  est RAM        ≈ {:.2} MB  (44 B/key + 48 KiB×shards)",
        est_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  est disk       ≈ {:.2} MB  (12 B/key compact)",
        est_disk as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  extrapolated   ≈ {:.1} GB payload @ 1e8 imgs × 3 algos × 8 vars",
        (1e8_f64 * 3.0 * 8.0 * 44.0) / (1024.0 * 1024.0 * 1024.0)
    );
}

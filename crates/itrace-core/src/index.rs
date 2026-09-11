//! Candidate-pair retrieval for large-scale dedup.
//!
//! At 10^8+ images an N×N matrix is infeasible; instead we index each image's
//! per-variant 64-bit gate-hash keys in Multi-Index Hash (MIH) tables — one
//! index per gate algorithm — and emit only near-duplicate candidate pairs
//! for exact re-scoring. Indexing all 8 orientation variants reproduces
//! variant-max comparison exactly (a rotated file's variant keys are a
//! permutation of the original's).
//!
//! MIH: split the 64-bit key into eight 8-bit substrings; any two keys that
//! agree on a whole substring are candidates. Hamming ≤ 7 ⇒ guaranteed recall
//! (some substring must be identical); larger radii keep high empirical recall
//! and are re-verified by exact cross-variant scoring downstream.
//!
//! Memory ≈ 44 B per indexed key (u64 key + u32 owner + 8×u32 table refs)
//! plus a fixed ~48 KiB per index for the direct-mapped bucket arrays;
//! 8 keys/image/algorithm ⇒ ~350 B/image per algorithm index, i.e. ~35 GB
//! per 100M-image index — shard by project or key-prefix beyond that.
//! `canonical_rot64` remains as a single-key fast path for exact
//! rotation/flip duplicates of ahash-style equivariant hashes.

use std::collections::HashMap;

use rayon::prelude::*;

const SUBS: usize = 8; // 64 bits → 8 × 8-bit substrings
const SUBKEYS: usize = 256; // distinct values of one 8-bit substring

/// Apply a dihedral transform to an 8×8 bit matrix packed row-major in a u64.
/// Reference implementation — kept for the equivalence test; the hot path
/// uses the branch-free SWAR versions below.
#[cfg(test)]
fn xform8x8(h: u64, f: fn(usize, usize) -> (usize, usize)) -> u64 {
    let mut out = 0u64;
    for i in 0..8usize {
        for j in 0..8usize {
            if (h >> (i * 8 + j)) & 1 == 1 {
                let (ni, nj) = f(i, j);
                out |= 1u64 << (ni * 8 + nj);
            }
        }
    }
    out
}

/// SWAR transpose of the packed 8×8 bit matrix (i,j) → (j,i): three
/// masked delta-swaps (off-diagonal distances 7, 14, 28 bits).
#[inline]
fn transpose8x8(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AA;
    x ^= t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC;
    x ^= t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0;
    x ^ t ^ (t << 28)
}

/// SWAR horizontal flip (i,j) → (i,7-j): reverse the bits of each byte.
#[inline]
fn fliph8x8(mut x: u64) -> u64 {
    x = ((x >> 1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1);
    x = ((x >> 2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2);
    ((x >> 4) & 0x0F0F0F0F0F0F0F0F) | ((x & 0x0F0F0F0F0F0F0F0F) << 4)
}

/// SWAR vertical flip (i,j) → (7-i,j): reverse byte order.
#[inline]
fn flipv8x8(x: u64) -> u64 {
    x.swap_bytes()
}

/// Rotation/flip-invariant canonical form of an 8×8 perceptual hash:
/// the minimum u64 over all 8 dihedral transforms. Two images that are exact
/// 90°-rotations or mirrors of each other share the same canonical key.
/// (Near-duplicate transforms land within a few bits — covered by MIH radius.)
///
/// The eight transforms are generated from transpose T, horizontal flip H
/// and vertical flip V (each a few SWAR ops): rot90 = H·T, rot270 = V·T,
/// rot180 = V·H, anti-transpose = H·V·T.
pub fn canonical_rot64(h: u64) -> u64 {
    let t = transpose8x8(h); // transpose
    let fh = fliph8x8(h); // flip h
    [
        h,
        fliph8x8(t),      // rot90
        fh.swap_bytes(),  // rot180 = V·H
        t.swap_bytes(),   // rot270 = V·T
        fh,               // flip h
        flipv8x8(h),      // flip v
        t,                // transpose
        fliph8x8(t.swap_bytes()), // anti-transpose = H·V·T
    ]
    .into_iter()
    .min()
    .unwrap()
}

/// Multi-Index Hash index over u64 keys — near O(1) candidate lookup on
/// Hamming space instead of O(N) scan. Each key carries an `owner` id
/// (e.g. image index) so multiple keys can map to one entity.
///
/// `tables[t]` is a direct-mapped array of 256 buckets indexed by the
/// 8-bit substring value — no hashing or bucket lookup misses on the
/// hot insert/query path.
pub struct MihIndex {
    keys: Vec<u64>,
    owners: Vec<u32>,
    tables: [Vec<Vec<u32>>; SUBS],
}

impl Default for MihIndex {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            owners: Vec::new(),
            tables: std::array::from_fn(|_| vec![Vec::new(); SUBKEYS]),
        }
    }
}

impl MihIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Insert `key` owned by `owner`.
    pub fn insert(&mut self, key: u64, owner: u32) {
        let pos = self.keys.len() as u32;
        self.keys.push(key);
        self.owners.push(owner);
        for (t, table) in self.tables.iter_mut().enumerate() {
            let sub = ((key >> (t * 8)) & 0xff) as usize;
            table[sub].push(pos);
        }
    }

    /// Owner ids of all stored keys within hamming `radius` of `key`,
    /// appended to `out` (sorted + deduped in place). Reusing one buffer
    /// across queries avoids an allocation per lookup.
    fn query_into(&self, key: u64, radius: u32, out: &mut Vec<u32>) {
        out.clear();
        for (t, table) in self.tables.iter().enumerate() {
            let sub = ((key >> (t * 8)) & 0xff) as usize;
            // Popcount-filter while streaming buckets; map positions to
            // owner ids directly — the final sort+dedup absorbs duplicate
            // positions (same key in several matching substrings) and
            // duplicate owners, so the result equals deduping positions
            // first, with one less sort pass.
            out.extend(
                table[sub]
                    .iter()
                    .copied()
                    .filter(|&p| (self.keys[p as usize] ^ key).count_ones() <= radius)
                    .map(|p| self.owners[p as usize]),
            );
        }
        out.sort_unstable();
        out.dedup();
    }

    /// Owner ids of all stored keys within hamming `radius` of `key`.
    pub fn query(&self, key: u64, radius: u32) -> Vec<u32> {
        let mut owners = Vec::new();
        self.query_into(key, radius, &mut owners);
        owners
    }

    /// Insert `key` for `owner`; return prior owners within `radius`.
    pub fn insert_query(&mut self, key: u64, owner: u32, radius: u32) -> Vec<u32> {
        let hits = self.query(key, radius);
        self.insert(key, owner);
        hits
    }
}

/// Fast `Hasher` for the u32 owner ids in the per-entry vote map — the std
/// SipHash default measurably dominates when each entry only sees a handful
/// of candidate hits. One multiply-xor round (fxhash-style) spreads
/// sequential ids well enough for power-of-two tables.
#[derive(Default)]
struct VoteHasher(u64);

impl std::hash::Hasher for VoteHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 =
                (self.0.rotate_left(5) ^ u64::from(b)).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(n)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

/// owner-id → per-algo vote bitmask.
type VoteMap = HashMap<u32, u32, std::hash::BuildHasherDefault<VoteHasher>>;

/// One row per image: per-algo variant keys (`variant_keys[a][v]` = the
/// 64-bit hash of orientation variant v under gate-hash algo a).
/// Indexing all variants reproduces variant-max semantics exactly.
pub struct DedupKeys {
    pub image_id: i64,
    pub variant_keys: Vec<Vec<u64>>,
}

/// Emit candidate image-index pairs flagged within `radius` bits by at
/// least `min_votes` DISTINCT gate algorithms (one vote per algorithm,
/// regardless of how many variant keys matched). Returns (i, j), i < j,
/// sorted and deduplicated.
///
/// One fully-populated MIH index is built per algorithm (in parallel),
/// then every entry queries every index (in parallel). A pair within
/// `radius` is found from both directions and normalized to (min, max),
/// so the emitted set is identical to a sequential insert-then-query
/// scan — a hit under algo a sets bit a of the pair's vote mask.
pub fn dedup_candidates(entries: &[DedupKeys], radius: u32, min_votes: u32) -> Vec<(u32, u32)> {
    if entries.is_empty() || entries[0].variant_keys.is_empty() {
        return Vec::new();
    }
    let m = entries[0].variant_keys.len();
    let indexes: Vec<MihIndex> = (0..m)
        .into_par_iter()
        .map(|a| {
            let mut idx = MihIndex::new();
            let mut uniq: Vec<u64> = Vec::new(); // reused across entries
            for (i, e) in entries.iter().enumerate() {
                uniq.clear();
                uniq.extend_from_slice(&e.variant_keys[a]);
                uniq.sort_unstable();
                uniq.dedup();
                for &key in &uniq {
                    idx.insert(key, i as u32);
                }
            }
            idx
        })
        .collect();
    let mut out: Vec<(u32, u32)> = entries
        .par_iter()
        .enumerate()
        .flat_map(|(i, e)| {
            let i = i as u32;
            // algo bitmask per other-image: a match under algo a sets bit a
            let mut hit = VoteMap::default();
            // Scratch buffers reused across every key query of this entry.
            let mut uniq: Vec<u64> = Vec::new();
            let mut hits: Vec<u32> = Vec::new();
            for (a, idx) in indexes.iter().enumerate() {
                uniq.clear();
                uniq.extend_from_slice(&e.variant_keys[a]);
                uniq.sort_unstable();
                uniq.dedup();
                for &key in &uniq {
                    idx.query_into(key, radius, &mut hits);
                    for &j in &hits {
                        if j != i {
                            *hit.entry(j).or_insert(0) |= 1 << a;
                        }
                    }
                }
            }
            hit.into_iter()
                .filter(|(_, mask)| mask.count_ones() >= min_votes)
                .map(|(j, _)| (j.min(i), j.max(i)))
                .collect::<Vec<_>>()
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each SWAR dihedral transform must equal the reference bit-loop
    /// transform on edge patterns and pseudo-random hashes.
    #[test]
    fn swar_transforms_match_reference() {
        type CoordMap = fn(usize, usize) -> (usize, usize);
        let transforms: [CoordMap; 8] = [
            |i, j| (j, 7 - i),     // rot90
            |i, j| (7 - i, 7 - j), // rot180
            |i, j| (7 - j, i),     // rot270
            |i, j| (i, 7 - j),     // flip h
            |i, j| (7 - i, j),     // flip v
            |i, j| (j, i),         // transpose
            |i, j| (7 - j, 7 - i), // anti-transpose
            |i, j| (i, j),         // identity
        ];
        let swar: [fn(u64) -> u64; 8] = [
            |x| fliph8x8(transpose8x8(x)),
            |x| fliph8x8(x).swap_bytes(),
            |x| transpose8x8(x).swap_bytes(),
            fliph8x8,
            flipv8x8,
            transpose8x8,
            |x| fliph8x8(transpose8x8(x).swap_bytes()),
            |x| x,
        ];
        let mut v = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            v ^= v << 13;
            v ^= v >> 7;
            v ^= v << 17;
            v
        };
        let edges = [
            0u64,
            u64::MAX,
            1,
            0x8000000000000000,
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
            0x00FF00FF00FF00FF,
            0xFF00FF00FF00FF00,
            0x0101010101010101,
            0x8080808080808080,
        ];
        for i in 0..512 {
            let h = if i < edges.len() { edges[i] } else { next() };
            for (f, s) in transforms.iter().zip(swar.iter()) {
                assert_eq!(s(h), xform8x8(h, *f), "transform mismatch at {h:#x}");
            }
        }
    }
}

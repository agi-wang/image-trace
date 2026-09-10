//! Candidate-pair retrieval for large-scale dedup.
//!
//! At 10^8+ images an N×N matrix is infeasible; instead we index each image's
//! rotation-canonicalized 64-bit hash in a Multi-Index Hash (MIH) table and
//! emit only near-duplicate candidate pairs for exact re-scoring — near-linear
//! total work with sub-linear per-image candidate sets.
//!
//! MIH: split the 64-bit key into eight 8-bit substrings; any two keys that
//! agree on a whole substring are candidates. Hamming ≤ 7 ⇒ guaranteed recall
//! (some substring must be identical); larger radii keep high empirical recall
//! and are re-verified by exact variant-max scoring downstream.
//!
//! Memory ≈ 40 bytes/image/index (u64 key + 8×u32 table refs) — ~4 GB per
//! 100M-image index. For sharding beyond that, run one index per project or
//! key-prefix.

use std::collections::HashMap;

const SUBS: usize = 8; // 64 bits → 8 × 8-bit substrings

/// Apply a dihedral transform to an 8×8 bit matrix packed row-major in a u64.
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

/// Rotation/flip-invariant canonical form of an 8×8 perceptual hash:
/// the minimum u64 over all 8 dihedral transforms. Two images that are exact
/// 90°-rotations or mirrors of each other share the same canonical key.
/// (Near-duplicate transforms land within a few bits — covered by MIH radius.)
pub fn canonical_rot64(h: u64) -> u64 {
    [
        h,
        xform8x8(h, |i, j| (j, 7 - i)),          // rot90
        xform8x8(h, |i, j| (7 - i, 7 - j)),      // rot180
        xform8x8(h, |i, j| (7 - j, i)),          // rot270
        xform8x8(h, |i, j| (i, 7 - j)),          // flip h
        xform8x8(h, |i, j| (7 - i, j)),          // flip v
        xform8x8(h, |i, j| (j, i)),              // transpose
        xform8x8(h, |i, j| (7 - j, 7 - i)),      // anti-transpose
    ]
    .into_iter()
    .min()
    .unwrap()
}

/// Multi-Index Hash index over u64 keys — near O(1) candidate lookup on
/// Hamming space instead of O(N) scan. Each key carries an `owner` id
/// (e.g. image index) so multiple keys can map to one entity.
#[derive(Default)]
pub struct MihIndex {
    keys: Vec<u64>,
    owners: Vec<u32>,
    tables: [HashMap<u8, Vec<u32>>; SUBS],
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
            let sub = ((key >> (t * 8)) & 0xff) as u8;
            table.entry(sub).or_default().push(pos);
        }
    }

    /// Owner ids of all stored keys within hamming `radius` of `key`.
    pub fn query(&self, key: u64, radius: u32) -> Vec<u32> {
        let mut cand: Vec<u32> = Vec::new();
        for (t, table) in self.tables.iter().enumerate() {
            let sub = ((key >> (t * 8)) & 0xff) as u8;
            if let Some(ids) = table.get(&sub) {
                cand.extend_from_slice(ids);
            }
        }
        cand.sort_unstable();
        cand.dedup();
        let mut owners: Vec<u32> = cand
            .into_iter()
            .filter(|&p| (self.keys[p as usize] ^ key).count_ones() <= radius)
            .map(|p| self.owners[p as usize])
            .collect();
        owners.sort_unstable();
        owners.dedup();
        owners
    }

    /// Insert `key` for `owner`; return prior owners within `radius`.
    pub fn insert_query(&mut self, key: u64, owner: u32, radius: u32) -> Vec<u32> {
        let hits = self.query(key, radius);
        self.insert(key, owner);
        hits
    }
}

/// One row per image: per-algo variant keys (`variant_keys[a][v]` = the
/// 64-bit hash of orientation variant v under gate-hash algo a).
/// Indexing all variants reproduces variant-max semantics exactly.
pub struct DedupKeys {
    pub image_id: i64,
    pub variant_keys: Vec<Vec<u64>>,
}

/// Emit candidate image-index pairs that at least `min_votes` gate hashes
/// flag within `radius` bits on ANY variant pair. Returns (i, j), i < j.
pub fn dedup_candidates(entries: &[DedupKeys], radius: u32, min_votes: u32) -> Vec<(u32, u32)> {
    if entries.is_empty() || entries[0].variant_keys.is_empty() {
        return Vec::new();
    }
    let m = entries[0].variant_keys.len();
    let mut indexes: Vec<MihIndex> = (0..m).map(|_| MihIndex::new()).collect();
    let mut votes: HashMap<(u32, u32), u32> = HashMap::new();
    for (i, e) in entries.iter().enumerate() {
        let mut hit: HashMap<u32, u32> = HashMap::new();
        for (a, idx) in indexes.iter_mut().enumerate() {
            for &key in &e.variant_keys[a] {
                for j in idx.insert_query(key, i as u32, radius) {
                    if j != i as u32 {
                        *hit.entry(j).or_insert(0) += 1;
                    }
                }
            }
        }
        let i = i as u32;
        for (j, v) in hit {
            if v >= min_votes {
                votes.insert((j.min(i), j.max(i)), v);
            }
        }
    }
    votes.into_keys().collect()
}

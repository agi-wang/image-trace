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
//!
//! # Sharding (`ShardedMihIndex`)
//!
//! Keys are routed to one of `2^shard_bits` shards by the **high**
//! `shard_bits` of the u64 key (`key >> (64 - shard_bits)`). Queries probe
//! every shard whose prefix is within Hamming distance `radius` of the
//! query prefix — necessary for exact recall vs a monolithic `MihIndex`
//! (a near-duplicate may differ in a high bit). Prefer `shard_bits` in
//! `0..=16` (1 … 65 536 shards).
//!
//! # On-disk layout (`save_dir` / `load_dir`)
//!
//! ```text
//! <dir>/
//!   meta.json          {"magic":"ITMIH1","version":1,"shard_bits":N,"key_count":K}
//!   shards/
//!     0000.bin         …  NNNN.bin   (zero-padded 4-digit shard id)
//! ```
//!
//! Each `NNNN.bin` is little-endian compact:
//! `u64 n`, then `n × u64` keys, then `n × u32` owners. Bucket tables are
//! rebuilt on load (simplest correct approach).
//!
//! # Persistent project gate index (`ITRACE_MIH_INDEX_DIR`)
//!
//! When the env var is set, CLI/server dedup load-or-build a project-scoped
//! bundle under `{ITRACE_MIH_INDEX_DIR}/project_{id}/`:
//!
//! ```text
//! project_{id}/
//!   meta.json          {"magic":"ITMIHP1","version":1,"shard_bits":N,
//!                       "gate_algo_count":M,"image_count":K}
//!   image_ids.bin      K × i64 LE — owner slot → image_id (sorted ascending)
//!   indexes/
//!     0/ … M-1/        each a ShardedMihIndex (`ITMIH1`) for one gate algo
//! ```
//!
//! Owners are dense `u32` slots into `image_ids.bin` (NOT ephemeral enumerate
//! indices). On load we require exact match of `shard_bits`, `gate_algo_count`,
//! the sorted image_id set vs current ready entries, and a BLAKE3
//! `feature_fingerprint` over the indexed key material (canonical
//! sorted-by-image_id order); any mismatch rebuilds and overwrites. The
//! fingerprint catches feature-blob changes that leave the image set
//! unchanged — recomputed vectors, newly registered features, etc.
//!
//! With `ITRACE_MIH_NODES` > 1 the scan runs on in-process
//! `MultiNodeMihIndex` clusters and persists under the sibling
//! `project_{id}_mn/` (`ITMIHN1` meta + `indexes/{a}/` cluster dirs +
//! `node_N/` shards) with the same invalidation axes plus `node_count`.
//! The single-node `project_{id}/` (`ITMIHP1`) layout stays untouched.
//!
//! # Crop/slice recall channel (`project_{id}_crop/`)
//!
//! Whole-image gate hashes cannot recall crops/slices (they move too many
//! bits). The crop channel indexes each image's `crophash_keys` windowed
//! phashes — all window slots × orientation variants flattened into one key
//! set — in a single `ShardedMihIndex`, and emits pairs with `min_hits`
//! distinct probe-key hits. Candidates are verified by NCC containment
//! (`slice::contains_rot4`), so a loose recall radius is safe. The bundle
//! lives in its own `project_{id}_crop/` directory (`ITMIHC1` meta +
//! `image_ids.bin` + `index/`) with the same invalidation rules.
//!
//! With `ITRACE_MIH_NODES` > 1 the crop channel persists under
//! `project_{id}_crop_mn/` (`ITMIHCN1`) — contiguous sorted-image-id
//! ranges like `project_{id}_sem_mn/`, one `ITMIHC1` node bundle each —
//! and queries scatter to all node indexes (exact parity with the
//! single-node candidate set: each image's keys live in one node).

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
    pub fn query_into(&self, key: u64, radius: u32, out: &mut Vec<u32>) {
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

    /// Write the compact shard payload format used by `shards/NNNN.bin`:
    /// `u64 n`, then `n × u64` keys, then `n × u32` owners (LE).
    pub(crate) fn save_bin(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&(self.keys.len() as u64).to_le_bytes())?;
        for &k in &self.keys {
            f.write_all(&k.to_le_bytes())?;
        }
        for &o in &self.owners {
            f.write_all(&o.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load a `NNNN.bin` written by [`save_bin`](Self::save_bin); bucket
    /// tables are rebuilt by replaying inserts.
    pub(crate) fn load_bin(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut nbuf = [0u8; 8];
        f.read_exact(&mut nbuf)?;
        let n = u64::from_le_bytes(nbuf) as usize;
        let mut key_buf = [0u8; 8];
        let mut keys = Vec::with_capacity(n);
        for _ in 0..n {
            f.read_exact(&mut key_buf)?;
            keys.push(u64::from_le_bytes(key_buf));
        }
        let mut owner_buf = [0u8; 4];
        let mut owners = Vec::with_capacity(n);
        for _ in 0..n {
            f.read_exact(&mut owner_buf)?;
            owners.push(u32::from_le_bytes(owner_buf));
        }
        let mut idx = Self::new();
        for (&k, &o) in keys.iter().zip(&owners) {
            idx.insert(k, o);
        }
        Ok(idx)
    }
}

/// Route `key` to a shard id using the high `shard_bits` of the key.
///
/// Bit choice: `shard = key >> (64 - shard_bits)` when `shard_bits > 0`,
/// else shard `0`. High bits keep low-order MIH substrings (the ones MIH
/// tables hash on most naturally for nearby keys) co-located more often
/// than low-bit routing, and give a stable total order for multi-node
/// ownership by shard id.
#[inline]
pub fn shard_id_for(key: u64, shard_bits: u32) -> usize {
    if shard_bits == 0 {
        0
    } else {
        (key >> (64 - shard_bits)) as usize
    }
}

/// Multi-shard wrapper over [`MihIndex`]. See module docs for routing and
/// on-disk layout.
pub struct ShardedMihIndex {
    shard_bits: u32,
    shards: Vec<MihIndex>,
}

impl ShardedMihIndex {
    /// Create an empty index with `2^shard_bits` shards (`shard_bits == 0` → 1).
    /// Panics if `shard_bits > 16`.
    pub fn new(shard_bits: u32) -> Self {
        assert!(
            shard_bits <= 16,
            "shard_bits {shard_bits} exceeds sensible max 16"
        );
        let n = 1usize << shard_bits;
        Self {
            shard_bits,
            shards: (0..n).map(|_| MihIndex::new()).collect(),
        }
    }

    pub fn shard_bits(&self) -> u32 {
        self.shard_bits
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn shard_of(&self, key: u64) -> usize {
        shard_id_for(key, self.shard_bits)
    }

    pub fn insert(&mut self, key: u64, owner: u32) {
        let sid = self.shard_of(key);
        self.shards[sid].insert(key, owner);
    }

    /// Owner ids within Hamming `radius`, merged across all shards whose
    /// prefix is reachable under that radius.
    pub fn query_into(&self, key: u64, radius: u32, out: &mut Vec<u32>) {
        out.clear();
        let mut scratch = Vec::new();
        if self.shard_bits == 0 {
            self.shards[0].query_into(key, radius, out);
            return;
        }
        let prefix = self.shard_of(key) as u64;
        // Prefix Hamming distance is a lower bound on full-key distance, so
        // only shards with popcount(sid ^ prefix) ≤ radius can contribute.
        for (sid, shard) in self.shards.iter().enumerate() {
            if ((sid as u64) ^ prefix).count_ones() <= radius {
                shard.query_into(key, radius, &mut scratch);
                out.extend_from_slice(&scratch);
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    pub fn query(&self, key: u64, radius: u32) -> Vec<u32> {
        let mut owners = Vec::new();
        self.query_into(key, radius, &mut owners);
        owners
    }

    pub fn insert_query(&mut self, key: u64, owner: u32, radius: u32) -> Vec<u32> {
        let hits = self.query(key, radius);
        self.insert(key, owner);
        hits
    }

    /// Persist to `path/` (`meta.json` + `shards/NNNN.bin`). See module docs.
    pub fn save_dir(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::create_dir_all(path)?;
        let shards_dir = path.join("shards");
        std::fs::create_dir_all(&shards_dir)?;
        let meta = serde_json::json!({
            "magic": "ITMIH1",
            "version": 1,
            "shard_bits": self.shard_bits,
            "key_count": self.len(),
        });
        std::fs::write(
            path.join("meta.json"),
            serde_json::to_vec_pretty(&meta).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?,
        )?;
        for (i, shard) in self.shards.iter().enumerate() {
            shard.save_bin(&shards_dir.join(format!("{i:04}.bin")))?;
        }
        Ok(())
    }

    /// Load a directory written by [`save_dir`]. Rebuilds MIH tables from
    /// the compact key/owner arrays.
    pub fn load_dir(path: &std::path::Path) -> std::io::Result<Self> {
        let meta_bytes = std::fs::read(path.join("meta.json"))?;
        let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let magic = meta.get("magic").and_then(|v| v.as_str()).unwrap_or("");
        if magic != "ITMIH1" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad mih magic: {magic}"),
            ));
        }
        let version = meta.get("version").and_then(|v| v.as_u64()).unwrap_or(0);
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported mih version: {version}"),
            ));
        }
        let shard_bits = meta
            .get("shard_bits")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing shard_bits")
            })? as u32;
        if shard_bits > 16 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("shard_bits {shard_bits} > 16"),
            ));
        }
        let mut idx = Self::new(shard_bits);
        let shards_dir = path.join("shards");
        for sid in 0..idx.shard_count() {
            idx.shards[sid] = MihIndex::load_bin(&shards_dir.join(format!("{sid:04}.bin")))?;
        }
        Ok(idx)
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


/// Resolve MIH shard bit-width for sharded dedup.
///
/// Precedence: `override_bits` if `Some`, else env `ITRACE_MIH_SHARD_BITS`,
/// else default `8`. Clamped to `0..=16` (`0` = single shard / monolithic).
pub fn resolve_shard_bits(override_bits: Option<u32>) -> u32 {
    let raw = override_bits.or_else(|| {
        std::env::var("ITRACE_MIH_SHARD_BITS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
    });
    raw.unwrap_or(8).min(16)
}

/// Base directory for persisted project MIH bundles (`ITRACE_MIH_INDEX_DIR`).
/// Empty / unset → `None` (in-memory rebuild each scan).
pub fn resolve_mih_index_dir() -> Option<std::path::PathBuf> {
    let v = std::env::var_os("ITRACE_MIH_INDEX_DIR")?;
    if v.is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(v))
}

/// `{base}/project_{id}` — project-scoped gate-index bundle root.
pub fn project_mih_index_path(base: &std::path::Path, project_id: i64) -> std::path::PathBuf {
    base.join(format!("project_{project_id}"))
}

const PROJECT_MIH_MAGIC: &str = "ITMIHP1";
const PROJECT_MIH_VERSION: u64 = 1;

/// Sorted image_ids (owner slots) and per-entry owner ids for `entries`.
fn owner_plan(entries: &[DedupKeys]) -> (Vec<i64>, Vec<u32>) {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_unstable_by_key(|&i| entries[i].image_id);
    let image_ids: Vec<i64> = order.iter().map(|&i| entries[i].image_id).collect();
    let mut owner_for_entry = vec![0u32; entries.len()];
    for (owner, &ei) in order.iter().enumerate() {
        owner_for_entry[ei] = owner as u32;
    }
    (image_ids, owner_for_entry)
}

/// `owner_plan` for the crop channel's [`CropKeys`].
fn owner_plan_crop(entries: &[CropKeys]) -> (Vec<i64>, Vec<u32>) {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_unstable_by_key(|&i| entries[i].image_id);
    let image_ids: Vec<i64> = order.iter().map(|&i| entries[i].image_id).collect();
    let mut owner_for_entry = vec![0u32; entries.len()];
    for (owner, &ei) in order.iter().enumerate() {
        owner_for_entry[ei] = owner as u32;
    }
    (image_ids, owner_for_entry)
}

pub(crate) fn write_image_ids_bin(path: &std::path::Path, image_ids: &[i64]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for &id in image_ids {
        f.write_all(&id.to_le_bytes())?;
    }
    Ok(())
}

pub(crate) fn read_image_ids_bin(path: &std::path::Path, expected: usize) -> std::io::Result<Vec<i64>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut out = Vec::with_capacity(expected);
    let mut buf = [0u8; 8];
    loop {
        match f.read(&mut buf)? {
            0 => break,
            8 => out.push(i64::from_le_bytes(buf)),
            n => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("image_ids.bin truncated ({n} trailing bytes)"),
                ));
            }
        }
    }
    if expected > 0 && out.len() != expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "image_ids.bin len {} != meta image_count {expected}",
                out.len()
            ),
        ));
    }
    Ok(out)
}

pub(crate) fn image_id_sets_equal(persisted: &[i64], mut current: Vec<i64>) -> bool {
    if persisted.len() != current.len() {
        return false;
    }
    let mut a: Vec<i64> = persisted.to_vec();
    a.sort_unstable();
    current.sort_unstable();
    a == current
}

/// BLAKE3 hex over the gate channel's indexed key material in canonical
/// order: image_ids sorted ascending, then each entry's variant keys in
/// stored order. Persisted in the bundle meta so feature-blob changes
/// with an unchanged image_id set still invalidate the index.
fn gate_fingerprint(entries: &[DedupKeys]) -> String {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_unstable_by_key(|&i| entries[i].image_id);
    let mut h = blake3::Hasher::new();
    for &i in &order {
        let e = &entries[i];
        h.update(&e.image_id.to_le_bytes());
        h.update(&(e.variant_keys.len() as u32).to_le_bytes());
        for keys in &e.variant_keys {
            h.update(&(keys.len() as u32).to_le_bytes());
            for &k in keys {
                h.update(&k.to_le_bytes());
            }
        }
    }
    h.finalize().to_hex().to_string()
}

/// [`gate_fingerprint`] for the crop channel's flattened key sets.
fn crop_fingerprint(entries: &[CropKeys]) -> String {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_unstable_by_key(|&i| entries[i].image_id);
    let mut h = blake3::Hasher::new();
    for &i in &order {
        let e = &entries[i];
        h.update(&e.image_id.to_le_bytes());
        h.update(&(e.keys.len() as u32).to_le_bytes());
        for &k in &e.keys {
            h.update(&k.to_le_bytes());
        }
    }
    h.finalize().to_hex().to_string()
}

fn build_sharded_indexes(
    entries: &[DedupKeys],
    shard_bits: u32,
    owner_for_entry: &[u32],
) -> Vec<ShardedMihIndex> {
    let m = entries[0].variant_keys.len();
    (0..m)
        .into_par_iter()
        .map(|a| {
            let mut idx = ShardedMihIndex::new(shard_bits);
            let mut uniq: Vec<u64> = Vec::new();
            for (i, e) in entries.iter().enumerate() {
                uniq.clear();
                uniq.extend_from_slice(&e.variant_keys[a]);
                uniq.sort_unstable();
                uniq.dedup();
                let owner = owner_for_entry[i];
                for &key in &uniq {
                    idx.insert(key, owner);
                }
            }
            idx
        })
        .collect()
}

/// Save a project gate-index bundle (see module docs). Overwrites `dir`.
/// `fingerprint` is [`gate_fingerprint`] over the source entries — the
/// loader refuses bundles whose stored value doesn't match a recompute.
pub fn save_project_gate_index(
    dir: &std::path::Path,
    indexes: &[ShardedMihIndex],
    image_ids: &[i64],
    shard_bits: u32,
    fingerprint: &str,
) -> std::io::Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    let key_count: usize = indexes.iter().map(|i| i.len()).sum();
    let meta = serde_json::json!({
        "magic": PROJECT_MIH_MAGIC,
        "version": PROJECT_MIH_VERSION,
        "shard_bits": shard_bits,
        "gate_algo_count": indexes.len(),
        "image_count": image_ids.len(),
        "key_count": key_count,
        "feature_fingerprint": fingerprint,
    });
    std::fs::write(
        dir.join("meta.json"),
        serde_json::to_vec_pretty(&meta).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?,
    )?;
    write_image_ids_bin(&dir.join("image_ids.bin"), image_ids)?;
    let indexes_dir = dir.join("indexes");
    std::fs::create_dir_all(&indexes_dir)?;
    for (a, idx) in indexes.iter().enumerate() {
        idx.save_dir(&indexes_dir.join(a.to_string()))?;
    }
    Ok(())
}

/// Try to load a project bundle compatible with `entries` / `shard_bits`.
/// Returns `None` when missing or invalidated (caller should rebuild).
pub fn try_load_project_gate_index(
    dir: &std::path::Path,
    entries: &[DedupKeys],
    shard_bits: u32,
) -> std::io::Result<Option<(Vec<ShardedMihIndex>, Vec<i64>)>> {
    if entries.is_empty() || entries[0].variant_keys.is_empty() {
        return Ok(None);
    }
    let meta_path = dir.join("meta.json");
    if !meta_path.is_file() {
        return Ok(None);
    }
    let meta_bytes = std::fs::read(&meta_path)?;
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let magic = meta.get("magic").and_then(|v| v.as_str()).unwrap_or("");
    if magic != PROJECT_MIH_MAGIC {
        return Ok(None);
    }
    let version = meta.get("version").and_then(|v| v.as_u64()).unwrap_or(0);
    if version != PROJECT_MIH_VERSION {
        return Ok(None);
    }
    let stored_bits = meta
        .get("shard_bits")
        .and_then(|v| v.as_u64())
        .unwrap_or(u64::MAX) as u32;
    if stored_bits != shard_bits {
        return Ok(None);
    }
    let gate_algo_count = meta
        .get("gate_algo_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let m = entries[0].variant_keys.len();
    if gate_algo_count != m {
        return Ok(None);
    }
    let image_count = meta
        .get("image_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let image_ids = match read_image_ids_bin(&dir.join("image_ids.bin"), image_count) {
        Ok(ids) => ids,
        Err(_) => return Ok(None),
    };
    if !image_id_sets_equal(&image_ids, entries.iter().map(|e| e.image_id).collect()) {
        return Ok(None);
    }
    // Feature-blob fingerprint: catches recomputed/changed vectors that
    // leave the image_id set untouched. Bundles written before the field
    // existed fail the check and rebuild once.
    let stored_fp = meta.get("feature_fingerprint").and_then(|v| v.as_str());
    if stored_fp != Some(gate_fingerprint(entries).as_str()) {
        return Ok(None);
    }
    let indexes_dir = dir.join("indexes");
    let mut indexes = Vec::with_capacity(m);
    for a in 0..m {
        let path = indexes_dir.join(a.to_string());
        if !path.is_dir() {
            return Ok(None);
        }
        let idx = match ShardedMihIndex::load_dir(&path) {
            Ok(i) => i,
            Err(_) => return Ok(None),
        };
        if idx.shard_bits() != shard_bits {
            return Ok(None);
        }
        indexes.push(idx);
    }
    Ok(Some((indexes, image_ids)))
}

/// Load a compatible project index or build+save one. `index_loaded` is true
/// when an existing on-disk bundle was used (no rebuild).
pub fn load_or_build_project_gate_index(
    dir: &std::path::Path,
    entries: &[DedupKeys],
    shard_bits: u32,
) -> std::io::Result<(Vec<ShardedMihIndex>, Vec<i64>, bool)> {
    if let Some((indexes, image_ids)) = try_load_project_gate_index(dir, entries, shard_bits)? {
        return Ok((indexes, image_ids, true));
    }
    let (image_ids, owner_for_entry) = owner_plan(entries);
    let indexes = build_sharded_indexes(entries, shard_bits, &owner_for_entry);
    save_project_gate_index(
        dir,
        &indexes,
        &image_ids,
        shard_bits,
        &gate_fingerprint(entries),
    )?;
    Ok((indexes, image_ids, false))
}

// ---------- multi-node project bundle (`project_{id}_mn/`, ITMIHN1) ----------

/// `project_{id}_mn/` — multi-node sibling of the single-node
/// `project_{id}/` gate bundle, used when `ITRACE_MIH_NODES` > 1 so the
/// two layouts never collide (flipping the node count can't silently
/// reuse the wrong format; a nodes change rebuilds the `_mn` bundle):
///
/// ```text
/// project_{id}_mn/
///   meta.json      {"magic":"ITMIHN1","version":1,"shard_bits":N,
///                   "node_count":N,"gate_algo_count":M,"image_count":K,
///                   "feature_fingerprint":"<blake3 hex>"}
///   image_ids.bin  K × i64 LE — owner slots, sorted ascending
///   indexes/{a}/   MultiNodeMihIndex::save_dir — top-level ITMIHN1
///                  meta (shard_bits + ownership ranges) + node_N/
///                  dirs, each node_N/meta.json + shards/NNNN.bin
/// ```
fn project_mih_multi_dir(dir: &std::path::Path) -> std::path::PathBuf {
    let mut s = dir.as_os_str().to_os_string();
    s.push("_mn");
    s.into()
}

/// Per-algo `MultiNodeMihIndex` build — same owner plan and key
/// dedup as [`build_sharded_indexes`], even shard ownership split.
fn build_multi_indexes(
    entries: &[DedupKeys],
    shard_bits: u32,
    nodes: u32,
    owner_for_entry: &[u32],
) -> Vec<crate::ownership::MultiNodeMihIndex> {
    let m = entries[0].variant_keys.len();
    (0..m)
        .into_par_iter()
        .map(|a| {
            let mut idx = crate::ownership::MultiNodeMihIndex::even(shard_bits, nodes);
            let mut uniq: Vec<u64> = Vec::new();
            for (i, e) in entries.iter().enumerate() {
                uniq.clear();
                uniq.extend_from_slice(&e.variant_keys[a]);
                uniq.sort_unstable();
                uniq.dedup();
                let owner = owner_for_entry[i];
                for &key in &uniq {
                    idx.insert(key, owner);
                }
            }
            idx
        })
        .collect()
}

/// Save a `project_{id}_mn/` bundle (layout above). Overwrites `dir`.
fn save_project_gate_index_multi(
    dir: &std::path::Path,
    indexes: &[crate::ownership::MultiNodeMihIndex],
    image_ids: &[i64],
    shard_bits: u32,
    nodes: u32,
    fingerprint: &str,
) -> std::io::Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    let key_count: usize = indexes.iter().map(|i| i.len()).sum();
    let meta = serde_json::json!({
        "magic": "ITMIHN1",
        "version": 1,
        "shard_bits": shard_bits,
        "node_count": nodes,
        "gate_algo_count": indexes.len(),
        "image_count": image_ids.len(),
        "key_count": key_count,
        "feature_fingerprint": fingerprint,
    });
    std::fs::write(
        dir.join("meta.json"),
        serde_json::to_vec_pretty(&meta)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
    )?;
    write_image_ids_bin(&dir.join("image_ids.bin"), image_ids)?;
    let indexes_dir = dir.join("indexes");
    std::fs::create_dir_all(&indexes_dir)?;
    for (a, idx) in indexes.iter().enumerate() {
        idx.save_dir(&indexes_dir.join(a.to_string()))?;
    }
    Ok(())
}

/// Try to load a `project_{id}_mn/` bundle compatible with `entries`,
/// `shard_bits`, and `nodes`. Same invalidation axes as the single-node
/// [`try_load_project_gate_index`] plus `node_count`; per-algo cluster
/// dirs are loaded via `MultiNodeMihIndex::load_dir` (which itself
/// rejects bad magic, incomplete/overlapping ranges, and shard_bits
/// mismatch). Any failure → `None` (caller rebuilds).
fn try_load_project_gate_index_multi(
    dir: &std::path::Path,
    entries: &[DedupKeys],
    shard_bits: u32,
    nodes: u32,
) -> std::io::Result<Option<(Vec<crate::ownership::MultiNodeMihIndex>, Vec<i64>)>> {
    if entries.is_empty() || entries[0].variant_keys.is_empty() {
        return Ok(None);
    }
    let meta_path = dir.join("meta.json");
    if !meta_path.is_file() {
        return Ok(None);
    }
    let meta_bytes = std::fs::read(&meta_path)?;
    let meta: serde_json::Value = match serde_json::from_slice(&meta_bytes) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let m = entries[0].variant_keys.len();
    let field = |k: &str| meta.get(k).and_then(|v| v.as_u64());
    if meta.get("magic").and_then(|v| v.as_str()) != Some("ITMIHN1")
        || field("version") != Some(1)
        || field("shard_bits") != Some(shard_bits as u64)
        || field("node_count") != Some(nodes as u64)
        || field("gate_algo_count") != Some(m as u64)
    {
        return Ok(None);
    }
    let image_count = field("image_count").unwrap_or(0) as usize;
    let image_ids = match read_image_ids_bin(&dir.join("image_ids.bin"), image_count) {
        Ok(ids) => ids,
        Err(_) => return Ok(None),
    };
    if !image_id_sets_equal(&image_ids, entries.iter().map(|e| e.image_id).collect()) {
        return Ok(None);
    }
    if meta.get("feature_fingerprint").and_then(|v| v.as_str())
        != Some(gate_fingerprint(entries).as_str())
    {
        return Ok(None);
    }
    let indexes_dir = dir.join("indexes");
    let mut indexes = Vec::with_capacity(m);
    for a in 0..m {
        let path = indexes_dir.join(a.to_string());
        if !path.is_dir() {
            return Ok(None);
        }
        match crate::ownership::MultiNodeMihIndex::load_dir(&path) {
            Ok(idx) if idx.shard_bits() == shard_bits && idx.node_count() == nodes => {
                indexes.push(idx)
            }
            _ => return Ok(None),
        }
    }
    Ok(Some((indexes, image_ids)))
}

/// Load a compatible `project_{id}_mn/` bundle or build+save one.
/// `index_loaded` is true when an existing on-disk bundle was reused.
fn load_or_build_project_gate_index_multi(
    dir: &std::path::Path,
    entries: &[DedupKeys],
    shard_bits: u32,
    nodes: u32,
) -> std::io::Result<(Vec<crate::ownership::MultiNodeMihIndex>, Vec<i64>, bool)> {
    if let Some((indexes, image_ids)) =
        try_load_project_gate_index_multi(dir, entries, shard_bits, nodes)?
    {
        return Ok((indexes, image_ids, true));
    }
    let (image_ids, owner_for_entry) = owner_plan(entries);
    let indexes = build_multi_indexes(entries, shard_bits, nodes, &owner_for_entry);
    save_project_gate_index_multi(
        dir,
        &indexes,
        &image_ids,
        shard_bits,
        nodes,
        &gate_fingerprint(entries),
    )?;
    Ok((indexes, image_ids, false))
}

/// Owner-slot-keyed MIH query contract shared by [`ShardedMihIndex`]
/// (single-node) and [`crate::ownership::MultiNodeMihIndex`] (in-process
/// multi-node) — both answer `query_into(key, radius)` with owner slots.
pub trait MihQueryIndex: Sync {
    fn query_into(&self, key: u64, radius: u32, out: &mut Vec<u32>);
}

impl MihQueryIndex for ShardedMihIndex {
    fn query_into(&self, key: u64, radius: u32, out: &mut Vec<u32>) {
        ShardedMihIndex::query_into(self, key, radius, out)
    }
}

impl MihQueryIndex for crate::ownership::MultiNodeMihIndex {
    fn query_into(&self, key: u64, radius: u32, out: &mut Vec<u32>) {
        crate::ownership::MultiNodeMihIndex::query_into(self, key, radius, out)
    }
}

/// Query pre-built per-algo indexes. Owners in the indexes are dense slots
/// into `image_ids`; returned pairs are **entry indices** into `entries`.
pub fn dedup_candidates_with_indexes<I: MihQueryIndex>(
    entries: &[DedupKeys],
    indexes: &[I],
    image_ids: &[i64],
    radius: u32,
    min_votes: u32,
) -> Vec<(u32, u32)> {
    if entries.is_empty() || indexes.is_empty() {
        return Vec::new();
    }
    // owner slot → entry index in the caller's `entries` slice
    let id_to_entry: HashMap<i64, u32> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.image_id, i as u32))
        .collect();
    let entry_of_owner: Vec<Option<u32>> = image_ids
        .iter()
        .map(|id| id_to_entry.get(id).copied())
        .collect();

    let mut out: Vec<(u32, u32)> = entries
        .par_iter()
        .enumerate()
        .flat_map(|(i, e)| {
            let i = i as u32;
            let mut hit = VoteMap::default();
            let mut uniq: Vec<u64> = Vec::new();
            let mut hits: Vec<u32> = Vec::new();
            for (a, idx) in indexes.iter().enumerate() {
                uniq.clear();
                uniq.extend_from_slice(&e.variant_keys[a]);
                uniq.sort_unstable();
                uniq.dedup();
                for &key in &uniq {
                    idx.query_into(key, radius, &mut hits);
                    for &owner in &hits {
                        if let Some(Some(j)) = entry_of_owner.get(owner as usize) {
                            if *j != i {
                                *hit.entry(*j).or_insert(0) |= 1 << a;
                            }
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

/// Like [`dedup_candidates_sharded`], but when `project_index_dir` is `Some`
/// loads or builds a persistent project gate-index bundle there.
///
/// Returns `(pairs, index_loaded)` where `index_loaded` is true iff an
/// existing on-disk index was reused. When `project_index_dir` is `None`,
/// behaviour matches [`dedup_candidates_sharded`] and `index_loaded` is false.
pub fn dedup_candidates_sharded_cached(
    entries: &[DedupKeys],
    radius: u32,
    min_votes: u32,
    shard_bits: u32,
    project_index_dir: Option<&std::path::Path>,
) -> std::io::Result<(Vec<(u32, u32)>, bool)> {
    let Some(dir) = project_index_dir else {
        return Ok((
            dedup_candidates_sharded(entries, radius, min_votes, shard_bits),
            false,
        ));
    };
    if entries.is_empty() || entries[0].variant_keys.is_empty() {
        return Ok((Vec::new(), false));
    }
    let nodes = mih_node_count();
    if nodes > 1 {
        // Multi-node scan: persist under the `project_{id}_mn/` sibling
        // (ITMIHN1) so the single-node `project_{id}/` bundle stays
        // untouched — flipping `ITRACE_MIH_NODES` back to 1 never reads a
        // multi-node bundle and vice versa.
        let mn_dir = project_mih_multi_dir(dir);
        let (indexes, image_ids, loaded) =
            load_or_build_project_gate_index_multi(&mn_dir, entries, shard_bits, nodes)?;
        let pairs = dedup_candidates_with_indexes(entries, &indexes, &image_ids, radius, min_votes);
        return Ok((pairs, loaded));
    }
    let (indexes, image_ids, loaded) =
        load_or_build_project_gate_index(dir, entries, shard_bits)?;
    let pairs = dedup_candidates_with_indexes(entries, &indexes, &image_ids, radius, min_votes);
    Ok((pairs, loaded))
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

/// Dev/test knob: `ITRACE_MIH_NODES=N` (N > 1) makes the candidate scan
/// run on an in-process [`crate::ownership::MultiNodeMihIndex`] — same
/// recall contract, exercised through shard-ownership routing +
/// scatter/gather. Default/unset/invalid = 1 → plain `ShardedMihIndex`.
/// With `ITRACE_MIH_INDEX_DIR` set the multi-node scan persists under
/// `project_{id}_mn/` (ITMIHN1); N ≤ 1 keeps the single-node
/// `ITMIHP1`/`project_{id}/` format untouched.
pub fn mih_node_count() -> u32 {
    std::env::var("ITRACE_MIH_NODES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1)
        .max(1)
}

/// Gate-index used by the in-memory candidate scan: monolithic by default,
/// multi-node when `ITRACE_MIH_NODES` > 1. Both expose identical
/// insert/query semantics — `MultiNodeMihIndex` parity is property-tested
/// against `ShardedMihIndex` in `crate::ownership`.
enum GateIndex {
    Mono(ShardedMihIndex),
    Multi(crate::ownership::MultiNodeMihIndex),
}

impl GateIndex {
    fn new(shard_bits: u32, nodes: u32) -> Self {
        if nodes > 1 {
            Self::Multi(crate::ownership::MultiNodeMihIndex::even(shard_bits, nodes))
        } else {
            Self::Mono(ShardedMihIndex::new(shard_bits))
        }
    }

    fn insert(&mut self, key: u64, owner: u32) {
        match self {
            Self::Mono(i) => i.insert(key, owner),
            Self::Multi(i) => i.insert(key, owner),
        }
    }

    fn query_into(&self, key: u64, radius: u32, out: &mut Vec<u32>) {
        match self {
            Self::Mono(i) => i.query_into(key, radius, out),
            Self::Multi(i) => i.query_into(key, radius, out),
        }
    }
}

/// Same contract as [`dedup_candidates`] but each per-algo index is a
/// [`ShardedMihIndex`]. Server `/dedup` and CLI `dedup` use this path
/// (via `resolve_shard_bits` / `dedup_confirmed`).
pub fn dedup_candidates_sharded(
    entries: &[DedupKeys],
    radius: u32,
    min_votes: u32,
    shard_bits: u32,
) -> Vec<(u32, u32)> {
    if entries.is_empty() || entries[0].variant_keys.is_empty() {
        return Vec::new();
    }
    let m = entries[0].variant_keys.len();
    let nodes = mih_node_count();
    let indexes: Vec<GateIndex> = (0..m)
        .into_par_iter()
        .map(|a| {
            let mut idx = GateIndex::new(shard_bits, nodes);
            let mut uniq: Vec<u64> = Vec::new();
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
            let mut hit = VoteMap::default();
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

fn confirm_pairs(
    entries: &[DedupKeys],
    pairs: &[(u32, u32)],
    threshold: f64,
) -> Vec<(usize, usize, f64)> {
    let score_of = |ea: usize, eb: usize| -> f64 {
        let mut best = 0.0f64;
        for (ka, kb) in entries[ea].variant_keys.iter().zip(&entries[eb].variant_keys) {
            for &a in ka {
                for &b in kb {
                    best = best.max(crate::hashes::hash_similarity(a, b));
                }
            }
        }
        best
    };
    let mut confirmed: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let (i, j) = (i as usize, j as usize);
            let s = score_of(i, j);
            if s >= threshold {
                Some((i, j, s))
            } else {
                None
            }
        })
        .collect();
    confirmed.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    confirmed
}

/// Confirmed duplicate pairs: `(entry_i, entry_j, score)` with `i < j`
/// indexing into the caller's `entries` slice.
pub type ConfirmedPairs = Vec<(usize, usize, f64)>;

/// MIH candidate recall then cross-variant max score verification.
/// Returns `(candidate_pair_count, confirmed)`.
///
/// Uses [`dedup_candidates_sharded`] with `shard_bits` (`0` = one shard).
pub fn dedup_confirmed(
    entries: &[DedupKeys],
    radius: u32,
    threshold: f64,
    min_votes: u32,
    shard_bits: u32,
) -> (usize, ConfirmedPairs) {
    let pairs = dedup_candidates_sharded(entries, radius, min_votes, shard_bits);
    let confirmed = confirm_pairs(entries, &pairs, threshold);
    (pairs.len(), confirmed)
}

/// Like [`dedup_confirmed`] with optional persistent project index directory.
/// Third return value is `index_loaded` (see [`dedup_candidates_sharded_cached`]).
pub fn dedup_confirmed_cached(
    entries: &[DedupKeys],
    radius: u32,
    threshold: f64,
    min_votes: u32,
    shard_bits: u32,
    project_index_dir: Option<&std::path::Path>,
) -> std::io::Result<(usize, ConfirmedPairs, bool)> {
    dedup_confirmed_cached_with_extra(
        entries,
        radius,
        threshold,
        min_votes,
        shard_bits,
        project_index_dir,
        &[],
    )
}

/// Like [`dedup_confirmed_cached`], but unions caller-supplied `extra`
/// candidate pairs (e.g. semantic-channel hits from
/// [`crate::semantic::semantic_candidates`]) into the MIH recall set
/// **before** [`confirm_pairs`] verification. `extra` pairs are entry
/// indices — normalized and deduplicated by
/// [`crate::semantic::union_candidate_pairs`]. Verification is unchanged:
/// an extra-recalled pair still needs the cross-variant max score ≥
/// `threshold`, so extra recall can never widen confirmed merges.
/// `extra = &[]` is bit-identical to [`dedup_confirmed_cached`].
pub fn dedup_confirmed_cached_with_extra(
    entries: &[DedupKeys],
    radius: u32,
    threshold: f64,
    min_votes: u32,
    shard_bits: u32,
    project_index_dir: Option<&std::path::Path>,
    extra: &[(u32, u32)],
) -> std::io::Result<(usize, ConfirmedPairs, bool)> {
    let (mut pairs, loaded) =
        dedup_candidates_sharded_cached(entries, radius, min_votes, shard_bits, project_index_dir)?;
    crate::semantic::union_candidate_pairs(&mut pairs, extra.iter().copied());
    let confirmed = confirm_pairs(entries, &pairs, threshold);
    Ok((pairs.len(), confirmed, loaded))
}

// ---------- crop/slice recall channel ----------
//
// The gate channel indexes whole-image hashes: a crop or slice tile moves
// too many bits to ever be recalled. The crop channel indexes the
// `crophash_keys` windowed-phash payloads instead — every image contributes
// all of its window keys (slots × orientation variants, deduplicated) to a
// single sharded index, and a pair is emitted when at least `min_hits` of
// the query image's distinct probe keys return the other owner. Candidates
// are verified downstream by NCC containment (`slice::contains_rot4`), not
// by the hash keys themselves.

/// One image's flattened crop-recall key set (window slots × variants).
#[derive(Clone)]
pub struct CropKeys {
    pub image_id: i64,
    pub keys: Vec<u64>,
}

/// Default minimum number of distinct probe-key hits needed to emit a
/// crop-channel candidate pair. Env override: `ITRACE_CROP_MIN_HITS`.
pub const DEFAULT_CROP_MIN_HITS: u32 = 2;

/// Resolve the crop-channel minimum distinct-key hits.
/// Precedence: `override_hits` > env `ITRACE_CROP_MIN_HITS` >
/// [`DEFAULT_CROP_MIN_HITS`]; clamped to ≥1.
pub fn resolve_crop_min_hits(override_hits: Option<u32>) -> u32 {
    let raw = override_hits.or_else(|| {
        std::env::var("ITRACE_CROP_MIN_HITS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
    });
    raw.unwrap_or(DEFAULT_CROP_MIN_HITS).max(1)
}

/// `{base}/project_{id}_crop` — the crop channel's own bundle root, kept
/// separate from the gate bundle so each invalidates independently.
pub fn project_crop_mih_index_path(
    base: &std::path::Path,
    project_id: i64,
) -> std::path::PathBuf {
    base.join(format!("project_{project_id}_crop"))
}

/// `{base}/project_{id}_crop_mn` — multi-node crop bundle root
/// (`ITMIHCN1`), sibling of `project_{id}_crop/` so flipping
/// `ITRACE_MIH_NODES` between 1 and N > 1 can never read a bundle of the
/// wrong shape.
///
/// Layout:
///
/// ```text
/// project_{id}_crop_mn/
///   meta.json      {"magic":"ITMIHCN1","version":1,"shard_bits":N,
///                   "node_count":M,"image_count":K,"key_count":T,
///                   "feature_fingerprint":"<blake3 hex>",
///                   "ranges":[{"start":id,"end":id,"count":c}]}
///   node_{i}/      complete ITMIHC1 bundle over node i's contiguous
///                  sorted-image-id range (meta.json + image_ids.bin +
///                  index/ shards)
/// ```
pub fn project_crop_mih_index_multi_path(
    base: &std::path::Path,
    project_id: i64,
) -> std::path::PathBuf {
    base.join(format!("project_{project_id}_crop_mn"))
}

/// Candidate pairs from a pre-built crop index. `hits` counts an owner at
/// most once per probe key (owners are deduped inside `query_into`), so
/// `min_hits` counts independent key matches, not repeats of one key.
pub fn crop_candidates_with_index(
    entries: &[CropKeys],
    idx: &ShardedMihIndex,
    image_ids: &[i64],
    radius: u32,
    min_hits: u32,
) -> Vec<(u32, u32)> {
    if entries.is_empty() {
        return Vec::new();
    }
    // owner slot → entry index in the caller's `entries` slice
    let id_to_entry: HashMap<i64, u32> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.image_id, i as u32))
        .collect();
    let entry_of_owner: Vec<Option<u32>> = image_ids
        .iter()
        .map(|id| id_to_entry.get(id).copied())
        .collect();

    let mut out: Vec<(u32, u32)> = entries
        .par_iter()
        .enumerate()
        .flat_map(|(i, e)| {
            let i = i as u32;
            let mut hit_count: HashMap<u32, u32, std::hash::BuildHasherDefault<VoteHasher>> =
                HashMap::default();
            let mut uniq: Vec<u64> = e.keys.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let mut hits: Vec<u32> = Vec::new();
            for &key in &uniq {
                idx.query_into(key, radius, &mut hits);
                for &owner in &hits {
                    if let Some(Some(j)) = entry_of_owner.get(owner as usize) {
                        if *j != i {
                            *hit_count.entry(*j).or_insert(0) += 1;
                        }
                    }
                }
            }
            hit_count
                .into_iter()
                .filter(|(_, c)| *c >= min_hits)
                .map(|(j, _)| (j.min(i), j.max(i)))
                .collect::<Vec<_>>()
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// In-memory sibling of [`crop_candidates_with_index`]: builds a sharded
/// index over `entries` and queries it.
pub fn crop_candidates(
    entries: &[CropKeys],
    radius: u32,
    min_hits: u32,
    shard_bits: u32,
) -> Vec<(u32, u32)> {
    if entries.is_empty() {
        return Vec::new();
    }
    let (image_ids, owner_for_entry) = owner_plan_crop(entries);
    let idx = build_crop_index(entries, shard_bits, &owner_for_entry);
    crop_candidates_with_index(entries, &idx, &image_ids, radius, min_hits)
}

fn build_crop_index(
    entries: &[CropKeys],
    shard_bits: u32,
    owner_for_entry: &[u32],
) -> ShardedMihIndex {
    let mut idx = ShardedMihIndex::new(shard_bits);
    let mut uniq: Vec<u64> = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        uniq.clear();
        uniq.extend_from_slice(&e.keys);
        uniq.sort_unstable();
        uniq.dedup();
        let owner = owner_for_entry[i];
        for &key in &uniq {
            idx.insert(key, owner);
        }
    }
    idx
}

const CROP_MIH_MAGIC: &str = "ITMIHC1";

fn save_crop_index(
    dir: &std::path::Path,
    idx: &ShardedMihIndex,
    image_ids: &[i64],
    shard_bits: u32,
    fingerprint: &str,
) -> std::io::Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    let meta = serde_json::json!({
        "magic": CROP_MIH_MAGIC,
        "version": PROJECT_MIH_VERSION,
        "shard_bits": shard_bits,
        "image_count": image_ids.len(),
        "key_count": idx.len(),
        "feature_fingerprint": fingerprint,
    });
    std::fs::write(
        dir.join("meta.json"),
        serde_json::to_vec_pretty(&meta).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?,
    )?;
    write_image_ids_bin(&dir.join("image_ids.bin"), image_ids)?;
    idx.save_dir(&dir.join("index"))
}

fn try_load_crop_index(
    dir: &std::path::Path,
    entries: &[CropKeys],
    shard_bits: u32,
) -> std::io::Result<Option<(ShardedMihIndex, Vec<i64>)>> {
    let meta_path = dir.join("meta.json");
    if entries.is_empty() || !meta_path.is_file() {
        return Ok(None);
    }
    let meta_bytes = std::fs::read(&meta_path)?;
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if meta.get("magic").and_then(|v| v.as_str()) != Some(CROP_MIH_MAGIC)
        || meta.get("version").and_then(|v| v.as_u64()) != Some(PROJECT_MIH_VERSION)
        || meta.get("shard_bits").and_then(|v| v.as_u64()) != Some(shard_bits as u64)
    {
        return Ok(None);
    }
    let image_count = meta
        .get("image_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let image_ids = match read_image_ids_bin(&dir.join("image_ids.bin"), image_count) {
        Ok(ids) => ids,
        Err(_) => return Ok(None),
    };
    if !image_id_sets_equal(&image_ids, entries.iter().map(|e| e.image_id).collect()) {
        return Ok(None);
    }
    let stored_fp = meta.get("feature_fingerprint").and_then(|v| v.as_str());
    if stored_fp != Some(crop_fingerprint(entries).as_str()) {
        return Ok(None);
    }
    let idx = match ShardedMihIndex::load_dir(&dir.join("index")) {
        Ok(i) => i,
        Err(_) => return Ok(None),
    };
    if idx.shard_bits() != shard_bits {
        return Ok(None);
    }
    Ok(Some((idx, image_ids)))
}

/// Load a compatible crop bundle or build+save one (see
/// [`load_or_build_project_gate_index`] — same contract, crop layout).
pub fn load_or_build_crop_index(
    dir: &std::path::Path,
    entries: &[CropKeys],
    shard_bits: u32,
) -> std::io::Result<(ShardedMihIndex, Vec<i64>, bool)> {
    if let Some((idx, image_ids)) = try_load_crop_index(dir, entries, shard_bits)? {
        return Ok((idx, image_ids, true));
    }
    let (image_ids, owner_for_entry) = owner_plan_crop(entries);
    let idx = build_crop_index(entries, shard_bits, &owner_for_entry);
    save_crop_index(dir, &idx, &image_ids, shard_bits, &crop_fingerprint(entries))?;
    Ok((idx, image_ids, false))
}

/// Like [`crop_candidates`] but load-or-builds a persistent bundle under
/// `dir` when given (`{ITRACE_MIH_INDEX_DIR}/project_{id}_crop/`).
/// Returns `(pairs, index_loaded)`.
///
/// With `ITRACE_MIH_NODES` > 1 the bundle moves to the
/// `project_{id}_crop_mn/` sibling (`ITMIHCN1`, derived by appending
/// `_mn` to `dir`'s name) — same ownership plan as the semantic
/// `_sem_mn` channel: contiguous sorted-image-id ranges, one complete
/// `ITMIHC1` node bundle each. The single-node `ITMIHC1` path is
/// untouched.
pub fn crop_candidates_cached(
    entries: &[CropKeys],
    radius: u32,
    min_hits: u32,
    shard_bits: u32,
    dir: Option<&std::path::Path>,
) -> std::io::Result<(Vec<(u32, u32)>, bool)> {
    let Some(dir) = dir else {
        return Ok((crop_candidates(entries, radius, min_hits, shard_bits), false));
    };
    if entries.is_empty() {
        return Ok((Vec::new(), false));
    }
    let nodes = mih_node_count();
    if nodes > 1 {
        let mn_dir = project_mih_multi_dir(dir);
        let (node_indexes, loaded) =
            load_or_build_crop_index_multi(&mn_dir, entries, nodes, shard_bits)?;
        return Ok((
            crop_candidates_multi(entries, &node_indexes, radius, min_hits),
            loaded,
        ));
    }
    let (idx, image_ids, loaded) = load_or_build_crop_index(dir, entries, shard_bits)?;
    Ok((
        crop_candidates_with_index(entries, &idx, &image_ids, radius, min_hits),
        loaded,
    ))
}

const CROP_MIH_MN_MAGIC: &str = "ITMIHCN1";

/// One node of a `project_{id}_crop_mn/` bundle: a complete
/// [`ShardedMihIndex`] over the node's owned images. `image_ids` maps
/// node-local owner slots → global image ids (sorted ascending).
pub struct NodeCropIndex {
    pub index: ShardedMihIndex,
    pub image_ids: Vec<i64>,
}

/// Partition `entries` into `nodes` contiguous non-empty chunks over the
/// sorted-image-id space — the same ownership plan as the semantic
/// `_sem_mn` channel (`semantic::sem_partition` on sorted ids).
fn partition_crop_nodes(entries: &[CropKeys], nodes: u32) -> Vec<Vec<CropKeys>> {
    let (sorted_ids, _) = owner_plan_crop(entries);
    let by_id: HashMap<i64, &CropKeys> = entries.iter().map(|e| (e.image_id, e)).collect();
    let n = (nodes as usize).min(sorted_ids.len()).max(1);
    (0..n)
        .map(|i| {
            sorted_ids[i * sorted_ids.len() / n..(i + 1) * sorted_ids.len() / n]
                .iter()
                .map(|id| (*by_id[id]).clone())
                .collect()
        })
        .collect()
}

/// Save a `project_{id}_crop_mn/` bundle (layout above). Overwrites
/// `dir`. `node_entries[i]` is node i's owned subset (sorted order);
/// `fingerprint` is the global [`crop_fingerprint`] over all entries.
fn save_crop_index_multi(
    dir: &std::path::Path,
    node_entries: &[Vec<CropKeys>],
    nodes: u32,
    shard_bits: u32,
    fingerprint: &str,
) -> std::io::Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    let mut key_count = 0usize;
    for (i, chunk) in node_entries.iter().enumerate() {
        let (nids, owner_for_entry) = owner_plan_crop(chunk);
        let idx = build_crop_index(chunk, shard_bits, &owner_for_entry);
        key_count += idx.len();
        save_crop_index(
            &dir.join(format!("node_{i}")),
            &idx,
            &nids,
            shard_bits,
            &crop_fingerprint(chunk),
        )?;
    }
    let ranges: Vec<serde_json::Value> = node_entries
        .iter()
        .map(|chunk| {
            serde_json::json!({
                "start": chunk.first().map(|e| e.image_id),
                "end": chunk.last().map(|e| e.image_id),
                "count": chunk.len(),
            })
        })
        .collect();
    let meta = serde_json::json!({
        "magic": CROP_MIH_MN_MAGIC,
        "version": 1,
        "shard_bits": shard_bits,
        "node_count": nodes,
        "image_count": node_entries.iter().map(|c| c.len()).sum::<usize>(),
        "key_count": key_count,
        "feature_fingerprint": fingerprint,
        "ranges": ranges,
    });
    std::fs::write(
        dir.join("meta.json"),
        serde_json::to_vec_pretty(&meta)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
    )
}

/// Try to load a `project_{id}_crop_mn/` bundle compatible with
/// `entries`, `nodes`, and `shard_bits`. Validates the top meta
/// (magic/version/shard_bits/node_count/image_count/global fingerprint
/// and the recomputed partition ranges), then runs each `node_{i}/`
/// through the single-node [`try_load_crop_index`] against its expected
/// chunk. Any miss → `None`; caller rebuilds.
fn try_load_crop_index_multi(
    dir: &std::path::Path,
    entries: &[CropKeys],
    nodes: u32,
    shard_bits: u32,
) -> std::io::Result<Option<Vec<NodeCropIndex>>> {
    let meta_path = dir.join("meta.json");
    if entries.is_empty() || !meta_path.is_file() {
        return Ok(None);
    }
    let meta_bytes = std::fs::read(&meta_path)?;
    let meta: serde_json::Value = match serde_json::from_slice(&meta_bytes) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let field = |k: &str| meta.get(k).and_then(|v| v.as_u64());
    if meta.get("magic").and_then(|v| v.as_str()) != Some(CROP_MIH_MN_MAGIC)
        || field("version") != Some(1)
        || field("shard_bits") != Some(shard_bits as u64)
        || field("node_count") != Some(nodes as u64)
        || field("image_count") != Some(entries.len() as u64)
        || meta.get("feature_fingerprint").and_then(|v| v.as_str())
            != Some(crop_fingerprint(entries).as_str())
    {
        return Ok(None);
    }
    let node_entries = partition_crop_nodes(entries, nodes);
    let ranges = meta.get("ranges").and_then(|v| v.as_array());
    if ranges.map(|r| r.len()) != Some(node_entries.len()) {
        return Ok(None);
    }
    let ranges = ranges.unwrap();
    let mut out = Vec::with_capacity(node_entries.len());
    for (i, chunk) in node_entries.iter().enumerate() {
        let r = &ranges[i];
        if r.get("start").and_then(|v| v.as_i64()) != chunk.first().map(|e| e.image_id)
            || r.get("end").and_then(|v| v.as_i64()) != chunk.last().map(|e| e.image_id)
            || r.get("count").and_then(|v| v.as_u64()) != Some(chunk.len() as u64)
        {
            return Ok(None);
        }
        match try_load_crop_index(&dir.join(format!("node_{i}")), chunk, shard_bits)? {
            Some((index, image_ids)) => out.push(NodeCropIndex { index, image_ids }),
            None => return Ok(None),
        }
    }
    Ok(Some(out))
}

/// Load a compatible `project_{id}_crop_mn/` bundle or build+save one.
/// `loaded` is true only on a full hit (top meta + every node bundle).
fn load_or_build_crop_index_multi(
    dir: &std::path::Path,
    entries: &[CropKeys],
    nodes: u32,
    shard_bits: u32,
) -> std::io::Result<(Vec<NodeCropIndex>, bool)> {
    if let Some(indexes) = try_load_crop_index_multi(dir, entries, nodes, shard_bits)? {
        return Ok((indexes, true));
    }
    let node_entries = partition_crop_nodes(entries, nodes);
    save_crop_index_multi(
        dir,
        &node_entries,
        nodes,
        shard_bits,
        &crop_fingerprint(entries),
    )?;
    let indexes = node_entries
        .iter()
        .map(|chunk| {
            let (image_ids, owner_for_entry) = owner_plan_crop(chunk);
            NodeCropIndex {
                index: build_crop_index(chunk, shard_bits, &owner_for_entry),
                image_ids,
            }
        })
        .collect();
    Ok((indexes, false))
}

/// Multi-node sibling of [`crop_candidates_with_index`]: every probe key
/// scatters to all node indexes, node-local owner slots map back through
/// each node's `image_ids`, votes accumulate exactly as in the
/// single-node path. Each image's keys live in exactly one node index,
/// so the merged hit set is identical to a global index — the candidate
/// contract is exact parity, not a superset.
pub fn crop_candidates_multi(
    entries: &[CropKeys],
    node_indexes: &[NodeCropIndex],
    radius: u32,
    min_hits: u32,
) -> Vec<(u32, u32)> {
    if entries.is_empty() || node_indexes.is_empty() {
        return Vec::new();
    }
    let id_to_entry: HashMap<i64, u32> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.image_id, i as u32))
        .collect();
    // Per node: node-local owner slot → entry index in `entries`.
    let entry_maps: Vec<Vec<Option<u32>>> = node_indexes
        .iter()
        .map(|n| {
            n.image_ids
                .iter()
                .map(|id| id_to_entry.get(id).copied())
                .collect()
        })
        .collect();

    let mut out: Vec<(u32, u32)> = entries
        .par_iter()
        .enumerate()
        .flat_map(|(i, e)| {
            let i = i as u32;
            let mut hit_count: HashMap<u32, u32, std::hash::BuildHasherDefault<VoteHasher>> =
                HashMap::default();
            let mut uniq: Vec<u64> = e.keys.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let mut hits: Vec<u32> = Vec::new();
            for (n, entry_map) in node_indexes.iter().zip(entry_maps.iter()) {
                for &key in &uniq {
                    n.index.query_into(key, radius, &mut hits);
                    for &owner in &hits {
                        if let Some(Some(j)) = entry_map.get(owner as usize) {
                            if *j != i {
                                *hit_count.entry(*j).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
            hit_count
                .into_iter()
                .filter(|(_, c)| *c >= min_hits)
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

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[test]
    fn shard_routing_stable_high_bits() {
        assert_eq!(shard_id_for(0, 0), 0);
        assert_eq!(shard_id_for(u64::MAX, 0), 0);
        // top 8 bits = 0xAB → shard 0xAB
        let key = 0xAB00_0000_0000_0001u64;
        assert_eq!(shard_id_for(key, 8), 0xAB);
        assert_eq!(shard_id_for(key, 4), 0xA);
        // flipping only low bits keeps the same shard
        assert_eq!(shard_id_for(key ^ 0xFFFF, 8), 0xAB);
        // flipping a high bit moves shard
        assert_eq!(shard_id_for(key ^ (1u64 << 63), 8), 0xAB ^ 0x80);
    }

    #[test]
    fn sharded_query_parity_with_monolithic() {
        let mut rng = 0xC0FFEE_u64;
        let n = 2_000usize;
        let mut keys = Vec::with_capacity(n);
        let mut owners = Vec::with_capacity(n);
        for i in 0..n {
            keys.push(xorshift(&mut rng));
            owners.push((i % 500) as u32);
        }
        let mut mono = MihIndex::new();
        let mut sharded = ShardedMihIndex::new(6); // 64 shards
        for (&k, &o) in keys.iter().zip(&owners) {
            mono.insert(k, o);
            sharded.insert(k, o);
        }
        assert_eq!(mono.len(), sharded.len());
        assert_eq!(sharded.shard_count(), 64);
        // Probe a mix of stored keys and fresh random queries.
        let mut probes = keys.clone();
        for _ in 0..200 {
            probes.push(xorshift(&mut rng));
        }
        for &radius in &[0u32, 7, 15] {
            for &q in &probes {
                let a = mono.query(q, radius);
                let b = sharded.query(q, radius);
                assert_eq!(
                    a, b,
                    "owner set mismatch radius={radius} query={q:#x}"
                );
            }
        }
    }

    #[test]
    fn sharded_save_load_roundtrip() {
        let mut rng = 0xDEAD_BEEF_u64;
        let mut idx = ShardedMihIndex::new(4);
        let mut keys = Vec::new();
        for i in 0..800u32 {
            let k = xorshift(&mut rng);
            keys.push(k);
            idx.insert(k, i);
        }
        let dir = std::env::temp_dir().join(format!(
            "itrace-mih-roundtrip-{}-{}",
            std::process::id(),
            xorshift(&mut rng)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        idx.save_dir(&dir).expect("save");
        let loaded = ShardedMihIndex::load_dir(&dir).expect("load");
        assert_eq!(loaded.shard_bits(), 4);
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(loaded.shard_count(), idx.shard_count());
        for &radius in &[0u32, 7, 15] {
            for &q in keys.iter().step_by(7) {
                assert_eq!(
                    idx.query(q, radius),
                    loaded.query(q, radius),
                    "roundtrip mismatch radius={radius}"
                );
            }
            // fresh queries too
            for _ in 0..30 {
                let q = xorshift(&mut rng);
                assert_eq!(idx.query(q, radius), loaded.query(q, radius));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dedup_candidates_sharded_parity() {
        let mut rng = 0x1234_5678_9ABCu64;
        let mut entries = Vec::new();
        for id in 0..40i64 {
            let mut variant_keys = Vec::new();
            for _algo in 0..3 {
                let base = xorshift(&mut rng);
                // 8 near-duplicate variant keys
                let keys: Vec<u64> = (0..8)
                    .map(|v| base ^ ((v as u64) * 0x11))
                    .collect();
                variant_keys.push(keys);
            }
            entries.push(DedupKeys {
                image_id: id,
                variant_keys,
            });
        }
        // Inject a near-dup of entry 0 into entry 1 under two algos.
        for a in 0..2 {
            entries[1].variant_keys[a] = entries[0].variant_keys[a]
                .iter()
                .map(|&k| k ^ 0x3)
                .collect();
        }
        let a = dedup_candidates(&entries, 7, 2);
        let b = dedup_candidates_sharded(&entries, 7, 2, 5);
        assert_eq!(a, b);
        let (_n, conf0) = dedup_confirmed(&entries, 7, 0.5, 2, 0);
        let (_n, conf5) = dedup_confirmed(&entries, 7, 0.5, 2, 5);
        assert_eq!(conf0, conf5);
    }

    /// Tests mutating `ITRACE_MIH_NODES` race in the default parallel
    /// test harness; serialize them behind one lock.
    static NODES_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// `ITRACE_MIH_NODES>1` routes the in-memory scan through
    /// `MultiNodeMihIndex`; candidate pairs must be identical to the
    /// single-node path.
    #[test]
    fn dedup_candidates_sharded_multi_node_env_parity() {
        let _guard = NODES_ENV_LOCK.lock().unwrap();
        let entries = sample_entries(40);
        let want = dedup_candidates_sharded(&entries, 7, 2, 5); // env unset → mono
        std::env::set_var("ITRACE_MIH_NODES", "4");
        let got = dedup_candidates_sharded(&entries, 7, 2, 5);
        std::env::remove_var("ITRACE_MIH_NODES");
        assert_eq!(want, got);
        // explicit 1 = mono; invalid values fall back to mono too
        std::env::set_var("ITRACE_MIH_NODES", "bogus");
        assert_eq!(want, dedup_candidates_sharded(&entries, 7, 2, 5));
        std::env::set_var("ITRACE_MIH_NODES", "1");
        assert_eq!(want, dedup_candidates_sharded(&entries, 7, 2, 5));
        std::env::remove_var("ITRACE_MIH_NODES");
    }

    #[test]
    fn resolve_shard_bits_clamps_and_defaults() {
        // Clear may race in parallel tests; only assert clamp/override path.
        assert_eq!(resolve_shard_bits(Some(0)), 0);
        assert_eq!(resolve_shard_bits(Some(8)), 8);
        assert_eq!(resolve_shard_bits(Some(16)), 16);
        assert_eq!(resolve_shard_bits(Some(99)), 16);
        // Default when no override and env unset-or-ignored by override: covered
        // by Some(_) paths; bare default is 8.
        let d = resolve_shard_bits(None);
        assert!(d <= 16, "resolved shard_bits {d} out of range");
    }

    fn sample_entries(n: i64) -> Vec<DedupKeys> {
        let mut rng = 0xA5A5_5A5A_u64;
        let mut entries = Vec::new();
        for id in 0..n {
            let mut variant_keys = Vec::new();
            for _algo in 0..3 {
                let base = xorshift(&mut rng);
                let keys: Vec<u64> = (0..8).map(|v| base ^ ((v as u64) * 0x11)).collect();
                variant_keys.push(keys);
            }
            entries.push(DedupKeys {
                image_id: id + 100, // non-dense ids
                variant_keys,
            });
        }
        // near-dup of 0 into 1 under two algos (when we have ≥2 entries)
        if entries.len() >= 2 {
            for a in 0..2 {
                entries[1].variant_keys[a] = entries[0].variant_keys[a]
                    .iter()
                    .map(|&k| k ^ 0x3)
                    .collect();
            }
        }
        entries
    }

    /// The semantic-channel union seam: an extra pair MIH recall missed
    /// (match under only 1 of 3 gate algos < min_votes=2) is folded into
    /// the candidate set BEFORE verification, so it confirms on its
    /// variant-max hash score exactly like an MIH hit — and a pair the
    /// verifier rejects still cannot confirm.
    #[test]
    fn dedup_confirmed_cached_with_extra_unions_before_verify() {
        let mut entries = sample_entries(40);
        // entry 2 = near-dup of entry 0 under algo 0 only → MIH (min_votes
        // 2) never emits (0, 2) even though its keys nearly match.
        entries[2].variant_keys[0] = entries[0].variant_keys[0]
            .iter()
            .map(|&k| k ^ 0x3)
            .collect();
        let (base_n, base_conf, base_loaded) =
            dedup_confirmed_cached_with_extra(&entries, 7, 0.9, 2, 5, None, &[]).unwrap();
        assert!(!base_loaded);
        assert!(!base_conf.iter().any(|&(i, j, _)| (i, j) == (0, 2)));

        let (union_n, conf, _) =
            dedup_confirmed_cached_with_extra(&entries, 7, 0.9, 2, 5, None, &[(2, 0)]).unwrap();
        assert!(union_n > base_n);
        assert!(
            conf.iter().any(|&(i, j, _)| (i, j) == (0, 2)),
            "extra candidate not confirmed: {conf:?}"
        );
        for &(i, j, _) in &base_conf {
            assert!(conf.iter().any(|&(a, b, _)| (a, b) == (i, j)));
        }

        // A semantically-recalled pair that fails verification stays out.
        let (_, conf_reject, _) =
            dedup_confirmed_cached_with_extra(&entries, 7, 0.9, 2, 5, None, &[(0, 5)]).unwrap();
        assert!(!conf_reject.iter().any(|&(i, j, _)| (i, j) == (0, 5)));
    }

    /// Flag-off contract: `extra = &[]` is bit-identical to the pre-R2
    /// `dedup_confirmed_cached` entry point (default smoke unchanged).
    #[test]
    fn dedup_confirmed_cached_with_extra_empty_parity() {
        let entries = sample_entries(40);
        let a = dedup_confirmed_cached(&entries, 7, 0.5, 2, 5, None).unwrap();
        let b = dedup_confirmed_cached_with_extra(&entries, 7, 0.5, 2, 5, None, &[]).unwrap();
        assert_eq!((a.0, a.2), (b.0, b.2));
        assert_eq!(a.1.len(), b.1.len());
        for (x, y) in a.1.iter().zip(&b.1) {
            assert_eq!((x.0, x.1, x.2.to_bits()), (y.0, y.1, y.2.to_bits()));
        }
    }

    #[test]
    fn project_gate_index_save_load_same_candidates() {
        // Lock + force-unset so the single-node layout assertions can't
        // race a multi-node env mutation in a parallel test.
        let _guard = NODES_ENV_LOCK.lock().unwrap();
        std::env::remove_var("ITRACE_MIH_NODES");
        let entries = sample_entries(30);
        let shard_bits = 4u32;
        let dir = std::env::temp_dir().join(format!(
            "itrace-mih-proj-{}-{}",
            std::process::id(),
            0x1111
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let expected = dedup_candidates_sharded(&entries, 7, 2, shard_bits);

        let (pairs1, loaded1) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
                .expect("build");
        assert!(!loaded1, "first call should build");
        assert_eq!(pairs1, expected);
        assert!(dir.join("meta.json").is_file());
        assert!(dir.join("image_ids.bin").is_file());
        assert!(dir.join("indexes/0").is_dir());

        let (pairs2, loaded2) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
                .expect("load");
        assert!(loaded2, "second call should load");
        assert_eq!(pairs2, expected);
        assert_eq!(pairs2, pairs1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn project_gate_index_invalidates_on_image_set_change() {
        let _guard = NODES_ENV_LOCK.lock().unwrap();
        std::env::remove_var("ITRACE_MIH_NODES");
        let mut entries = sample_entries(20);
        let shard_bits = 3u32;
        let dir = std::env::temp_dir().join(format!(
            "itrace-mih-inval-{}-{}",
            std::process::id(),
            0x2222
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let (_, loaded) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
                .expect("build");
        assert!(!loaded);

        // Add an image → set mismatch → rebuild (index_loaded=false)
        let mut extra = sample_entries(1);
        extra[0].image_id = 9999;
        entries.push(extra.remove(0));
        let (pairs_new, loaded_rebuild) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
                .expect("rebuild");
        assert!(!loaded_rebuild, "changed image set must invalidate");
        let expected = dedup_candidates_sharded(&entries, 7, 2, shard_bits);
        assert_eq!(pairs_new, expected);

        // Same set again → load
        let (_, loaded_ok) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
                .expect("reload");
        assert!(loaded_ok);

        // shard_bits change → invalidate
        let (_, loaded_bits) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits + 1, Some(&dir))
                .expect("bits");
        assert!(!loaded_bits, "shard_bits mismatch must invalidate");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Feature-blob change with an unchanged image_id set must force a
    /// rebuild — this is the fingerprint check, the invalidation gap the
    /// old meta could not see.
    #[test]
    fn project_gate_index_invalidates_on_feature_change() {
        let _guard = NODES_ENV_LOCK.lock().unwrap();
        std::env::remove_var("ITRACE_MIH_NODES");
        let mut entries = sample_entries(20);
        let shard_bits = 3u32;
        let dir = std::env::temp_dir().join(format!(
            "itrace-mih-fp-{}-{}",
            std::process::id(),
            0x3333
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let (_, loaded) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
                .expect("build");
        assert!(!loaded);
        let (_, loaded2) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
                .expect("load");
        assert!(loaded2, "unchanged features must reuse the index");

        // Same image_ids, one mutated variant key → fingerprint mismatch
        entries[3].variant_keys[0][0] ^= 0x1;
        let (_, loaded3) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
                .expect("rebuild");
        assert!(!loaded3, "mutated feature vector must invalidate");
        let expected = dedup_candidates_sharded(&entries, 7, 2, shard_bits);
        let (pairs3, _) =
            dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
                .expect("reload after rebuild");
        assert_eq!(pairs3, expected);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `ITRACE_MIH_NODES>1` + `ITRACE_MIH_INDEX_DIR`: the multi-node scan
    /// persists under `project_{id}_mn/` (ITMIHN1), leaves the
    /// single-node `project_{id}/` slot untouched, and a second scan
    /// loads the cluster bundles with identical candidate pairs.
    #[test]
    fn project_gate_index_multi_node_roundtrip() {
        let entries = sample_entries(30);
        let shard_bits = 4u32;
        let dir =
            std::env::temp_dir().join(format!("itrace-mih-mn-{}-{}", std::process::id(), 0x4444));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(project_mih_multi_dir(&dir));

        let _guard = NODES_ENV_LOCK.lock().unwrap();
        std::env::set_var("ITRACE_MIH_NODES", "4");
        let r1 = dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir));
        let r2 = dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir));
        std::env::remove_var("ITRACE_MIH_NODES");
        let (pairs1, loaded1) = r1.expect("build");
        let (pairs2, loaded2) = r2.expect("load");
        assert!(!loaded1, "first multi-node call should build");
        assert!(loaded2, "second multi-node call should load");
        assert_eq!(pairs1, pairs2);

        // The _mn sibling got the ITMIHN1 cluster layout; the plain
        // project_{id}/ dir must NOT have been written.
        let mn = project_mih_multi_dir(&dir);
        assert!(mn.join("meta.json").is_file());
        assert!(mn.join("image_ids.bin").is_file());
        assert!(mn.join("indexes/0/meta.json").is_file());
        assert!(mn.join("indexes/0/node_0").is_dir());
        let top_meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(mn.join("meta.json")).unwrap()).unwrap();
        assert_eq!(top_meta["magic"], "ITMIHN1");
        assert_eq!(top_meta["node_count"], 4);
        assert!(!dir.join("meta.json").exists());

        // Parity: candidate set equals the single-node fresh build.
        let expected = dedup_candidates_sharded(&entries, 7, 2, shard_bits);
        assert_eq!(pairs1, expected);

        // Back to nodes=1 on the same dir: uses project_{id}/ (ITMIHP1),
        // and the _mn bundle is left alone.
        let (p3, l3) = dedup_candidates_sharded_cached(&entries, 7, 2, shard_bits, Some(&dir))
            .expect("single-node build");
        assert!(!l3);
        let meta1: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta1["magic"], "ITMIHP1");
        assert_eq!(p3, expected);

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&mn);
    }

    /// Every invalidation axis on the `_mn` bundle forces a rebuild —
    /// never a silent reuse of a stale multi-node index.
    #[test]
    fn project_gate_index_multi_node_invalidation() {
        let mut entries = sample_entries(20);
        let shard_bits = 3u32;
        let dir = std::env::temp_dir().join(format!(
            "itrace-mih-mn-inv-{}-{}",
            std::process::id(),
            0x5555
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let build = |e: &[DedupKeys], bits: u32, nodes: u32| {
            load_or_build_project_gate_index_multi(&dir, e, bits, nodes).unwrap()
        };

        let (_, _, l1) = build(&entries, shard_bits, 4);
        assert!(!l1);
        let (_, _, l2) = build(&entries, shard_bits, 4);
        assert!(l2, "unchanged bundle must hit");

        // shard_bits mismatch → rebuild
        let (_, _, l) = build(&entries, shard_bits + 1, 4);
        assert!(!l);
        // node_count mismatch → rebuild (stored bits=4,nodes=4 now)
        let (_, _, l) = build(&entries, shard_bits + 1, 8);
        assert!(!l);
        let (_, _, l) = build(&entries, shard_bits + 1, 8);
        assert!(l, "rebuilt bundle for new geometry must hit");

        // image-set change → rebuild
        let mut extra = sample_entries(1);
        extra[0].image_id = 9999;
        entries.push(extra.remove(0));
        let (_, _, l) = build(&entries, shard_bits + 1, 8);
        assert!(!l, "changed image set must invalidate");

        // feature fingerprint change → rebuild
        entries[0].variant_keys[0][0] ^= 0x1;
        let (_, _, l) = build(&entries, shard_bits + 1, 8);
        assert!(!l, "mutated feature must invalidate");

        // missing per-algo cluster dir → rebuild
        std::fs::remove_dir_all(dir.join("indexes/1")).unwrap();
        let (_, _, l) = build(&entries, shard_bits + 1, 8);
        assert!(!l, "incomplete cluster dirs must invalidate");

        // corrupt cluster meta (bad magic) → load_dir fails → rebuild
        let bad_meta = r#"{"magic":"WRONG","version":1}"#;
        std::fs::write(dir.join("indexes/0/meta.json"), bad_meta).unwrap();
        let (_, _, l) = build(&entries, shard_bits + 1, 8);
        assert!(!l, "bad cluster magic must invalidate");

        // top-level bad magic → rebuild
        std::fs::write(dir.join("meta.json"), bad_meta).unwrap();
        let (_, _, l) = build(&entries, shard_bits + 1, 8);
        assert!(!l, "bad top-level magic must invalidate");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn project_mih_index_path_layout() {
        let p = project_mih_index_path(std::path::Path::new("/tmp/mih"), 42);
        assert_eq!(p, std::path::PathBuf::from("/tmp/mih/project_42"));
        let c = project_crop_mih_index_path(std::path::Path::new("/tmp/mih"), 42);
        assert_eq!(c, std::path::PathBuf::from("/tmp/mih/project_42_crop"));
    }

    /// Crop channel: pairs are emitted on `min_hits` distinct probe-key
    /// hits (a hit counts an owner once per probe), not per-algo votes.
    #[test]
    fn crop_candidates_hit_counting() {
        let mut rng = 0xBADC0FFEE_u64;
        // 30 images with 16 random window keys each.
        let mut entries: Vec<CropKeys> = (0..30i64)
            .map(|image_id| CropKeys {
                image_id,
                keys: (0..16).map(|_| xorshift(&mut rng)).collect(),
            })
            .collect();
        // entry 1 is a "crop" of entry 0: 3 keys within radius 4.
        for k in 0..3 {
            entries[1].keys[k] = entries[0].keys[k] ^ 0x5;
        }
        // entry 2 shares exactly one near key with entry 0.
        entries[2].keys[0] = entries[0].keys[0] ^ 0x3;

        let pairs1 = crop_candidates(&entries, 4, 1, 0);
        assert!(pairs1.contains(&(0, 1)));
        assert!(pairs1.contains(&(0, 2)));
        let pairs2 = crop_candidates(&entries, 4, 2, 0);
        assert!(pairs2.contains(&(0, 1)));
        assert!(!pairs2.contains(&(0, 2)), "single hit must not emit");
        // sharded parity
        let pairs_s = crop_candidates(&entries, 4, 2, 6);
        assert_eq!(pairs2, pairs_s);
    }

    #[test]
    fn crop_index_save_load_roundtrip() {
        // Lock + force-unset: `crop_candidates_cached` branches on
        // `ITRACE_MIH_NODES` for the `_crop_mn` layout.
        let _guard = NODES_ENV_LOCK.lock().unwrap();
        std::env::remove_var("ITRACE_MIH_NODES");
        let mut rng = 0xFEED_FACE_u64;
        let entries: Vec<CropKeys> = (0..25i64)
            .map(|image_id| CropKeys {
                image_id: image_id + 7, // non-dense
                keys: (0..12).map(|_| xorshift(&mut rng)).collect(),
            })
            .collect();
        let dir = std::env::temp_dir().join(format!(
            "itrace-mih-crop-{}-{}",
            std::process::id(),
            xorshift(&mut rng)
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let expected = crop_candidates(&entries, 5, 1, 4);
        let (p1, loaded1) = crop_candidates_cached(&entries, 5, 1, 4, Some(&dir)).expect("build");
        assert!(!loaded1);
        assert_eq!(p1, expected);
        let (p2, loaded2) = crop_candidates_cached(&entries, 5, 1, 4, Some(&dir)).expect("load");
        assert!(loaded2);
        assert_eq!(p1, p2);
        assert!(dir.join("index/meta.json").is_file());

        // different image set → invalidate → rebuild
        let mut changed = entries.clone();
        changed[0].image_id = 4242;
        let (p3, loaded3) =
            crop_candidates_cached(&changed, 5, 1, 4, Some(&dir)).expect("rebuild");
        assert!(!loaded3);
        assert_eq!(p3, crop_candidates(&changed, 5, 1, 4));

        // same ids but a mutated feature key → fingerprint mismatch → rebuild
        // (mutate `changed` so the image set stays identical to the bundle)
        let mut mutated = changed.clone();
        mutated[2].keys[0] ^= 0x1;
        let (p4, loaded4) =
            crop_candidates_cached(&mutated, 5, 1, 4, Some(&dir)).expect("fp rebuild");
        assert!(!loaded4, "mutated crop keys must invalidate");
        assert_eq!(p4, crop_candidates(&mutated, 5, 1, 4));
        let (_, loaded5) =
            crop_candidates_cached(&mutated, 5, 1, 4, Some(&dir)).expect("fp reload");
        assert!(loaded5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `project_{id}_crop_mn/` (ITMIHCN1) round-trip: contiguous sorted-id
    /// partition, per-node ITMIHC1 bundles, exact candidate parity with
    /// the single-node index, full cache hit, and single-node
    /// coexistence on the same index dir.
    #[test]
    fn crop_index_multi_roundtrip_parity() {
        let _guard = NODES_ENV_LOCK.lock().unwrap();
        let mut rng = 0xC0FFEE_u64;
        // Shared key prefixes across images guarantee real candidate
        // pairs; each image's keys live in exactly one node index.
        let entries: Vec<CropKeys> = (0..9i64)
            .map(|k| {
                let mut keys: Vec<u64> = (0..8).map(|_| xorshift(&mut rng)).collect();
                keys.push(0xAAAA_0000 + (k % 3) as u64);
                keys.push(0xAAAA_0001 + (k % 3) as u64);
                keys.push(0xBBBB_0000 + (k / 4) as u64);
                CropKeys {
                    image_id: k * 3 + 5, // non-dense
                    keys,
                }
            })
            .collect();
        let dir = std::env::temp_dir().join(format!(
            "itrace-mih-crop-mn-{}-{}",
            std::process::id(),
            0x3333
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mn_dir = project_mih_multi_dir(&dir);

        std::env::set_var("ITRACE_MIH_NODES", "3");
        let (p1, loaded1) =
            crop_candidates_cached(&entries, 4, 2, 4, Some(&dir)).expect("mn build");
        assert!(!loaded1);
        // Expected pairs from the single-node global index (in-memory).
        let expected = crop_candidates(&entries, 4, 2, 4);
        assert_eq!(p1, expected, "multi-node candidates must equal single-node");
        assert!(!expected.is_empty(), "test needs real pairs");

        // Layout: ITMIHCN1 top meta + node_{0,1,2}/ ITMIHC1 bundles;
        // single-node `project_{id}_crop` (dir) NOT created.
        let top: serde_json::Value =
            serde_json::from_slice(&std::fs::read(mn_dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(top["magic"], "ITMIHCN1");
        assert_eq!(top["node_count"], 3);
        assert_eq!(top["image_count"], 9);
        assert_eq!(top["ranges"].as_array().unwrap().len(), 3);
        for i in 0..3 {
            let nmeta: serde_json::Value = serde_json::from_slice(
                &std::fs::read(mn_dir.join(format!("node_{i}/meta.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(nmeta["magic"], "ITMIHC1");
            assert!(mn_dir.join(format!("node_{i}/image_ids.bin")).is_file());
            assert!(mn_dir.join(format!("node_{i}/index")).is_dir());
        }
        assert!(!dir.join("meta.json").exists());

        // Second call: full cache hit, identical pairs.
        let (p2, loaded2) = crop_candidates_cached(&entries, 4, 2, 4, Some(&dir)).expect("mn hit");
        assert!(loaded2);
        assert_eq!(p2, p1);

        // Single-node on the same dir writes `project_{id}_crop` and
        // leaves `_crop_mn` intact.
        std::env::remove_var("ITRACE_MIH_NODES");
        let (p3, loaded3) =
            crop_candidates_cached(&entries, 4, 2, 4, Some(&dir)).expect("sn build");
        assert!(!loaded3);
        assert_eq!(p3, expected);
        let sn_meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(sn_meta["magic"], "ITMIHC1");
        assert!(!sn_meta.as_object().unwrap().contains_key("ranges"));
        // _crop_mn top meta still ITMIHCN1 and still cache-hits.
        let top2: serde_json::Value =
            serde_json::from_slice(&std::fs::read(mn_dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(top2["magic"], "ITMIHCN1");
        std::env::set_var("ITRACE_MIH_NODES", "3");
        let (p4, loaded4) =
            crop_candidates_cached(&entries, 4, 2, 4, Some(&dir)).expect("mn rehit");
        assert!(loaded4);
        assert_eq!(p4, p1);

        std::env::remove_var("ITRACE_MIH_NODES");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&mn_dir);
    }

    /// `_crop_mn` invalidation axes: node_count / fingerprint /
    /// image-set / missing node dir / corrupt node or top meta → rebuild,
    /// never silent reuse; rebuilt bundle cache-hits.
    #[test]
    fn crop_index_multi_invalidation() {
        let _guard = NODES_ENV_LOCK.lock().unwrap();
        std::env::set_var("ITRACE_MIH_NODES", "3");
        let mut rng = 0xBAD5EED_u64;
        let entries: Vec<CropKeys> = (0..8i64)
            .map(|k| CropKeys {
                image_id: k * 2 + 1,
                keys: (0..6).map(|_| xorshift(&mut rng)).collect(),
            })
            .collect();
        let dir = std::env::temp_dir().join(format!(
            "itrace-mih-crop-mn-inv-{}-{}",
            std::process::id(),
            0x4444
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mn_dir = project_mih_multi_dir(&dir);
        let build = |e: &[CropKeys]| crop_candidates_cached(e, 4, 2, 4, Some(&dir)).expect("call");

        let (_, loaded) = build(&entries);
        assert!(!loaded);
        let (_, loaded) = build(&entries);
        assert!(loaded, "sanity: fresh bundle cache-hits");

        // node_count change → rebuild.
        std::env::set_var("ITRACE_MIH_NODES", "4");
        let (_, loaded) = build(&entries);
        assert!(!loaded, "node_count change must rebuild");
        let (_, loaded) = build(&entries);
        assert!(loaded);
        std::env::set_var("ITRACE_MIH_NODES", "3");
        let (_, loaded) = build(&entries);
        assert!(!loaded, "node_count flip-back must rebuild");
        let (_, loaded) = build(&entries);
        assert!(loaded);

        // Same ids, mutated key → fingerprint mismatch → rebuild.
        let mut mutated = entries.clone();
        mutated[1].keys[0] ^= 0x1;
        let (_, loaded) = build(&mutated);
        assert!(!loaded, "feature fingerprint change must rebuild");
        let (_, loaded) = build(&mutated);
        assert!(loaded);

        // Image-set change → rebuild.
        let mut changed = mutated.clone();
        changed[0].image_id = 999;
        let (_, loaded) = build(&changed);
        assert!(!loaded);
        let (_, loaded) = build(&changed);
        assert!(loaded);

        // Missing node dir → rebuild.
        std::fs::remove_dir_all(mn_dir.join("node_2")).unwrap();
        let (_, loaded) = build(&changed);
        assert!(!loaded, "missing node dir must rebuild");
        let (_, loaded) = build(&changed);
        assert!(loaded);

        // Corrupt node meta magic → rebuild.
        let p = mn_dir.join("node_1/meta.json");
        let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        m["magic"] = serde_json::json!("BROKEN1");
        std::fs::write(&p, serde_json::to_vec(&m).unwrap()).unwrap();
        let (_, loaded) = build(&changed);
        assert!(!loaded, "corrupt node meta must rebuild");
        let (_, loaded) = build(&changed);
        assert!(loaded);

        // Corrupt top meta magic → rebuild.
        let p = mn_dir.join("meta.json");
        let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        m["magic"] = serde_json::json!("BROKEN1");
        std::fs::write(&p, serde_json::to_vec(&m).unwrap()).unwrap();
        let (_, loaded) = build(&changed);
        assert!(!loaded, "corrupt top meta must rebuild");
        let (_, loaded) = build(&changed);
        assert!(loaded);

        std::env::remove_var("ITRACE_MIH_NODES");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&mn_dir);
    }
}

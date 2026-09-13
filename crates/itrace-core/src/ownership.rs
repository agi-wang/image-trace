//! Multi-node shard ownership + in-process scatter/gather (Phase 3).
//!
//! `ShardedMihIndex` splits the u64 key space into `2^shard_bits` shards.
//! [`ShardOwnership`] partitions that shard space across logical nodes by
//! contiguous shard ranges; [`MultiNodeMihIndex`] runs one node-local
//! [`MihIndex`] per owned shard in-process, routes inserts to the owning
//! node, and fans queries out only to nodes that own a shard in the
//! Hamming-ball probe set — the same probe set a monolithic
//! `ShardedMihIndex` would scan.
//!
//! This is the in-process foundation for a networked deployment: a real RPC
//! layer would keep `ShardOwnership` (or an equivalent routing table) on the
//! coordinator, replace each `NodeIndex` with a stub that forwards
//! `insert`/`query_shards` over the wire, and merge the returned owner sets
//! exactly as `MultiNodeMihIndex` does today.

use crate::index::{shard_id_for, MihIndex};

/// Contiguous half-open shard range `[start, end)` owned by one node.
/// `start == end` is an empty range (node currently holds no shards —
/// useful for staged drain/join).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardRange {
    pub start: u32,
    pub end: u32,
}

impl ShardRange {
    pub fn contains(&self, shard_id: u32) -> bool {
        shard_id >= self.start && shard_id < self.end
    }

    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Partition of `0 .. 2^shard_bits` shard ids across `node_count` nodes as
/// contiguous ranges — one [`ShardRange`] per node, ordered by node id.
///
/// Contiguity keeps `owner_of` O(1) after a binary search, makes node-local
/// storage a simple array offset, and preserves the stable shard-id order
/// that `shard_id_for` was chosen to provide.
#[derive(Clone, Debug)]
pub struct ShardOwnership {
    shard_bits: u32,
    /// `ranges[node_id]`, sorted by `start`, covering every shard exactly
    /// once (validated at construction).
    ranges: Vec<ShardRange>,
}

impl ShardOwnership {
    /// Evenly split `2^shard_bits` shards across `node_count` nodes; earlier
    /// nodes take one extra shard when the split is uneven. Panics if
    /// `node_count == 0` or `shard_bits > 16`. Nodes beyond `2^shard_bits`
    /// get empty ranges.
    pub fn even(shard_bits: u32, node_count: u32) -> Self {
        assert!(node_count > 0, "node_count must be >= 1");
        assert!(
            shard_bits <= 16,
            "shard_bits {shard_bits} exceeds sensible max 16"
        );
        let total = 1u32 << shard_bits;
        let base = total / node_count;
        let extra = total % node_count;
        let mut ranges = Vec::with_capacity(node_count as usize);
        let mut start = 0u32;
        for n in 0..node_count {
            let len = base + u32::from(n < extra);
            ranges.push(ShardRange {
                start,
                end: start + len,
            });
            start += len;
        }
        Self {
            shard_bits,
            ranges,
        }
    }

    /// Build from explicit ranges; must cover `0 .. 2^shard_bits` exactly
    /// once, in order (empty ranges permitted).
    pub fn from_ranges(shard_bits: u32, ranges: Vec<ShardRange>) -> Result<Self, String> {
        if ranges.is_empty() {
            return Err("ownership needs at least one node range".into());
        }
        let this = Self {
            shard_bits,
            ranges,
        };
        this.validate()?;
        Ok(this)
    }

    /// Check ranges are sorted, non-overlapping, and tile
    /// `0 .. 2^shard_bits` with no gaps.
    pub fn validate(&self) -> Result<(), String> {
        let total = 1u32 << self.shard_bits;
        let mut expect = 0u32;
        for (node, r) in self.ranges.iter().enumerate() {
            if r.start > r.end {
                return Err(format!("node {node}: inverted range {r:?}"));
            }
            if r.start != expect {
                return Err(format!(
                    "node {node}: range {r:?} does not start at {expect} (gap or overlap)"
                ));
            }
            expect = r.end;
        }
        if expect != total {
            return Err(format!("ranges end at {expect}, shard space has {total} shards"));
        }
        Ok(())
    }

    pub fn shard_bits(&self) -> u32 {
        self.shard_bits
    }

    pub fn shard_count(&self) -> u32 {
        1u32 << self.shard_bits
    }

    pub fn node_count(&self) -> u32 {
        self.ranges.len() as u32
    }

    /// Node that owns `shard_id`.
    pub fn owner_of(&self, shard_id: u32) -> u32 {
        assert!(
            shard_id < self.shard_count(),
            "shard_id {shard_id} out of range"
        );
        // Ranges are sorted and contiguous; the owner is the last range
        // whose start <= shard_id (empty ranges have start == end, so a
        // sid equal to such a start falls through to the next non-empty
        // range — correct, since an empty range contains nothing).
        let idx = self.ranges.partition_point(|r| r.start <= shard_id) - 1;
        debug_assert!(self.ranges[idx].contains(shard_id));
        idx as u32
    }

    /// Shard range owned by `node`.
    pub fn shards_of(&self, node: u32) -> ShardRange {
        self.ranges[node as usize]
    }

    pub fn ranges(&self) -> &[ShardRange] {
        &self.ranges
    }
}

/// One logical node's local storage: `MihIndex`es for exactly the shards in
/// `range`, indexed as `shards[sid - range.start]`. Shards the node does
/// not own consume no memory — at 10^11 scale a node holds only its share.
struct NodeIndex {
    range: ShardRange,
    shards: Vec<MihIndex>,
}

impl NodeIndex {
    fn new(range: ShardRange) -> Self {
        Self {
            range,
            shards: (0..range.len()).map(|_| MihIndex::new()).collect(),
        }
    }

    fn insert(&mut self, shard_id: u32, key: u64, owner: u32) {
        debug_assert!(self.range.contains(shard_id));
        self.shards[(shard_id - self.range.start) as usize].insert(key, owner);
    }

    /// Query only owned shards whose id is within `radius` of `prefix`
    /// (same probe filter as `ShardedMihIndex::query_into`). Returns whether
    /// any owned shard was in the probe set — i.e. whether a networked
    /// deployment would have contacted this node at all.
    fn query_into(
        &self,
        key: u64,
        radius: u32,
        prefix: u64,
        out: &mut Vec<u32>,
        scratch: &mut Vec<u32>,
    ) -> bool {
        let mut probed = false;
        for (i, shard) in self.shards.iter().enumerate() {
            let sid = self.range.start as u64 + i as u64;
            if (sid ^ prefix).count_ones() <= radius {
                probed = true;
                shard.query_into(key, radius, scratch);
                out.extend_from_slice(scratch);
            }
        }
        probed
    }

    fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }
}

/// In-process multi-node MIH index: inserts route by shard ownership,
/// queries scatter to the nodes owning probe-set shards and gather their
/// owner sets.
pub struct MultiNodeMihIndex {
    ownership: ShardOwnership,
    nodes: Vec<NodeIndex>,
}

impl MultiNodeMihIndex {
    pub fn new(ownership: ShardOwnership) -> Self {
        debug_assert!(ownership.validate().is_ok());
        let nodes = ownership
            .ranges()
            .iter()
            .map(|&r| NodeIndex::new(r))
            .collect();
        Self { ownership, nodes }
    }

    /// Convenience: `ShardOwnership::even(shard_bits, node_count)` + `new`.
    pub fn even(shard_bits: u32, node_count: u32) -> Self {
        Self::new(ShardOwnership::even(shard_bits, node_count))
    }

    pub fn ownership(&self) -> &ShardOwnership {
        &self.ownership
    }

    pub fn shard_bits(&self) -> u32 {
        self.ownership.shard_bits()
    }

    pub fn node_count(&self) -> u32 {
        self.ownership.node_count()
    }

    pub fn len(&self) -> usize {
        self.nodes.iter().map(|n| n.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn insert(&mut self, key: u64, owner: u32) {
        let sid = shard_id_for(key, self.shard_bits()) as u32;
        let node = self.ownership.owner_of(sid) as usize;
        self.nodes[node].insert(sid, key, owner);
    }

    /// Owners within Hamming `radius`, plus the ids of the nodes the query
    /// scattered to (nodes owning ≥1 probe-set shard). With `radius <
    /// shard_bits` this is a strict subset of all nodes — the property that
    /// makes sharding worthwhile.
    pub fn query_with_nodes(&self, key: u64, radius: u32) -> (Vec<u32>, Vec<u32>) {
        let mut out = Vec::new();
        let mut contacted = Vec::new();
        let mut scratch = Vec::new();
        let prefix = shard_id_for(key, self.shard_bits()) as u64;
        for (node_id, node) in self.nodes.iter().enumerate() {
            if node.query_into(key, radius, prefix, &mut out, &mut scratch) {
                contacted.push(node_id as u32);
            }
        }
        out.sort_unstable();
        out.dedup();
        (out, contacted)
    }

    /// Owner ids within Hamming `radius` — identical result to
    /// `ShardedMihIndex::query` on the same key set.
    pub fn query(&self, key: u64, radius: u32) -> Vec<u32> {
        self.query_with_nodes(key, radius).0
    }

    /// `ShardedMihIndex`-compatible `query_into`: clears `out` and appends
    /// the sorted, deduplicated owner set.
    pub fn query_into(&self, key: u64, radius: u32, out: &mut Vec<u32>) {
        out.clear();
        out.extend_from_slice(&self.query(key, radius));
    }

    /// Insert `key` for `owner`; return prior owners within `radius`.
    pub fn insert_query(&mut self, key: u64, owner: u32, radius: u32) -> Vec<u32> {
        let hits = self.query(key, radius);
        self.insert(key, owner);
        hits
    }

    /// Persist the whole cluster under `path/` — top-level `meta.json`
    /// (`ITMIHN1`: shard_bits + ownership ranges) plus one `node_N/`
    /// directory per node, each written by [`NodeIndex::save_dir`].
    pub fn save_dir(&self, path: &std::path::Path) -> std::io::Result<()> {
        if path.exists() {
            std::fs::remove_dir_all(path)?;
        }
        std::fs::create_dir_all(path)?;
        let meta = serde_json::json!({
            "magic": NODE_MIH_MAGIC,
            "version": 1,
            "shard_bits": self.shard_bits(),
            "key_count": self.len(),
            "ranges": self
                .ownership
                .ranges()
                .iter()
                .map(|r| serde_json::json!({"start": r.start, "end": r.end}))
                .collect::<Vec<_>>(),
        });
        write_meta(&path.join("meta.json"), &meta)?;
        for (node_id, node) in self.nodes.iter().enumerate() {
            node.save_dir(
                &path.join(node_dir_name(node_id as u32)),
                node_id as u32,
                self.shard_bits(),
            )?;
        }
        Ok(())
    }

    /// Load a cluster written by [`save_dir`](Self::save_dir).
    pub fn load_dir(path: &std::path::Path) -> std::io::Result<Self> {
        let meta = read_meta(&path.join("meta.json"))?;
        let shard_bits = check_meta(&meta, path)?;
        let ranges: Vec<ShardRange> = meta
            .get("ranges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad_data("missing ranges"))?
            .iter()
            .map(|r| {
                let get = |k: &str| -> std::io::Result<u32> {
                    r.get(k)
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32)
                        .ok_or_else(|| bad_data("range missing start/end"))
                };
                Ok(ShardRange {
                    start: get("start")?,
                    end: get("end")?,
                })
            })
            .collect::<std::io::Result<_>>()?;
        let ownership = ShardOwnership::from_ranges(shard_bits, ranges).map_err(bad_data)?;
        let mut nodes = Vec::with_capacity(ownership.node_count() as usize);
        for node_id in 0..ownership.node_count() {
            nodes.push(NodeIndex::load_dir(
                &path.join(node_dir_name(node_id)),
                node_id,
                shard_bits,
            )?);
        }
        Ok(Self { ownership, nodes })
    }

    /// Persist just `node_id`'s owned shards (the per-node unit a real
    /// deployment would write on each host).
    pub fn save_node_dir(&self, node_id: u32, path: &std::path::Path) -> std::io::Result<()> {
        self.nodes[node_id as usize].save_dir(path, node_id, self.shard_bits())
    }
}

/// Magic for multi-node index bundles (top-level + per-node meta).
const NODE_MIH_MAGIC: &str = "ITMIHN1";

fn node_dir_name(node_id: u32) -> String {
    format!("node_{node_id}")
}

fn bad_data(msg: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

fn write_meta(path: &std::path::Path, meta: &serde_json::Value) -> std::io::Result<()> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(meta).map_err(bad_data)?,
    )
}

fn read_meta(path: &std::path::Path) -> std::io::Result<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(path)?).map_err(bad_data)
}

/// Validate magic/version, return `shard_bits`.
fn check_meta(meta: &serde_json::Value, path: &std::path::Path) -> std::io::Result<u32> {
    let magic = meta.get("magic").and_then(|v| v.as_str()).unwrap_or("");
    if magic != NODE_MIH_MAGIC {
        return Err(bad_data(format!("bad mih-node magic in {path:?}: {magic}")));
    }
    if meta.get("version").and_then(|v| v.as_u64()).unwrap_or(0) != 1 {
        return Err(bad_data("unsupported mih-node version"));
    }
    let bits = meta
        .get("shard_bits")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| bad_data("missing shard_bits"))? as u32;
    if bits > 16 {
        return Err(bad_data(format!("shard_bits {bits} > 16")));
    }
    Ok(bits)
}

impl NodeIndex {
    /// Node-local layout: `meta.json` (magic `ITMIHN1`, `node_id`, owned
    /// `shard_ids` — only non-empty shards are listed) + `shards/NNNN.bin`
    /// in the same compact format as `ShardedMihIndex` (`u64 n`, `n×u64`
    /// keys, `n×u32` owners, LE).
    fn save_dir(
        &self,
        path: &std::path::Path,
        node_id: u32,
        shard_bits: u32,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(path.join("shards"))?;
        let mut shard_ids = Vec::new();
        let mut key_count = 0usize;
        for (i, shard) in self.shards.iter().enumerate() {
            if shard.is_empty() {
                continue;
            }
            let sid = self.range.start + i as u32;
            shard.save_bin(&path.join("shards").join(format!("{sid:04}.bin")))?;
            shard_ids.push(sid);
            key_count += shard.len();
        }
        let meta = serde_json::json!({
            "magic": NODE_MIH_MAGIC,
            "version": 1,
            "shard_bits": shard_bits,
            "node_id": node_id,
            "range": {"start": self.range.start, "end": self.range.end},
            "shard_ids": shard_ids,
            "key_count": key_count,
        });
        write_meta(&path.join("meta.json"), &meta)
    }

    fn load_dir(path: &std::path::Path, node_id: u32, shard_bits: u32) -> std::io::Result<Self> {
        let meta = read_meta(&path.join("meta.json"))?;
        let stored_bits = check_meta(&meta, path)?;
        let range_v = meta
            .get("range")
            .ok_or_else(|| bad_data("missing range"))?;
        let get = |k: &str| -> std::io::Result<u32> {
            range_v
                .get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| bad_data("range missing start/end"))
        };
        let range = ShardRange {
            start: get("start")?,
            end: get("end")?,
        };
        if meta.get("node_id").and_then(|v| v.as_u64()) != Some(node_id as u64) {
            return Err(bad_data(format!(
                "node dir {path:?} does not hold node {node_id}"
            )));
        }
        let mut shards: Vec<MihIndex> =
            (0..range.len()).map(|_| MihIndex::new()).collect();
        for v in meta
            .get("shard_ids")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad_data("missing shard_ids"))?
        {
            let sid = v.as_u64().ok_or_else(|| bad_data("bad shard_ids"))? as u32;
            if !range.contains(sid) {
                return Err(bad_data(format!("shard {sid} outside node range {range:?}")));
            }
            shards[(sid - range.start) as usize] =
                MihIndex::load_bin(&path.join("shards").join(format!("{sid:04}.bin")))?;
        }
        if stored_bits != shard_bits {
            return Err(bad_data(format!(
                "node {node_id} shard_bits {stored_bits} != cluster {shard_bits}"
            )));
        }
        Ok(Self { range, shards })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::ShardedMihIndex;

    /// Deterministic SplitMix64 — no rand dep in this crate.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn ownership_even_partitions_cover_all_shards() {
        let o = ShardOwnership::even(4, 4); // 16 shards / 4 nodes
        assert_eq!(o.shard_count(), 16);
        assert_eq!(
            o.ranges(),
            &[
                ShardRange { start: 0, end: 4 },
                ShardRange { start: 4, end: 8 },
                ShardRange { start: 8, end: 12 },
                ShardRange { start: 12, end: 16 },
            ]
        );
        for sid in 0..16 {
            assert!(o.shards_of(o.owner_of(sid)).contains(sid));
        }
        assert_eq!(o.owner_of(0), 0);
        assert_eq!(o.owner_of(5), 1);
        assert_eq!(o.owner_of(15), 3);
        o.validate().unwrap();
    }

    #[test]
    fn ownership_even_uneven_split_puts_remainder_first() {
        let o = ShardOwnership::even(4, 3); // 16 / 3 -> 6,5,5
        assert_eq!(
            o.ranges(),
            &[
                ShardRange { start: 0, end: 6 },
                ShardRange { start: 6, end: 11 },
                ShardRange { start: 11, end: 16 },
            ]
        );
        assert_eq!(o.owner_of(6), 1);
    }

    #[test]
    fn ownership_from_ranges_rejects_gaps_and_overlaps() {
        // gap between 4 and 6
        assert!(ShardOwnership::from_ranges(
            3,
            vec![
                ShardRange { start: 0, end: 4 },
                ShardRange { start: 6, end: 8 },
            ]
        )
        .is_err());
        // overlap at 4
        assert!(ShardOwnership::from_ranges(
            3,
            vec![
                ShardRange { start: 0, end: 5 },
                ShardRange { start: 4, end: 8 },
            ]
        )
        .is_err());
        // missing tail
        assert!(ShardOwnership::from_ranges(
            3,
            vec![ShardRange { start: 0, end: 7 }]
        )
        .is_err());
        // valid custom partition with an empty node
        let o = ShardOwnership::from_ranges(
            3,
            vec![
                ShardRange { start: 0, end: 2 },
                ShardRange { start: 2, end: 2 },
                ShardRange { start: 2, end: 8 },
            ],
        )
        .unwrap();
        assert_eq!(o.owner_of(1), 0);
        assert_eq!(o.owner_of(2), 2);
    }

    /// Candidate owner sets must match a monolithic ShardedMihIndex for any
    /// node count and radius — this is the parity contract a networked
    /// scatter/gather has to preserve.
    fn parity_case(shard_bits: u32, node_count: u32, n_keys: usize) {
        let mut rng = Rng(0xdead_beef + node_count as u64);
        let mut mono = ShardedMihIndex::new(shard_bits);
        let mut multi = MultiNodeMihIndex::even(shard_bits, node_count);
        let mut keys = Vec::with_capacity(n_keys);
        for owner in 0..n_keys as u32 {
            let k = rng.next();
            keys.push(k);
            mono.insert(k, owner);
            multi.insert(k, owner);
        }
        assert_eq!(mono.len(), multi.len());
        for radius in [0u32, 1, 2, 5, shard_bits + 4] {
            for _ in 0..200 {
                let q = rng.next();
                assert_eq!(
                    multi.query(q, radius),
                    mono.query(q, radius),
                    "parity failure: bits={shard_bits} nodes={node_count} \
                     radius={radius} key={q:#x}"
                );
            }
            // also probe near existing keys (mutation within radius)
            for &k in keys.iter().step_by(7) {
                let q = k ^ (rng.next() % 8); // flip <=3 low bits
                assert_eq!(multi.query(q, radius), mono.query(q, radius));
            }
        }
    }

    #[test]
    fn multi_node_parity_2_nodes() {
        parity_case(4, 2, 2_000);
    }

    #[test]
    fn multi_node_parity_4_nodes() {
        parity_case(6, 4, 3_000);
    }

    #[test]
    fn multi_node_parity_shard_bits_zero() {
        parity_case(0, 3, 500);
    }

    /// With `radius < shard_bits` the probe set is a proper subset of the
    /// shard space, so at least one node must be skipped entirely.
    #[test]
    fn multi_node_query_only_contacts_probe_set_owners() {
        let shard_bits = 4; // 16 shards
        let multi = MultiNodeMihIndex::even(shard_bits, 4); // 4 shards/node
        let key = 0x5abc_0000_0000_0000u64; // prefix 0b0101 = shard 5 -> node 1

        // radius 0: only shard 5 probed -> only node 1 contacted
        let (_, contacted) = multi.query_with_nodes(key, 0);
        assert_eq!(contacted, vec![1]);

        // radius 1: shards {5} ∪ {5^1,5^2,5^4,5^8} = {1,4,5,7,13}
        //           -> nodes {0,1,1,1,3} = {0,1,3}; node 2 skipped
        let (_, contacted) = multi.query_with_nodes(key, 1);
        assert_eq!(contacted, vec![0, 1, 3]);

        // bigger space: 256 shards / 8 nodes (32 each). radius 2 probes
        // 1+8+28 = 37 of 256 shards -> whole node ranges are skipped.
        let multi8 = MultiNodeMihIndex::even(8, 8);
        let key8 = 0x5a00_0000_0000_0000u64; // prefix 0x5a = 90 -> node 2
        let (_, contacted) = multi8.query_with_nodes(key8, 0);
        assert_eq!(contacted, vec![2]);
        let (_, contacted) = multi8.query_with_nodes(key8, 2);
        assert!(
            contacted.len() < 8,
            "radius 2 probes 37/256 shards; must skip nodes: {contacted:?}"
        );

        // full-radius query legitimately contacts every node
        let (_, contacted) = multi8.query_with_nodes(key8, 16);
        assert_eq!(contacted, (0..8).collect::<Vec<u32>>());
    }

    /// Crop-channel shape: many keys per owner (window slots × variants),
    /// a single index, small radius — the workload `crophash_keys` feeds.
    #[test]
    fn multi_node_parity_crop_like_many_keys_per_owner() {
        let shard_bits = 5;
        let mut rng = Rng(0x00c0_ffee);
        let mut mono = ShardedMihIndex::new(shard_bits);
        let mut multi = MultiNodeMihIndex::even(shard_bits, 4);
        // 200 owners × ~30 keys each, clustered like per-image variants
        for owner in 0..200u32 {
            let base = rng.next();
            for _ in 0..30 {
                let k = base ^ (rng.next() & 0xffff); // low-bit variants
                mono.insert(k, owner);
                multi.insert(k, owner);
            }
        }
        for radius in [0u32, 2, 4, 8] {
            for _ in 0..300 {
                let q = rng.next();
                assert_eq!(multi.query(q, radius), mono.query(q, radius));
            }
        }
    }

    /// Round-trip: save each node to disk, reload the cluster, and require
    /// query parity against both the live multi-node index and a monolithic
    /// `ShardedMihIndex` built from the same keys.
    #[test]
    fn multi_node_persist_roundtrip_parity() {
        let shard_bits = 4;
        let node_count = 4;
        let dir = std::env::temp_dir().join(format!(
            "itrace-mnode-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let mut rng = Rng(42);
        let mut mono = ShardedMihIndex::new(shard_bits);
        let mut multi = MultiNodeMihIndex::even(shard_bits, node_count);
        for owner in 0..500u32 {
            let k = rng.next();
            mono.insert(k, owner);
            multi.insert(k, owner);
        }

        // Per-node save (what a real deployment writes on each host)…
        for node in 0..node_count {
            multi
                .save_node_dir(node, &dir.join(format!("node_{node}")))
                .expect("save node");
        }
        // …but shard ids are only reassembled through the cluster meta, so
        // write a full cluster bundle too and load through it.
        multi.save_dir(&dir).expect("save cluster");
        let loaded = MultiNodeMihIndex::load_dir(&dir).expect("load cluster");
        assert_eq!(loaded.len(), multi.len());
        assert_eq!(
            loaded.ownership().ranges(),
            multi.ownership().ranges()
        );

        for radius in [0u32, 1, 3, 10] {
            for _ in 0..200 {
                let q = rng.next();
                let want = mono.query(q, radius);
                assert_eq!(loaded.query(q, radius), want, "radius={radius}");
                assert_eq!(multi.query(q, radius), want);
            }
        }
        // Node dirs only contain their own shard files — check ownership
        // boundaries held on disk (node 0 owns shards 0..4 at bits=4).
        let node0_shards = std::fs::read_dir(dir.join("node_0/shards"))
            .unwrap()
            .count();
        let total_shards: usize = (0..node_count)
            .map(|n| {
                std::fs::read_dir(dir.join(format!("node_{n}/shards")))
                    .unwrap()
                    .count()
            })
            .sum();
        assert!(node0_shards <= 4);
        assert!(total_shards <= 16);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_node_persist_rejects_wrong_magic() {
        let dir = std::env::temp_dir().join(format!("itrace-mnode-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("meta.json"), br#"{"magic":"ITMIH1","version":1}"#).unwrap();
        assert!(MultiNodeMihIndex::load_dir(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_node_insert_query_routes_and_recalls() {
        let mut multi = MultiNodeMihIndex::even(4, 4);
        // same shard (high nibble 0), hamming distance 1
        let hits = multi.insert_query(0x0000_0000_0000_0000, 7, 4);
        assert!(hits.is_empty());
        let hits = multi.insert_query(0x0000_0000_0000_0001, 8, 4);
        assert_eq!(hits, vec![7]);
        // far key -> no recall
        let hits = multi.insert_query(0xffff_ffff_ffff_ffff, 9, 4);
        assert!(hits.is_empty());
        assert_eq!(multi.len(), 3);
    }
}

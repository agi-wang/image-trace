//! Union-Find grouping over pairwise similarity scores.

use rayon::prelude::*;

/// Disjoint-set with path compression + union by rank.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n] }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    pub fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }

    /// Groups as {root -> member indices}.
    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        use std::collections::HashMap;
        let n = self.parent.len();
        let mut map: HashMap<usize, Vec<usize>> = HashMap::with_capacity(n);
        for i in 0..n {
            let r = self.find(i);
            map.entry(r).or_default().push(i);
        }
        map.into_values().collect()
    }
}

/// Build groups from an N×N score matrix at a threshold.
/// Returns (groups of indices len>1, ungrouped indices).
pub fn cluster(matrix: &[Vec<f64>], threshold: f64) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = matrix.len();
    // Collect the above-threshold edges in parallel — collect() keeps
    // the (i, j) scan order, so the sequential unions below see the
    // exact same edge sequence as a serial scan and produce the same
    // union-find trees.
    let edges: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            matrix[i]
                .iter()
                .enumerate()
                .skip(i + 1)
                .filter_map(move |(j, &s)| (s >= threshold).then_some((i, j)))
        })
        .collect();
    let mut uf = UnionFind::new(n);
    for (i, j) in edges {
        uf.union(i, j);
    }
    let mut groups = Vec::new();
    let mut ungrouped = Vec::new();
    for members in uf.groups() {
        if members.len() > 1 {
            groups.push(members);
        } else {
            ungrouped.extend(members);
        }
    }
    (groups, ungrouped)
}

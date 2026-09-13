//! Dump per-pair algorithm votes across a set of images:
//! `smart_probe threshold img1.png img2.png ...`
//! For every pair with ≥1 vote at `threshold`, prints the filenames and
//! the algos that fired (cross-variant max similarity ≥ threshold).

use std::collections::HashMap;

use itrace_core::features;

type FeatureSets = HashMap<String, HashMap<u8, Vec<u8>>>;

fn compute(img_path: &str) -> FeatureSets {
    let bytes = std::fs::read(img_path).expect("read image");
    let img = itrace_core::image_io::decode(&bytes).expect("decode image");
    let mut out: FeatureSets = HashMap::new();
    for (v, name, bytes, _dims) in features::compute_all_variants(&img) {
        out.entry(name).or_default().insert(v, bytes);
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: smart_probe <threshold> img1.png [img2.png ...]");
        std::process::exit(2);
    }
    let threshold: f64 = args[0].parse().expect("threshold");
    let feats: Vec<FeatureSets> = args[1..].iter().map(|p| compute(p)).collect();
    let names: Vec<String> = args[1..]
        .iter()
        .map(|p| {
            std::path::Path::new(p)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string()
        })
        .collect();

    // algo -> (feature, extractor)
    let mut algos: Vec<(&str, &dyn features::FeatureExtractor)> = Vec::new();
    for ext in features::EXTRACTORS {
        for algo in ext.algorithms() {
            algos.push((algo, *ext));
        }
    }

    for i in 0..feats.len() {
        for j in (i + 1)..feats.len() {
            let mut hits: Vec<String> = Vec::new();
            for &(algo, ext) in &algos {
                let (Some(ma), Some(mb)) =
                    (feats[i].get(ext.feature_name()), feats[j].get(ext.feature_name()))
                else {
                    continue;
                };
                let mut best = 0.0f64;
                for ba in ma.values() {
                    for bb in mb.values() {
                        best = best.max(ext.similarity(ba, bb));
                    }
                }
                if best >= threshold {
                    hits.push(format!("{algo}={best:.3}"));
                }
            }
            if hits.len() >= 2 {
                println!("{} <-> {} : {}", names[i], names[j], hits.join(" "));
            }
        }
    }
}

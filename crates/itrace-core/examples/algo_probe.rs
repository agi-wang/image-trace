//! Per-algorithm pair probe: `algo_probe ref.png cand1.png [cand2.png ...]`
//! prints the cross-variant-max similarity of every registered feature
//! extractor for each candidate vs the reference — a quick way to see
//! which algorithms fire on rotated/cropped/sliced near-duplicates.

use std::collections::HashMap;

use itrace_core::features;

/// feature_name -> (variant -> payload) for one image.
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

fn variant_max(ext: &dyn features::FeatureExtractor, a: &FeatureSets, b: &FeatureSets) -> f64 {
    let (Some(ma), Some(mb)) = (a.get(ext.feature_name()), b.get(ext.feature_name())) else {
        return 0.0;
    };
    let mut best = 0.0f64;
    for (&v, ba) in ma {
        for (&w, bb) in mb {
            best = best.max(ext.similarity(ba, bb));
            let _ = (v, w);
        }
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: algo_probe ref.png cand1.png [cand2.png ...]");
        std::process::exit(2);
    }
    let ra = compute(&args[0]);
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
    // header
    print!("{:<38}", "candidate");
    for ext in features::EXTRACTORS {
        for algo in ext.algorithms() {
            print!("{:>12}", algo);
        }
    }
    println!();
    for (cand, name) in args[1..].iter().zip(&names) {
        let cb = compute(cand);
        print!("{:<38}", name);
        for ext in features::EXTRACTORS {
            for _algo in ext.algorithms() {
                print!("{:>12.4}", variant_max(*ext, &ra, &cb));
            }
        }
        println!();
    }
}

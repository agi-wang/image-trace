//! Benchmark: real-photo transform robustness.
//! Usage: bench <orig_dir> <var_dir>
//! var files: <base>__<transform>.jpg

use std::collections::HashMap;
use std::path::PathBuf;

use itrace_core::compare::{pair_score, Prepared};
use itrace_core::{image_io, slice};
use rayon::prelude::*;

const ALGOS: &[&str] = &[
    "phash", "dhash", "ahash", "whash", "colorhash", "ssim", "histogram", "orb",
];

fn base_of(name: &str) -> &str {
    name.split("__").next().unwrap_or(name)
}
fn transform_of(name: &str) -> &str {
    name.split("__").nth(1).map(|s| s.trim_end_matches(".jpg")).unwrap_or("orig")
}

fn main() -> anyhow::Result<()> {
    let orig_dir = PathBuf::from(std::env::args().nth(1).unwrap());
    let var_dir = PathBuf::from(std::env::args().nth(2).unwrap());
    let desc: Vec<String> = vec!["orb".into()];

    let origs: Vec<(String, Prepared)> = std::fs::read_dir(&orig_dir)?
        .filter_map(|e| {
            let p = e.ok()?.path();
            let n = p.file_stem()?.to_str()?.to_string();
            Prepared::load(&p, &desc, true).ok().map(|pp| (n, pp))
        })
        .collect();
    let vars: Vec<(String, Prepared)> = std::fs::read_dir(&var_dir)?
        .filter_map(|e| {
            let p = e.ok()?.path();
            let n = p.file_stem()?.to_str()?.to_string();
            Prepared::load(&p, &desc, true).ok().map(|pp| (n, pp))
        })
        .collect();
    println!("origs={} vars={}", origs.len(), vars.len());

    // scores[var_idx][algo] = (same_score_raw, same_score_rot, max_other_raw, max_other_rot)
    type Row = (String, String, HashMap<&'static str, (f64, f64, f64, f64)>);
    let rows: Vec<Row> = vars
        .par_iter()
        .map(|(vname, vp)| {
            let vbase = base_of(vname).to_string();
            let vtf = transform_of(vname).to_string();
            let mut per_algo = HashMap::new();
            for &algo in ALGOS {
                let mut same = (0f64, 0f64);
                let mut other = (0f64, 0f64);
                for (oname, op) in &origs {
                    let raw = pair_score(algo, op, vp, false);
                    let rot = pair_score(algo, op, vp, true);
                    if *oname == vbase {
                        same = (raw, rot);
                    } else {
                        other.0 = other.0.max(raw);
                        other.1 = other.1.max(rot);
                    }
                }
                per_algo.insert(algo, (same.0, same.1, other.0, other.1));
            }
            (vname.clone(), vtf, per_algo)
        })
        .collect();

    // ---- per-transform table ----
    let mut tfs: Vec<String> = rows.iter().map(|r| r.1.clone()).collect();
    tfs.sort();
    tfs.dedup();
    println!("\n=== same-pair mean score / hit@0.85 / margin(vs best other) [rotation_invariant] ===");
    print!("{:>10}", "transform");
    for a in ALGOS {
        print!(" | {:>9}", a);
    }
    println!();
    for tf in &tfs {
        let grp: Vec<_> = rows.iter().filter(|r| &r.1 == tf).collect();
        print!("{:>10}", tf);
        for &algo in ALGOS {
            let n = grp.len() as f64;
            let mean = grp.iter().map(|r| r.2[algo].1).sum::<f64>() / n;
            let hits = grp.iter().filter(|r| r.2[algo].1 >= 0.85).count();
            let margin = grp.iter().map(|r| r.2[algo].1 - r.2[algo].3).sum::<f64>() / n;
            print!(" | {:.2}/{:>2}%/{:+.2}", mean, (hits as f64 / n * 100.0) as u32, margin);
        }
        println!();
    }

    println!("\n=== same-pair [non-rot] for reference ===");
    print!("{:>10}", "transform");
    for a in ALGOS {
        print!(" | {:>9}", a);
    }
    println!();
    for tf in &tfs {
        let grp: Vec<_> = rows.iter().filter(|r| &r.1 == tf).collect();
        print!("{:>10}", tf);
        for &algo in ALGOS {
            let n = grp.len() as f64;
            let mean = grp.iter().map(|r| r.2[algo].0).sum::<f64>() / n;
            print!(" | {:>9}", format!("{mean:.2}"));
        }
        println!();
    }

    // ---- negative-pair FP rate at 0.85 (rot-inv) ----
    println!("\n=== false-positive rate @0.85 (different images, rot-inv) ===");
    for &algo in ALGOS {
        let fp = rows.iter().filter(|r| r.2[algo].3 >= 0.85).count();
        println!("  {algo:>10}: {fp}/{} var-images confused with a wrong orig", rows.len());
    }

    // ---- slice coverage on slice/crop transforms ----
    println!("\n=== slice_match coverage (2x2) ===");
    for tf in &["slice_tl", "crop60", "cropc"] {
        let mut covs = Vec::new();
        for (vname, _) in &vars {
            if transform_of(vname) != *tf {
                continue;
            }
            let base = base_of(vname);
            let op = origs.iter().find(|(n, _)| n == base).unwrap();
            let ga = &op.1.gray;
            let gb_path = var_dir.join(format!("{vname}.jpg"));
            let img = image_io::decode_file(&gb_path)?;
            let gb = image_io::to_gray(&image_io::resize_max_side(&img, 384));
            let res = slice::slice_match(ga, &gb, 2, 2, 0.6);
            covs.push((res.coverage, res.is_slice_of_a));
        }
        let n = covs.len() as f64;
        let mean = covs.iter().map(|c| c.0).sum::<f64>() / n;
        let hit = covs.iter().filter(|c| c.1).count();
        println!("  {tf:>8}: mean coverage {mean:.2}, is_slice {hit}/{} ", covs.len());
    }
    Ok(())
}

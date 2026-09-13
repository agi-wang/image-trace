//! Backend parity contract: the same `ImageStore` suite runs against
//! `SqliteStore` (always) and `PostgresStore` (when
//! `ITRACE_TEST_DATABASE_URL` is set — e.g. `docker compose up -d
//! postgres`, then
//! `ITRACE_TEST_DATABASE_URL=postgres://itrace:itrace@localhost:5432/itrace
//!  cargo test -p itrace-store --test store_contract`).
//!
//! Assertions only touch rows the test itself created, so a shared
//! Postgres database across reruns is fine (identity ids never collide;
//! list queries filter by the project/image ids created below).

use std::sync::Arc;

use itrace_store::{ImageStore, NewImage, NewRun, SqliteStore};

fn img(project_id: i64, name: &str) -> NewImage {
    NewImage {
        project_id,
        filename: name.into(),
        file_path: format!("contract/{name}"),
        file_hash: format!("hash-{name}"),
        phash: Some("ph".into()),
        dhash: None,
        ahash: None,
        whash: Some("wh".into()),
        colorhash: None,
        extracted_from: None,
        file_size: Some(1234),
        width: Some(64),
        height: Some(32),
    }
}

/// The shared contract — any `ImageStore` impl must satisfy all of it.
fn contract_suite(store: &Arc<dyn ImageStore>) {
    // ----- blob facade -----
    store.write_file("contract/blob.bin", b"payload").unwrap();
    assert!(store.file_exists("contract/blob.bin"));
    assert_eq!(store.read_file("contract/blob.bin").unwrap(), b"payload");
    store.delete_file("contract/blob.bin").unwrap();
    assert!(!store.file_exists("contract/blob.bin"));
    assert!(store.resolve("contract/x.png").is_some()); // fs blob backend in tests

    // ----- projects -----
    let p = store
        .create_project("contract-proj", Some("d"))
        .unwrap();
    let got = store.get_project(p.id).unwrap();
    assert_eq!(got.name, "contract-proj");
    assert_eq!(got.description.as_deref(), Some("d"));
    assert_eq!(got.image_count, 0);
    assert!(!got.created_at.is_empty());
    store.ensure_project(p.id).unwrap();
    assert!(store.ensure_project(i64::MAX).is_err());
    assert!(store.list_projects(0, 500).unwrap().iter().any(|x| x.id == p.id));

    // ----- images -----
    let a = store.insert_image(&img(p.id, "a.jpg")).unwrap();
    let b = store.insert_image(&img(p.id, "b.jpg")).unwrap();
    assert!(a.id != b.id);
    assert_eq!(a.feature_status, "pending");
    assert_eq!(a.width, Some(64));
    assert!(!a.created_at.is_empty());

    let listed = store.list_images(p.id, 0, 100).unwrap();
    assert_eq!(
        listed.iter().map(|r| r.id).collect::<Vec<_>>(),
        vec![a.id, b.id]
    );
    let page = store.list_images(p.id, 1, 1).unwrap();
    assert_eq!(page[0].id, b.id);

    // meta ordering matches list_images ordering
    let meta = store.list_image_meta(p.id).unwrap();
    assert_eq!(meta.iter().map(|m| m.id).collect::<Vec<_>>(), vec![a.id, b.id]);
    assert_eq!(meta[0].filename, "a.jpg");
    assert_eq!(meta[0].width, Some(64));

    assert_eq!(store.get_image(a.id).unwrap().file_hash, "hash-a.jpg");
    assert_eq!(store.get_image_path(b.id).unwrap(), "contract/b.jpg");
    assert!(store.get_image(i64::MAX).is_err());
    assert_eq!(store.get_project(p.id).unwrap().image_count, 2);

    // ----- feature store -----
    store
        .put_features(
            a.id,
            &[
                (0, "phash", &[1u8; 8][..], 64),
                (1, "phash", &[2u8; 8][..], 64),
                (0, "dhash", &[3u8; 8][..], 64),
            ],
        )
        .unwrap();
    store.put_feature(b.id, 0, "phash", &[7u8; 8], 64).unwrap();
    // re-upsert overwrites, not duplicates
    store.put_feature(a.id, 0, "phash", &[9u8; 8], 64).unwrap();

    let map = store
        .load_feature_map(&[a.id, b.id], "phash", &[0, 1])
        .unwrap();
    assert_eq!(map[&a.id][&0], vec![9u8; 8]);
    assert_eq!(map[&a.id][&1], vec![2u8; 8]);
    assert_eq!(map[&b.id][&0], vec![7u8; 8]);

    let maps = store
        .load_feature_maps(&[a.id, b.id], &["phash", "dhash"], &[0])
        .unwrap();
    assert_eq!(maps.len(), 2);
    assert_eq!(maps[1][&a.id][&0], vec![3u8; 8]);
    assert!(!maps[1].contains_key(&b.id));

    assert!(store.load_feature_map(&[], "phash", &[0]).unwrap().is_empty());
    assert!(store
        .load_feature_map(&[a.id], "phash", &[])
        .unwrap()
        .is_empty());

    assert_eq!(store.feature_algorithm_count(a.id).unwrap(), 2);
    assert_eq!(store.feature_algorithm_count(b.id).unwrap(), 1);

    assert!(!store.features_ready(&[a.id, b.id]).unwrap());
    store.set_feature_status(a.id, "ready").unwrap();
    store.set_feature_status(b.id, "ready").unwrap();
    assert!(store.features_ready(&[a.id, b.id]).unwrap());
    assert!(store.features_ready(&[]).unwrap());

    // ----- pair cache (ordered-pair canonicalization + overwrite) -----
    assert!(store.get_pair_score("x", "y", "phash", false).unwrap().is_none());
    store.put_pair_score("x", "y", "phash", false, 0.75).unwrap();
    // reversed arg order reads the same canonical row
    assert_eq!(
        store.get_pair_score("y", "x", "phash", false).unwrap(),
        Some(0.75)
    );
    store.put_pair_score("x", "y", "phash", false, 0.9).unwrap();
    assert_eq!(
        store.get_pair_score("x", "y", "phash", false).unwrap(),
        Some(0.9)
    );
    // rotation_invariant is part of the key
    assert!(store
        .get_pair_score("x", "y", "phash", true)
        .unwrap()
        .is_none());

    // ----- analysis runs -----
    let r1 = store
        .insert_run(&NewRun {
            project_id: p.id,
            algorithm: "phash".into(),
            threshold: 0.9,
            total_images: 2,
            groups_count: 1,
            unique_count: 0,
            summary: Some("s1".into()),
        })
        .unwrap();
    let r2 = store
        .insert_run(&NewRun {
            project_id: p.id,
            algorithm: "phash".into(),
            threshold: 0.9,
            total_images: 2,
            groups_count: 2,
            unique_count: 0,
            summary: None,
        })
        .unwrap();
    assert!(r2 > r1);
    let runs = store.list_runs(p.id, 0, 10).unwrap();
    assert!(runs.iter().any(|r| r.id == r1) && runs.iter().any(|r| r.id == r2));
    assert!(runs.windows(2).all(|w| w[0].id > w[1].id)); // DESC
    let got_run = store.get_run(r1).unwrap();
    assert_eq!(got_run.groups_count, 1);
    assert!((got_run.threshold - 0.9).abs() < 1e-9);
    let latest = store
        .latest_run(p.id, "phash", 0.9)
        .unwrap()
        .expect("latest");
    assert_eq!(latest.id, r2);
    assert!(store
        .latest_run(p.id, "phash", 0.12345)
        .unwrap()
        .is_none());
    assert!(store.latest_run(p.id, "nope", 0.9).unwrap().is_none());

    // ----- deletes -----
    let rec = store.delete_image(b.id).unwrap();
    assert_eq!(rec.file_hash, "hash-b.jpg");
    assert!(store.get_image(b.id).is_err());
    // delete_image prunes pair_cache rows touching the image's hash
    store
        .put_pair_score("hash-a.jpg", "zzz", "phash", false, 0.5)
        .unwrap();
    store.delete_image(a.id).unwrap();
    assert!(store
        .get_pair_score("zzz", "hash-a.jpg", "phash", false)
        .unwrap()
        .is_none());

    let removed = store.delete_project(p.id).unwrap();
    assert!(removed.iter().all(|r| r.project_id == p.id));
    assert!(store.get_project(p.id).is_err());
}

fn sqlite_store() -> Arc<dyn ImageStore> {
    let dir = std::env::temp_dir().join(format!(
        "itrace-contract-sqlite-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    Arc::new(SqliteStore::open(&dir).unwrap())
}

#[test]
fn contract_sqlite() {
    contract_suite(&sqlite_store());
}

/// Postgres parity — requires a reachable database:
///   docker compose up -d postgres
///   ITRACE_TEST_DATABASE_URL=postgres://itrace:itrace@localhost:5432/itrace \
///     cargo test -p itrace-store --test store_contract contract_postgres
/// Skips (passes vacuously) when the env var is unset. In CI the
/// build-test job provides a postgres:16 service container.
#[test]
fn contract_postgres() {
    let Ok(url) = std::env::var("ITRACE_TEST_DATABASE_URL") else {
        eprintln!("ITRACE_TEST_DATABASE_URL unset — skipping postgres contract test");
        return;
    };
    let dir = std::env::temp_dir().join(format!("itrace-contract-pg-{}", std::process::id()));
    let store: Arc<dyn ImageStore> =
        Arc::new(itrace_store::pg::PostgresStore::connect(&url, &dir).unwrap());
    contract_suite(&store);
}

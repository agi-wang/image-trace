# Postgres metadata backend

`itrace-store` exposes two `ImageStore` implementations:

- `SqliteStore` — **default**; single-file `{data_dir}/image-trace.db`
- `PostgresStore` — `ITRACE_STORE=postgres` + `ITRACE_DATABASE_URL`

Blobs (uploads/extracted/thumbnails) always stay on `BlobStore`
(fs or S3 via `ITRACE_STORAGE`) — only metadata and feature vectors
move to Postgres.

## Selecting the backend

```bash
# sqlite (unchanged default)
itrace-cli dedup 7

# postgres — env or flag (flag wins):
ITRACE_STORE=postgres \
ITRACE_DATABASE_URL=postgres://itrace:itrace@localhost:5432/itrace \
itrace-server

itrace-cli --store postgres projects            # same as env
```

Precedence: `--store` flag > `ITRACE_STORE` > sqlite. The URL comes
from `ITRACE_DATABASE_URL` (preferred) or `DATABASE_URL`. An unknown
backend name is an error, not a silent sqlite fallback.

Dev instance: `docker compose up -d postgres` (user/db `itrace`,
password `itrace`, port 5432). The schema self-applies on connect
(`CREATE TABLE IF NOT EXISTS` — no migration runner needed yet).

## Pointing Postgres at an existing blob dir

`--data-dir` (CLI) / `DATA_DIR` (server) still anchors the fs
`BlobStore`. To move an existing deployment's *metadata* to Postgres
while keeping its files:

1. `docker compose up -d postgres`
2. `ITRACE_STORE=postgres ITRACE_DATABASE_URL=… itrace-cli --data-dir <existing> …`

Blobs resolve identically — `file_path` keys like `uploads/x.jpg`
are backend-agnostic.

## What does **not** migrate

**There is no sqlite→pg dump tool yet.** `PostgresStore::connect`
creates an *empty* schema. If you switch an existing `--data-dir` to
`ITRACE_STORE=postgres`, the Postgres tables start empty while the
blob dir keeps its files — i.e. projects/images/features must be
re-ingested (`add` + `precompute`) or a future dump tool must copy
`projects`, `images`, `feature_store`, `pair_cache`, `analysis_runs`
from `image-trace.db` into PG.

Recommended today:

- **Fresh/empty Postgres + re-ingest** — the safe path. Blobs can be
  reused by re-adding from the original sources (re-uploading into
  `uploads/` rewrites the same keys harmlessly).
- **Persistent MIH index** (`ITRACE_MIH_INDEX_DIR`) rebuilds itself —
  the feature fingerprint changes when vectors are rewritten, so stale
  bundles self-invalidate; nothing to copy.

## Tests

```bash
docker compose up -d postgres
ITRACE_TEST_DATABASE_URL=postgres://itrace:itrace@localhost:5432/itrace \
  cargo test -p itrace-store --test store_contract
```

CI runs the same contract suite against a `postgres:16` service
container on `build-test`.

# Image Trace — Test Coverage Report

Generated: 2026-03-09

## Summary

| Metric | Value |
|:---|:---|
| **Total Tests** | 385 (361 backend + 24 desktop) |
| **Test Files** | 14 (13 backend + 1 desktop) |
| **Status** | ✅ All passing |

## Test Files

| File | Tests | Covers |
|:---|:---|:---|
| `test_image_processor.py` | 56 | Hash computation, similarity, descriptors, extractor factory, grouping, caching |
| `test_utils.py` | 50 | DB utils, file ops, format support, unique filenames, grouping, cleanup |
| `test_smart_compare.py` | 41 | 一键查重, recompute_features, _ensure_precompute, system_info, extract, edge cases |
| `test_pairwise_and_viz.py` | 32 | Pairwise matrix, visualize_match, rotation API, thumbnail, match_data |
| `test_frontend_backend_alignment.py` | 29 | 前端 api.ts ↔ 后端 JSON shape 全量对齐校验 |
| `test_similarity_cache.py` | 27 | **[NEW]** Cache key ordering, _raw_similarity branches, get_or_compute cache hit/miss, invalidation, eviction |
| `test_report_and_endpoints.py` | 25 | /report, /analysis_runs, GET /results, /download, resize, orientation |
| `test_feature_matrix.py` | 25 | Vector serialization, matrix ops, precompute pipeline, are_features_ready, FeatureStore model, algo mapping |
| `test_new_algorithms.py` | 24 | SSIM, Histogram, Template, AKAZE/KAZE, Colorhash, Hybrid fusion |
| `test_api.py` | 22 | CRUD endpoints: projects, upload, images, compare |
| `test_document_parser.py` | 13 | PDF extraction, Office docs, format detection, path conversion |
| `test_internals.py` | 11 | **[NEW]** _resolve_static_path, _delete_image_artifacts, _migrate_db_schema |
| `test_document_advanced.py` | 6 | **[NEW]** _convert_to_jpeg (RGB/RGBA/corrupt), _render_pdf_pages (single/multi/corrupt) |
| `test-resolve-binary.cjs` | 24 | Desktop: binary resolution, build scripts, package.json, CI workflow |

## Frontend–Backend API Alignment Matrix

Every frontend `api.ts` function is verified against the backend JSON response shape:

| Frontend Function | Backend Endpoint | Test Class |
|:---|:---|:---|
| `createProject()` | `POST /projects` | TestProjectCRUDAlignment |
| `getProjects()` | `GET /projects` | TestProjectCRUDAlignment |
| `getProject()` | `GET /projects/{id}` | TestProjectCRUDAlignment |
| `deleteProject()` | `DELETE /projects/{id}` | TestProjectCRUDAlignment |
| `uploadImages()` | `POST /upload` | TestUploadAlignment |
| `uploadDocument()` | `POST /upload` (document) | TestUploadAlignment |
| `extractDocument()` | `POST /extract/{file_path}` | TestExtractEndpoint |
| `getDocument()` | _(stub, no backend)_ | — |
| `getProjectDocuments()` | _(stub, no backend)_ | — |
| `getProjectImages()` | `GET /images/{project_id}` | TestImagesAlignment |
| `deleteImage()` | `DELETE /images/{id}` | TestImagesAlignment |
| `analyzeImages()` | `POST /compare/{id}` | TestAnalysisAlignment |
| `getComparisonResults()` | `GET /results/{id}` | TestAnalysisAlignment |
| `getAnalysisRuns()` | `GET /analysis_runs` | TestAnalysisRunsAlignment |
| `getAnalysisRunDetail()` | `GET /analysis_runs/detail/{id}` | TestAnalysisRunsAlignment |
| `smartCompare()` | `POST /smart_compare/{id}` | TestSmartCompareAlignment |
| `recomputeFeatures()` | `POST /recompute_features/{id}` | TestSmartCompareAlignment |
| `checkHealth()` | `GET /health` | TestHealthAlignment |
| `visualizeMatch()` | `POST /visualize_match` | TestVisualizeMatchAlignment |
| `getPairwiseMatrix()` | `GET /pairwise_matrix/{id}` | TestPairwiseMatrixAlignment |
| `getMatchData()` | `POST /match_data` | TestMatchDataAlignment |
| `getSystemInfo()` | `GET /system_info` | TestSystemInfoAlignment |
| `getFeatureStatus()` | `GET /feature_status/{id}` | TestFeatureStatusAlignment |

## Coverage by Module

### main.py (API Endpoints)
| Endpoint | Tested In |
|:---|:---|
| `GET /` | test_api + test_alignment |
| `GET /health` | test_api + test_alignment |
| `POST /projects` | test_api + test_alignment |
| `GET /projects` | test_api + test_alignment |
| `GET /projects/{id}` | test_api + test_alignment |
| `DELETE /projects/{id}` | test_api + test_alignment |
| `POST /upload` | test_api + test_alignment |
| `POST /compare/{id}` | test_api + test_alignment |
| `POST /smart_compare/{id}` | test_smart_compare + test_alignment |
| `POST /recompute_features/{id}` | test_smart_compare + test_alignment |
| `POST /extract/{file_path}` | test_smart_compare |
| `GET /results/{id}` | test_report + test_alignment |
| `GET /analysis_runs` | test_report + test_alignment |
| `GET /analysis_runs/detail/{id}` | test_report + test_alignment |
| `GET /images/{project_id}` | test_api + test_alignment |
| `DELETE /images/{id}` | test_api + test_alignment |
| `GET /download/{file_path}` | test_report + test_alignment |
| `POST /visualize_match` | test_pairwise + test_alignment |
| `GET /pairwise_matrix/{id}` | test_pairwise + test_alignment |
| `GET /thumbnail/{id}` | test_pairwise + test_alignment |
| `POST /match_data` | test_pairwise + test_alignment |
| `GET /report/{id}` | test_report + test_alignment |
| `GET /feature_status/{id}` | test_feature_matrix + test_alignment |
| `GET /system_info` | test_smart_compare + test_alignment |

### main.py (Internal Functions)
| Function | Tested In |
|:---|:---|
| `_ensure_precompute` | test_smart_compare (3 tests: BackgroundTasks path, Thread fallback, daemon flag) |
| `_bg_precompute_features` | test_smart_compare (via upload + feature_status verification) |
| `SMART_ALGOS` | test_smart_compare (4 tests: count, validity, tier coverage, no duplicates) |
| `_resolve_static_path` | **test_internals** (3 tests: None, data/ prefix, normal path) |
| `_delete_image_artifacts` | **test_internals** (2 tests: file+cache cleanup, missing file) |
| `_migrate_db_schema` | **test_internals** (5 tests: non-sqlite skip, missing DB skip, column addition, table creation, idempotency) |

### image_processor.py (Similarity Cache Layer)
| Function | Tested In |
|:---|:---|
| `_cache_key` | **test_similarity_cache** (3 tests: ordering, same key, empty strings) |
| `_raw_similarity` | **test_similarity_cache** (7 tests: all algo branches + unknown fallback) |
| `get_or_compute_similarity` | **test_similarity_cache** (6 tests: miss/hit/no-session/rotation/multi-algo/symmetry) |
| `invalidate_similarity_cache` | **test_similarity_cache** (3 tests: removal, unknown hash, empty table) |
| `invalidate_feature_cache` | **test_similarity_cache** (3 tests: clear caches, unknown path, preserve others) |
| `_evict_if_needed` | **test_similarity_cache** (4 tests: below/over/empty/exact threshold) |

### feature_matrix.py
| Test Class | Tests | Covers |
|:---|:---|:---|
| `TestVectorSerialization` | 2 | b64 roundtrip (float32, uint8) |
| `TestHashBits` | 2 | hash_str_to_bits, empty hash |
| `TestSingleVariantFeatures` | 1 | All 11 feature types computed |
| `TestVariantGeneration` | 1 | 8 orientation variants |
| `TestHashMatrix` | 2 | Identical (1.0) and different (0.0) |
| `TestCosineMatrix` | 3 | Identical, orthogonal, rotation invariant |
| `TestFeatureStatus` | 2 | Empty project, after upload |
| `TestPrecomputePipeline` | 2 | Full DB pipeline, 8 variants stored |
| `TestAreFeaturesReady` | 3 | Empty list, pending image, ready image |
| `TestBuildMatrix` | 2 | Empty vectors, missing image fills zeros |
| `TestAlgoFeatureMapping` | 3 | ALGO↔FEATURE inverse consistency, feature types |
| `TestFeatureStoreModel` | 2 | Create/read record, 8 variants per image |

### image_processor.py
| Function | Tested In |
|:---|:---|
| `compute_file_md5` | test_image_processor |
| `compute_image_features` | test_image_processor |
| `hamming_distance` | test_image_processor |
| `calculate_similarity` | test_image_processor |
| `compute_descriptor` (ORB/BRISK/SIFT/AKAZE/KAZE) | test_image_processor, test_new_algorithms |
| `calculate_descriptor_similarity` | test_image_processor |
| `get_cached_descriptor` | test_image_processor |
| `draw_feature_matches` | test_image_processor |
| `find_similar_images` | test_image_processor |
| `group_similar_images` | test_image_processor |
| `is_image_file` | test_image_processor |
| `resize_image_if_needed` | test_report_and_endpoints |
| `_create_extractor` | test_image_processor |
| `compute_descriptor_with_kp` | test_image_processor |
| `calculate_ssim_similarity` | test_new_algorithms |
| `calculate_histogram_similarity` | test_new_algorithms |
| `calculate_template_similarity` | test_new_algorithms |
| `calculate_hybrid_similarity` | test_new_algorithms |
| `_generate_orientation_variants` | test_report_and_endpoints |
| `compute_features_for_variants` | test_report_and_endpoints |
| `compare_with_orientations` | test_report_and_endpoints |

### utils.py
| Function | Tested In |
|:---|:---|
| `get_database_url` | test_utils |
| `get_session` | test_utils |
| `ensure_directory` | test_utils |
| `generate_unique_filename` | test_utils |
| `save_upload_file` | test_utils |
| `get_file_size_mb` | test_utils |
| `format_file_size` | test_utils |
| `is_supported_image_format` | test_utils, test_report_and_endpoints |
| `is_supported_document_format` | test_utils |
| `delete_file_if_exists` | test_report_and_endpoints |
| `group_similar_by_metric` | test_utils |
| `compare_images_in_project` | test_utils |
| `cleanup_project_files` | test_utils |

### document_parser.py
| Function | Tested In |
|:---|:---|
| `DocumentParser.__init__` | test_document_parser |
| `_guess_image_extension` | test_document_parser (PNG, JPEG, GIF, BMP, TIFF, unknown) |
| `process_document` | test_document_parser (with/without image, unsupported, nonexistent) |
| `extract_images_from_document` | test_document_parser |
| `_rel_to_base` | test_document_parser |
| `_convert_to_jpeg` | **test_document_advanced** (3 tests: RGB, RGBA, corrupt fallback) |
| `_render_pdf_pages` | **test_document_advanced** (3 tests: single page, corrupt, multi-page) |

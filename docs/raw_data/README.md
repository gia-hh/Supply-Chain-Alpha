# Raw Data Policy

`data/raw/` is append-only source material.

Rules:
- never edit vendor/source files in place;
- create `MANIFEST.json` with file sizes and SHA-256 hashes;
- derived fields belong in `data/interim/` or `data/processed/`;
- no future document is allowed to change an earlier point-in-time graph snapshot;
- publication timestamp, document id, and source path/URL are mandatory provenance for relationship records.

The CSV files in this directory are schema-reference headers, not raw source
data. Production raw files live under `data/raw/official_sources/` and are
covered exactly by `data/raw/MANIFEST.json`.

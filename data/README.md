# data/

All data directories are gitignored. Run the downloader to populate them.

```bash
python data/download_epic.py --data-dir data/epic_raw
python scripts/prepare_epic.py --epic-dir data/epic_raw --output-dir data/epic_processed
```

| Directory | Contents | Size |
|-----------|----------|------|
| `epic_raw/` | Downloaded EPIC v2.0 zips + extracted audio + raw transcripts | ~10 GB |
| `epic_processed/` | Cleaned transcripts paired with audio, ready for experiments | ~1 GB |

See [../docs/dataset.md](../docs/dataset.md) for full corpus documentation.

# Thesis 2: Reckoning — TON_IoT Source Preflight

Status: **feasibility-only**. This command creates no partitions, labels, features, or trained models. It does not certify a dataset for publication.

The handoff specifies the **full official processed TON_IoT Network CSV collection**, not the provider's sampled `Train_Test_datasets` as an equivalent replacement. Official acquisition and licensing: https://research.unsw.edu.au/projects/toniot-datasets . The official portal may require a web browser/login; verify the source independently before continuing. Do not upload or commit raw source files.

Run from the repository root, replacing the local input path with the directory containing the provider's processed **Network** CSVs. Start with a bounded inventory:

```powershell
uv run python scripts/17_preflight_toniot_source.py --input-root "C:\PATH\TO\TON_IOT\Processed\Network" --dry-run
uv run python scripts/17_preflight_toniot_source.py --input-root "C:\PATH\TO\TON_IOT\Processed\Network" --out outputs/17_revision_campaign/toniot/source_preflight_sample.json --max-rows-per-file 10000
```

After reviewing the first manifest, produce complete row counts and SHA-256 hashes (this reads all input data, so may take considerable time):

```powershell
uv run python scripts/17_preflight_toniot_source.py --input-root "C:\PATH\TO\TON_IOT\Processed\Network" --out outputs/17_revision_campaign/toniot/source_preflight_full.json --max-rows-per-file 0 --sha256
```

The resulting JSON contains file paths relative to the input root, source-file sizes, column names, candidate label/time/identity columns, observed label counts, optional full-file SHA-256, and clear sample/full-scan flags. It deliberately excludes row-level values other than category counts. Inspect unexpected schemas or candidate fields manually before constructing a taxonomy or split.

**Mandatory subsequent gates, not implemented by this preflight:** source authenticity and permission verification, dataset-specific taxonomy, leakage-feature exclusions, source/time split construction, train/val/test non-overlap, Protocol-B holdout support, duplicate/identity leakage, and bounded runtime pilots. The official dataset describes network PCAP, Zeek logs/CSVs, processed CSVs, sampled train/test data, and security-event timestamp ground truth as distinct resources; mixing these silently would undermine provenance.

No result from this script alone justifies promotion of TON_IoT into `config/datasets.yml` as a primary environment.

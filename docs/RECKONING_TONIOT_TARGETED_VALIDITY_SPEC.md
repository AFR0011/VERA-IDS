# TON_IoT targeted validity audit — Phase B7

Status: **audit procedure approved for execution, dataset role NOT decided**. User approved the proposed final targeted temporal, address/identity-leakage, and attack-episode-provenance investigation on 2026-10-18. No feature set, model, holdout partition, threshold, dataset promotion or manuscript conclusion has been approved.

## Prior evidence fixed at handoff

1. Complete processed source: 23 sequential Network CSV chunks, **22,339,021 rows**; nine attack families and benign, official counts matched. File 6 alone has \`uid\`. All nine attack-family time windows are non-overlapping. Consecutive processed file timestamp boundaries meet exactly, so treating filenames as independent capture units is invalid without further proof.
2. Ground-truth source: 18 attack-only CSVs, **17,801,188 rows as stored**. Whole-file GT 15/16 duplication confirmed by SHA-256. Ground-truth class counts differ sharply from processed counts, especially DoS, ransomware, password and backdoor. One backwards timestamp transition was observed in GT 1.
3. Internal reconciliation: deterministic per-GT-file/type reservoir of 512 rows in 27 strata yielded 13,824/13,824 six-field flow-key and attack-label matches, 12,800 distinct sampled keys. **Not independently verified attack labels or evidence of multiple independent executions.**

These are source-derived observations, not newly invented data. See \`RECKONING_TONIOT_FULL_METADATA_RESULTS.md\` and \`RECKONING_TONIOT_GROUNDTRUTH_RESULTS.md\`.

## New diagnostic

Script: \`scripts/17_audit_toniot_validity.py\`. Synthetic tests: \`tests/test_toniot_validity_audit.py\`.

This is **one complete read-only pass through the 23 processed files**; it stores aggregate timestamp-bin counts, exact class-specific counts for whether source or destination lies in the provider-documented 192.168.159.30–39 attack-origin IP range, and bounded deterministic per-class reservoirs (default maximum 4,096 records per class) of endpoint five-tuples. The provider README is the source of the range, not something inferred from labels.

- **Temporal separation:** Report attack-family min/max windows, 1-hour diagnostic timestamp bins, co-occurring benign traffic per attack-family bin, and order reversals. Detect globally nonoverlapping attack-family intervals.
- **Identity/label association:** Report complete-source fractions of rows whose source/destination IP falls in the publicly documented attacker address range. Compute *sample-based* per-class source/destination address diversity, five-tuple diversity, and source-IP overlap with benign and other attack-family samples. A strong association motivates proxy-leakage controls, but does not itself prove a particular predictor is invalid or a result biased. Sample measures are not prevalence-adjusted.
- **Episode proxies:** Count gaps of at least 300, 1,800 or 3,600 seconds between consecutive records of the **same family** in input order. Gaps are diagnostic choices only, **not** independent attacks or segmentation thresholds. Absence of gaps does not establish one episode, and their presence does not establish multiple independent attack executions. Provider's flow rows do not supply an authenticated attack campaign ID or execution diary.
- **Source integrity:** Default expected file count = 23; row count = 22,339,021. Fail without new report if different, malformed, invalid timestamp or contradictory binary/type label. Entire source read; no processed rows written.
- **Privacy:** No row-level flows, learned features, source/destination addresses, identity hashes or sampled keys in output. The documented public attacker-origin range may appear by name in the audit metadata.

This new audit **does not** classify, learn, fit, train, select features, create partitions, verify independent labels, reconstruct original event instructions, or certify LOAO eligibility. The GT rows are not rejoined: prior bounded sample already established strong *internal consistency*, while the missing requirement is independent source provenance.

## Local execution (PowerShell, VERA-IDS root)

\`\`\`powershell
git status --short
git switch post-comnet-archival-value
git fetch origin
git merge --ff-only origin/post-comnet-archival-value
uv run python -m pytest -p no:cacheprovider
uv run python scripts/17_audit_toniot_validity.py --processed-root "data/raw/ToN_IoT/Processed_Network_dataset" --out "outputs/17_revision_campaign/toniot/targeted_validity_audit.json"
\`\`\`

If switching or fast-forward fails, do not force. If pytest fails, stop and provide output before data scanning. On success, upload \`targeted_validity_audit.json\`. It contains approximately one week's hourly aggregate counts and per-class identity/temporal diagnostics; it does not include user-specific source filesystem paths.

## Decision gates after the evidence

- **Independent unit gate:** Do the source, official collection records or accompanying logs establish multiple genuinely independent attack executions/capture units for relevant families? Time gaps, source-IP changes and chunk numbers are **insufficient alone**. If not, this criterion remains unmet.
- **Support gate:** Any proposed independent partition must satisfy the frozen Protocol B train/val/test per-family support criteria without changing them to admit TON_IoT. Distinguish single-episode row support from independent event support.
- **Leakage gate:** Exclusions, ablations and handling of timestamp/identity/proxies must be predeclared; do not infer safe predictor sets from this association audit alone. Do not tune the design on test outcomes.
- **Outcome decision reserved to user:** choose whether to conduct further provenance acquisition, retain TON_IoT in a carefully qualified Protocol A or external-stress lane, or choose an alternative candidate with better independent acquisition evidence. Do **not** silently promote this source as a third primary Protocol B LOAO dataset.

The historical \`release/v2026.10-comnet/\` is immutable, and the draft PR remains unmerged pending user-local tests.

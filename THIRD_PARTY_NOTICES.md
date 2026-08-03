# Third-Party Notices

No third-party source code or raw dataset is intentionally bundled in this
release. Python dependencies are installed separately and remain governed by
their upstream licenses.

The research uses or discusses CICIDS2017, CICIoT2023, NSL-KDD, and UNSW-NB15.
Their official acquisition pages and citation/rights caveats are listed in
`DATASETS.md`. Excluding data from Git does not remove citation obligations.

The configuration includes literature-derived reference profiles named
`adewole2025_xgb` and `neto2023_rf`. Complete bibliographic records and a
parameter-to-publication verification were not present in the declared source.
Do not describe these profiles as exact reproductions until citations and method
matching are independently confirmed.

The package-local workflow modules are original project implementations by Ali
Farrokhnejad. They call the separately installed libraries listed in the locked
environment; no dependency source code is vendored.

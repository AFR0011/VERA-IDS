# Errata and Interpretation Notes

## E-001 — Protocol A system macro-F1 denominator

Resolved in `v2026.08`. Protocol A now averages system per-class F1 over labels
with genuine ground-truth support. The earlier declared-output-label value is
retained as an explicitly historical column. See
[PROTOCOL_A_CORRECTION.md](PROTOCOL_A_CORRECTION.md).

## E-002 — External UNSW-NB15 Protocol B

The recorded setup has zero benign validation/test support. It is invalid for
Protocol B comparison and is retained only as a diagnostic. It is not used in
rankings, figures, or positive result claims.

## E-003 — Rejection-policy scope

The exploratory broad threshold grid and the validation-selected rejection
policy are different experimental lanes. Neither is a universal canonical
policy; configurations and output roots are intentionally distinct.

## E-004 — Model availability

The approximately 63.9 GB historical model collection is not published. Full
workflows train models from provider-acquired data; byte-identical reproduction
from downloadable models is not claimed.

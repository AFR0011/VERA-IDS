"""Locations for local C03 tools; no data access or working-directory changes."""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
POLICY = TOOLS / "C03_APPROVED_SELECTOR_POLICY.json"
LOCAL = ROOT / ".release-audit" / "c03"
ORIGINAL_SNAPSHOT = LOCAL / "C03_input_content_manifest.json"
RELOCATION = TOOLS / "RELOCATION.json"

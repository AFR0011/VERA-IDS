"""Remove stale Office metadata from an already accepted, comment-free DOCX."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
EP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

TITLE = (
    "Beyond Closed-Set Accuracy: A Validity-Aware Evaluation Framework for "
    "Machine Learning-Based Intrusion Detection"
)


def _core_properties() -> bytes:
    ET.register_namespace("cp", CP)
    ET.register_namespace("dc", DC)
    root = ET.Element(f"{{{CP}}}coreProperties")
    ET.SubElement(root, f"{{{DC}}}title").text = TITLE
    ET.SubElement(root, f"{{{DC}}}subject").text = "Repository-edition thesis source for VERA-IDS v2026.08"
    ET.SubElement(root, f"{{{DC}}}creator")
    ET.SubElement(root, f"{{{CP}}}lastModifiedBy")
    ET.SubElement(root, f"{{{DC}}}description").text = (
        "Cleaned temporary source used to render the CC BY 4.0 repository edition."
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _app_properties() -> bytes:
    ET.register_namespace("", EP)
    root = ET.Element(f"{{{EP}}}Properties")
    ET.SubElement(root, f"{{{EP}}}Application").text = "VERA-IDS release pipeline"
    ET.SubElement(root, f"{{{EP}}}AppVersion").text = "2026.8"
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def sanitize(source: Path, output: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        names = set(archive.namelist())
        comment_parts = {
            name
            for name in names
            if name.startswith("word/")
            and ("comment" in name.lower() or name.lower().endswith("people.xml"))
        }
        if comment_parts:
            raise ValueError(f"comment/person parts remain: {sorted(comment_parts)}")

        tracked_tags = {
            f"{{{W}}}ins",
            f"{{{W}}}del",
            f"{{{W}}}moveFrom",
            f"{{{W}}}moveTo",
        }
        for name in names:
            if not name.startswith("word/") or not name.endswith(".xml"):
                continue
            root = ET.fromstring(archive.read(name))
            if any(element.tag in tracked_tags for element in root.iter()):
                raise ValueError(f"tracked revision markup remains in {name}")

        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as cleaned:
            for info in archive.infolist():
                if info.filename == "docProps/core.xml":
                    data = _core_properties()
                elif info.filename == "docProps/app.xml":
                    data = _app_properties()
                else:
                    data = archive.read(info.filename)
                cleaned.writestr(info, data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sanitize(args.source.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

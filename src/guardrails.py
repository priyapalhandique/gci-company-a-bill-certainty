"""Executable guardrails for the GCI final assignment.

Run directly for a source-integrity check, or import from the notebook.
Any failure raises ``GuardrailError`` so a bad build cannot be submitted.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from config import (
    CLIENT_CSV,
    ID_COL,
    MAX_PDF_BYTES,
    MAX_SLIDES,
    RECORD_CSV,
    SENSITIVE_FIELDS,
    SOURCE_HASHES,
    TARGET,
    WORK,
)


class GuardrailError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(strict: bool = True, require_originals: bool = False) -> dict:
    """Verify original files where present and the working copies everywhere.

    The notebook is portable: it can run on Google Colab, where the original
    course folder does not exist. In that environment the working copies are
    still checked byte-for-byte against the SHA-256 recorded at build time, so
    the data actually analysed cannot drift. ``require_originals=True`` keeps
    the stricter local-build behaviour.
    """
    manifest = json.loads(SOURCE_HASHES.read_text(encoding="utf-8"))
    results = {}
    present = 0
    missing = []
    for name, entry in manifest.items():
        path = Path(entry["path"])
        if not path.exists():
            missing.append(name)
            results[name] = {"ok": None, "status": "not_present", "path": str(path)}
            continue
        digest = _sha256(path)
        ok = digest == entry["sha256"]
        results[name] = {
            "ok": ok,
            "status": "verified" if ok else "mismatch",
            "sha256": digest,
            "path": str(path),
        }
        present += 1
        if strict and not ok:
            raise GuardrailError(f"Source hash changed: {name} ({path})")
    if require_originals and missing:
        raise GuardrailError(f"Missing source files: {missing}")

    copies = verify_copies()
    results["_summary"] = {
        "originals_recorded": len(manifest),
        "originals_present": present,
        "originals_missing": missing,
        "copies": copies,
    }
    return results


def verify_copies() -> dict:
    """The working copies must match the originals at build start."""
    manifest = json.loads(SOURCE_HASHES.read_text(encoding="utf-8"))
    pairs = {
        "client_csv": (CLIENT_CSV, manifest["client_csv"]["sha256"]),
        "record_csv": (RECORD_CSV, manifest["record_csv"]["sha256"]),
    }
    out = {}
    for name, (path, expected) in pairs.items():
        if not path.exists():
            raise GuardrailError(f"Missing working copy: {path}")
        digest = _sha256(path)
        out[name] = digest == expected
        if digest != expected:
            raise GuardrailError(f"Working copy does not match source: {path}")
    return out


def assert_under_work(path: str | Path) -> Path:
    p = Path(path).resolve()
    if WORK.resolve() not in p.parents and p != WORK.resolve():
        raise GuardrailError(f"Refusing to write outside work/: {p}")
    return p


def set_seeds(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)


def assert_dataset(df, expected_rows: int = 100_000) -> dict:
    checks = {}
    checks["rows"] = len(df) == expected_rows
    checks["unique_customers"] = df[ID_COL].is_unique
    checks["churn_binary"] = set(df[TARGET].dropna().unique()) <= {0, 1}
    checks["churn_count"] = int(df[TARGET].sum())
    failed = [k for k, v in checks.items() if v is not True and k != "churn_count"]
    if failed:
        raise GuardrailError(f"Dataset assertions failed: {failed}; checks={checks}")
    return checks


def assert_no_target_leak(features) -> None:
    banned = {TARGET, ID_COL}
    leaked = banned.intersection(set(features))
    if leaked:
        raise GuardrailError(f"Target/ID leakage in feature set: {sorted(leaked)}")


def assert_fair_features(features) -> None:
    leaked = set(SENSITIVE_FIELDS).intersection(set(features))
    if leaked:
        raise GuardrailError(f"Sensitive fields in deployed feature set: {sorted(leaked)}")


def check_slide_pdf(pdf_path: str | Path, expected_name: str) -> dict:
    """Validate page count, size and filename. Requires pypdf."""
    from pypdf import PdfReader

    p = Path(pdf_path)
    if p.name != expected_name:
        raise GuardrailError(f"PDF filename must be {expected_name}, got {p.name}")
    if p.stat().st_size > MAX_PDF_BYTES:
        raise GuardrailError(f"PDF exceeds 10MB: {p.stat().st_size} bytes")
    pages = len(PdfReader(str(p)).pages)
    if pages > MAX_SLIDES:
        raise GuardrailError(f"PDF has {pages} pages; max {MAX_SLIDES}")
    return {"pages": pages, "bytes": p.stat().st_size, "name": p.name}


def check_citations(slide_text: str, reference_numbers: set[int]) -> dict:
    cited = {int(x) for x in re.findall(r"\[(\d{1,2})\]", slide_text)}
    missing = sorted(cited - reference_numbers)
    uncited = sorted(reference_numbers - cited)
    if missing:
        raise GuardrailError(f"Citations missing from references: {missing}")
    return {"cited": sorted(cited), "uncited": uncited}


def main() -> None:
    src = verify_sources()
    summary = src["_summary"]
    print(
        "Original files verified: "
        f"{summary['originals_present']}/{summary['originals_recorded']}"
    )
    if summary["originals_missing"]:
        print("Not present in this environment:", ", ".join(summary["originals_missing"]))
    print(f"Working copies verified: {summary['copies']}")
    print("GUARDRAIL CHECK: PASS")


if __name__ == "__main__":
    main()

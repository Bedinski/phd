"""Persistence layer: run-id minting, artifact dirs, write/read round-trip."""

from __future__ import annotations

from datetime import date, datetime, timezone

from phd.models import (
    Claim,
    Confidence,
    ResearchFindings,
    Source,
)
from phd.persistence import (
    STAGE_DIRS,
    append_transcript,
    ensure_run_layout,
    mint_run_id,
    read_artifact,
    stage_dir,
    write_artifact,
)


def test_mint_run_id_format() -> None:
    rid = mint_run_id()
    # YYYYMMDD-HHMMSS-XXXXXX (6 hex chars)
    parts = rid.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 8 and parts[0].isdigit()
    assert len(parts[1]) == 6 and parts[1].isdigit()
    assert len(parts[2]) == 6


def test_run_layout_creates_all_stage_dirs(tmp_path) -> None:
    rid = mint_run_id()
    base = ensure_run_layout(rid, root=tmp_path)
    assert base == tmp_path / rid
    for stage_subdir in STAGE_DIRS.values():
        assert (base / stage_subdir).is_dir()


def test_write_and_read_artifact(tmp_path) -> None:
    rid = mint_run_id()
    ensure_run_layout(rid, root=tmp_path)
    findings = ResearchFindings(
        run_id=rid,
        thesis="t",
        universe="u",
        strategy_type="sx",
        research_questions=["q?"],
        claims=[
            Claim(
                id="C001",
                statement="A statement that is reasonably specific and cited.",
                sources=[
                    Source(
                        url="https://example.com/x",
                        title="X",
                        publisher="Pub",
                        accessed_at=date(2026, 5, 25),
                    )
                ],
                confidence=Confidence.HIGH,
                relevance_to_thesis="x" * 25,
            )
        ],
        generated_at=datetime(2026, 5, 25, 12, tzinfo=timezone.utc),
    )
    write_artifact(rid, "research", "findings.json", findings, root=tmp_path)
    loaded = read_artifact(rid, "research", "findings.json", ResearchFindings, root=tmp_path)
    assert loaded.claims[0].id == "C001"


def test_transcript_append(tmp_path) -> None:
    rid = mint_run_id()
    ensure_run_layout(rid, root=tmp_path)
    append_transcript(rid, "review", {"type": "assistant", "blocks": []}, root=tmp_path)
    append_transcript(rid, "review", {"type": "result", "subtype": "success"}, root=tmp_path)
    path = stage_dir(rid, "review", root=tmp_path) / "transcript.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 2

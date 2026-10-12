"""A machine's own manual (HARM-S7-T11)."""

from __future__ import annotations

import io

import numpy as np
import pytest

from api import machines, manuals
from rag.ingest import parse_generic

GB7_MANUAL = """# Gearbox GB-7 service notes

## Pinion inspection
Check the 29-tooth pinion for pitting and root cracks every 2000 hours. A
cracked tooth shows as impulsive vibration and strong mesh sidebands; stop the
line conveyor before opening the cover.

## Lubrication
GB-7 runs on a synthetic PAO gear oil, ISO VG 320, changed every 8000 hours.

## Contacts
Maintenance planner: line 3 office.
"""


def test_a_manual_is_cut_into_citable_sections():
    chunks = parse_generic(GB7_MANUAL, "GB-7-OM")
    assert [c.id for c in chunks] == ["GB-7-OM-1", "GB-7-OM-2", "GB-7-OM-3"]
    assert chunks[0].title == "Pinion inspection" and "29-tooth" in chunks[0].text
    long = parse_generic("## One\n" + " ".join(["word"] * 600), "X-OM")
    assert [c.section for c in long] == ["1.1", "1.2", "1.3"]
    plain = parse_generic("first paragraph\n\nsecond paragraph", "P-OM")
    assert len(plain) == 1 and plain[0].section == "1"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(machines, "DATA_DIR", tmp_path)
    monkeypatch.setattr(machines, "MACHINES_DIR", tmp_path / "machines")
    monkeypatch.setattr(machines, "UPLOADS_DIR", tmp_path / "uploads")
    return tmp_path


def _gearbox(name: str = "Gearbox GB-7") -> machines.Machine:
    uid = machines.new_upload({"name": "x", "sample_rate_hz": 25600, "unit": "g"})
    buf = io.BytesIO()
    np.save(buf, np.random.default_rng(1).normal(size=(3, 8192)).astype(np.float32))
    machines.save_upload_file(uid, "run.npy", buf.getvalue())
    return machines.create({
        "name": name, "nominal_rpm": 1480, "vib_unit": "g", "iso_class": "none",
        "family": "gearbox", "params": {"teeth": 29},
        "bearing": {"pitch_mm": 46, "ball_mm": 11.1, "n_balls": 9, "contact_angle_deg": 0},
        "components": ["gear", "bearing", "shaft"], "recording": f"upload:{uid}",
        "channel": 0, "baseline_h": (0.0, 1.0)})


def test_an_upload_is_checked(data_dir):
    m = _gearbox()
    with pytest.raises(machines.CommissionError, match="Markdown or plain text"):
        manuals.save(m.id, "manual.pdf", b"%PDF")
    with pytest.raises(machines.CommissionError, match="UTF-8"):
        manuals.save(m.id, "manual.md", b"\xff\xfe\x00")
    with pytest.raises(machines.CommissionError, match="no machine"):
        manuals.save("nobody", "manual.md", b"# x\ntext")


def test_a_machine_s_manual_is_cited_for_it_and_never_for_another(data_dir, tmp_path):
    from kg.load import build_graph
    from kg.store import NetworkXStore
    from rag.pipeline import Pipeline

    m = _gearbox()
    saved = manuals.save(m.id, "gb7.md", GB7_MANUAL.encode())
    assert saved["sections"] == 3 and manuals.info(m.id)["doc_id"] == saved["doc_id"]
    doc = saved["doc_id"]
    assert [c.doc for c in manuals.load_all()] == [doc] * 3

    store = NetworkXStore(tmp_path / "g.json", autoload=False)
    store.merge_graph(build_graph())
    store.merge_graph(machines.kg_graph(m))
    assert store.node(m.equipment_id).props["manuals"] == f"{doc},GB-201-OM"
    p = Pipeline(store=store, llm=None, audit_path=tmp_path / "audit.jsonl")
    p.add_chunks(saved["chunks"])

    def event(equipment: str) -> dict:
        return {"equipment_id": equipment, "state": "WARNING", "hhi": 0.7, "rul_hours": 90,
                "top_classes": [{"name": "broken_tooth", "score": 0.8}],
                "top_surprised_features": [
                    {"kind": "HARMONIC_FEAT_VIB_ORDER:mesh_sb", "direction": "up", "weight": 1},
                    {"kind": "HARMONIC_FEAT_VIB_KURTOSIS", "direction": "up", "weight": 0.8}]}

    own = p.diagnose(event(m.equipment_id))
    assert any(h.chunk_id.startswith(doc) for h in own.hits), [h.chunk_id for h in own.hits]
    assert all(h.chunk_id.startswith((doc, "GB-201-OM")) for h in own.hits)
    other = p.diagnose(event("GB-201"))
    assert not any(h.chunk_id.startswith(doc) for h in other.hits)
    assert doc not in " ".join(other.response.get("references", []))

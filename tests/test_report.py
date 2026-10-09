"""The README generator and the resume-bullet facts must never read results/dev/."""
import builtins
import io
import json
from pathlib import Path

from pawmatch import report

DEV, REAL = 0.987654, 0.123456


def _result(v):
    m = {"value": v, "ci95": [v, v]}
    return {"model": "x", "closed_set": {"rank1": m, "rank5": m}, "map_leave_one_out": m,
            "verification": {"auc_clamped_as_paper": m}}


def test_report_never_reads_results_dev(tmp_path, monkeypatch):
    (tmp_path / "results" / "dev").mkdir(parents=True)
    (tmp_path / "results" / "dev" / "m1_dev.json").write_text(json.dumps(_result(DEV)))
    (tmp_path / "results" / "m1_real.json").write_text(json.dumps(_result(REAL)))
    (tmp_path / "README.md").write_text("<!-- m1:start -->\n<!-- m1:end -->\n<!-- m4:start -->\n<!-- m4:end -->\n")
    monkeypatch.chdir(tmp_path)

    real_open = io.open

    def guarded_open(file, *args, **kwargs):
        assert "dev" not in Path(file).resolve().relative_to(tmp_path).parts, f"read {file}"
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)

    report.readme()
    readme = Path("README.md").read_text()
    assert "12.35%" in readme and "98.77" not in readme

    facts = "\n".join(report.facts())
    assert str(REAL) in facts and str(DEV) not in facts


def test_external_figures_are_never_offered_as_results(tmp_path):
    (tmp_path / "external.json").write_text(json.dumps({"shelter_intakes_2025": {"value": 5800000}}))
    assert report.facts(tmp_path) == []

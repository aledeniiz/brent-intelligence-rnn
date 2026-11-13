import csv
import json
import pytest
from scripts.export_results import export

def test_reporting_respects_completed_full_run(tmp_path):
    root=tmp_path/"artifacts"
    reports=tmp_path/"reports"
    run=root/"runs"/"example"
    run.mkdir(parents=True)
    (root/"active.json").write_text(json.dumps({"run":"runs/example"}))
    metrics={"mae":2.,"mse":8.,"rmse":2.828427,"naive_rmse":3.,"relative_rmse":2.828427/3.}
    report={
        "mode":"historical-evaluation",
        "folds":[{"fold":i} for i in range(5)],
        "selection":{"architecture":"gru"},
        "data_sha256":"abc",
        "last_data_date":"2026-01-01",
        "config":{"horizons":[5,63,252]},
        "test_period":{"n_anchors":10},
        "uncertainty":{"test_coverage":{"5":0.8,"63":0.7,"252":0.6}},
        "models":{kind:{"test":{str(h):metrics for h in (5,63,252)},
                         "parameter_count":42,"validation_score":1.0,
                         "folds":[{"seconds":1.0}],"refit":{"seconds":0.5}}
                  for kind in ("rnn","lstm","gru")}
    }
    (run/"summary.json").write_text(json.dumps(report))
    result=export(root,reports)
    with open(result["table"],newline="",encoding="utf-8") as fh:
        rows=list(csv.DictReader(fh))
    assert len(rows)==9
    assert sum(row["seleccionado"]=="True" for row in rows)==3
    assert (reports/"presentation_facts.md").exists()

def test_reporting_rejects_smoke(tmp_path):
    root=tmp_path/"artifacts"
    (root/"active.json").parent.mkdir(parents=True)
    (root/"active.json").write_text(json.dumps({"run":"smoke/fake"}))
    folder=root/"smoke"/"fake"
    folder.mkdir(parents=True)
    (folder/"summary.json").write_text(json.dumps({"mode":"synthetic-smoke-or-debug"}))
    with pytest.raises(ValueError):
        export(root,tmp_path/"reports")

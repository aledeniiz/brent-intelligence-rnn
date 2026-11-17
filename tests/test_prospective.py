from copy import deepcopy
import json

import pandas as pd
import pytest

from brent_intelligence.frozen import file_hash
from brent_intelligence.prospective import archive_forecast, evaluate_prospective


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "prices.csv"
    dates = pd.bdate_range("2020-01-01", periods=300)
    frame = pd.DataFrame({"Date": dates, "brent": [100.+i for i in range(300)]})
    frame.to_csv(data, index=False)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "production.pt").write_bytes(b"unchanged-existing-weights")
    (bundle / "preprocessing.json").write_text('{}', encoding="utf8")
    (bundle / "manifest.json").write_text('{}', encoding="utf8")
    forecast = {"format_version": 1, "generated_utc": "2020-01-02T10:00:00+00:00",
                "last_date": "2020-01-01", "last_price": 100.,
                "model": {"weights_sha256": file_hash(bundle / "production.pt"),
                          "trained_data_end": "2019-12-31", "architecture": "lstm"},
                "horizons": [5, 63, 252], "prices": [104., 160., 350.],
                "lower": [90., 120., 300.], "upper": [110., 180., 400.]}
    return data, tmp_path / "journal", bundle, forecast


def live(setup, **kwargs):
    _, directory, bundle, forecast = setup
    return archive_forecast(forecast, directory=directory, bundle=bundle,
                            provenance="live_issue", now="2020-01-02T12:00:00Z", **kwargs)


def evaluate(setup, as_of="2020-01-09T12:00:00Z", **kwargs):
    data, directory, _, _ = setup
    return evaluate_prospective(data=data, directory=directory, output=None, as_of=as_of, **kwargs)


@pytest.mark.parametrize("provenance", ["legacy_import", "historical_reconstruction"])
def test_legacy_payload_is_preserved_and_cannot_be_promoted(setup, provenance):
    _, directory, bundle, forecast = setup
    original = deepcopy(forecast)
    prior = archive_forecast(forecast, directory=directory, provenance=provenance,
                             now="2020-01-02T12:00:00Z")
    path = next((directory / "forecasts").glob("*/*.json"))
    content = path.read_bytes()
    again = live(setup)
    assert again == prior and path.read_bytes() == content and forecast == original
    result = evaluate(setup, "2022-01-01T00:00:00Z")
    assert result["evaluated_points"] == 0
    assert {d["status"] for d in result["details"]} == {"excluded_nonprospective"}


def test_only_mature_observed_sessions_are_scored_against_matched_naive(setup):
    live(setup)
    early = evaluate(setup, "2020-01-07T12:00:00Z")
    assert early["evaluated_points"] == 0
    result = evaluate(setup)
    assert result["evaluated_points"] == 1
    group = next(g for g in result["groups"] if g["horizon"] == 5)
    assert group["n"] == 1 and group["mae"] == group["rmse"] == 1
    assert group["naive_mae"] == group["naive_rmse"] == 5
    assert group["improvement_pct"] == 80
    mature = next(d for d in result["details"] if d["status"] == "matured")
    assert mature["target_date"] == "2020-01-08"  # fifth quotation, not Jan 6
    assert all(g["rmse"] is None for g in result["groups"] if g["horizon"] != 5)


def test_late_registration_is_not_retrospectively_prospective(setup):
    _, directory, bundle, forecast = setup
    archive_forecast(forecast, directory=directory, bundle=bundle,
                     provenance="live_issue", now="2020-02-01T12:00:00Z")
    result = evaluate(setup, "2020-02-01T13:00:00Z")
    assert result["evaluated_points"] == 0
    assert next(d for d in result["details"] if d["horizon"] == 5)["status"] == "excluded_target_known_before_recording"


def test_revisions_are_archived_but_only_first_issue_is_scored(setup):
    first = live(setup)
    _, directory, bundle, forecast = setup
    forecast = deepcopy(forecast)
    forecast["generated_utc"] = "2020-01-03T10:00:00Z"
    forecast["prices"][0] = 108.
    archive_forecast(forecast, directory=directory, bundle=bundle,
                     provenance="live_issue", now="2020-01-03T12:00:00Z")
    result = evaluate(setup)
    assert result["forecast_records"] == 2 and result["evaluated_points"] == 1
    mature = next(d for d in result["details"] if d["status"] == "matured")
    assert mature["forecast_id"] == first["forecast_id"] and mature["prediction"] == 104.
    assert sum(d["status"] == "excluded_later_revision" for d in result["details"]) == 3


def test_first_observed_labels_survive_revised_prices_and_missing_calendar_rows(setup):
    live(setup)
    original = evaluate(setup)
    data, directory, _, _ = setup
    label = next((directory / "labels").glob("*.json"))
    original_bytes = label.read_bytes()
    frame = pd.read_csv(data)
    frame.loc[5, "brent"] = 999.
    frame = frame.drop(index=0)  # the original origin can even disappear later
    frame.to_csv(data, index=False)
    revised = evaluate(setup)
    assert revised["groups"] == original["groups"]
    assert revised["warnings"] and label.read_bytes() == original_bytes


def test_future_records_and_future_label_observations_do_not_leak_backward(setup):
    live(setup)
    assert evaluate(setup, "2020-01-02T11:00:00Z")["evaluated_points"] == 0
    evaluate(setup, "2020-01-15T12:00:00Z")
    result = evaluate(setup)  # Jan 9 must not use label first registered Jan 15
    assert result["evaluated_points"] == 0
    assert next(d for d in result["details"] if d["horizon"] == 5)["status"] == "pending_label_not_observed_as_of"


def test_tampered_records_fail_before_replacing_evaluation(setup, tmp_path):
    live(setup)
    data, directory, _, _ = setup
    report = tmp_path / "evaluation.json"
    report.write_bytes(b"prior-valid-report")
    path = next((directory / "forecasts").glob("*/*.json"))
    altered = json.loads(path.read_text(encoding="utf8"))
    altered["forecast"]["prices"][0] = 105.
    path.write_text(json.dumps(altered), encoding="utf8")
    with pytest.raises(ValueError, match="alterado"):
        evaluate_prospective(data=data, directory=directory, output=report)
    assert report.read_bytes() == b"prior-valid-report"


def test_complete_model_version_separates_same_weights_with_new_preprocessing(setup):
    first = live(setup)
    _, directory, bundle, forecast = setup
    (bundle / "preprocessing.json").write_text('{"new_scaler": true}', encoding="utf8")
    revised = deepcopy(forecast)
    revised["generated_utc"] = "2020-01-03T10:00:00Z"
    second = archive_forecast(revised, directory=directory, bundle=bundle,
                              provenance="live_issue", now="2020-01-03T12:00:00Z")
    assert second["model_version"] != first["model_version"]
    assert evaluate(setup)["evaluated_points"] == 2


def test_invalid_forecasts_and_training_after_origin_are_not_scored(setup):
    _, directory, bundle, forecast = setup
    with pytest.raises(ValueError, match="posterior"):
        archive_forecast(forecast, directory=directory, now="2020-01-01T00:00:00Z")
    invalid = deepcopy(forecast)
    invalid["prices"][0] = float('nan')
    with pytest.raises(ValueError, match="invalidas"):
        archive_forecast(invalid, directory=directory, now="2020-01-02T12:00:00Z")
    invalid = deepcopy(forecast)
    invalid["last_date"] = "2020-01-03"
    with pytest.raises(ValueError, match="Origen futuro"):
        archive_forecast(invalid, directory=directory, bundle=bundle,
                         provenance="live_issue", now="2020-01-02T12:00:00Z")
    forecast["model"]["trained_data_end"] = "2020-01-03"
    live(setup)
    result = evaluate(setup)
    assert result["evaluated_points"] == 0
    assert all(d["status"] == "excluded_training_after_origin" for d in result["details"])


def test_aggregation_uses_multiple_origins_and_squared_errors(setup):
    live(setup)
    _, directory, bundle, forecast = setup
    other = deepcopy(forecast)
    other.update(last_date="2020-01-02", last_price=101., generated_utc="2020-01-03T10:00:00Z")
    other["prices"][0] = 109.  # target 106, error +3; first issue has error -1
    archive_forecast(other, directory=directory, bundle=bundle,
                     provenance="live_issue", now="2020-01-03T12:00:00Z")
    result = evaluate(setup, "2020-01-10T12:00:00Z")
    group = next(g for g in result["groups"] if g["horizon"] == 5)
    assert group["n"] == 2 and group["mae"] == 2
    assert group["rmse"] == pytest.approx(5**.5)
    assert group["naive_mae"] == group["naive_rmse"] == 5


def test_repeated_evaluation_is_idempotent_for_forecasts_labels_and_scores(setup):
    live(setup)
    first = evaluate(setup)
    _, directory, _, _ = setup
    files = {p: p.read_bytes() for p in directory.rglob("*.json")}
    again = evaluate(setup, "2020-01-09T13:00:00Z")
    assert again["evaluated_points"] == first["evaluated_points"] == 1
    assert again["groups"] == first["groups"]
    assert {p: p.read_bytes() for p in directory.rglob("*.json")} == files

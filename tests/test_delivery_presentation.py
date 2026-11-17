"""Exactly five presentation slides from the SAME v4 evaluated model."""
from pathlib import Path
import pandas as pd
from pptx import Presentation
from scripts.build_delivery_presentation import build_delivery_deck


def test_five_slides_match_v4_architecture_and_forecasts(tmp_path):
    stats = {
        str(h): {"rmse":rmse,"improvement_pct":imp}
        for h,rmse,imp in [(5,6.04,0.),(63,19.93,.20),(252,21.14,18.26)]
    }
    summary={
        "mode":"complete-delivery-candidates",
        "selected_architecture":"rnn",
        "full_data_models":{"rnn":{"architecture":"rnn"}},
        "historical_test_diagnostic":{
            "models":{arch:{"metrics":stats} for arch in ("rnn","lstm","gru")}
        },
    }
    forecast={
        "latest_observed_date":"2026-11-11",
        "last_observed_price":92.14,
        "predictions":{"5":92.14,"63":91.94,"252":78.27}
    }
    path=build_delivery_deck(summary,forecast,pd.Series([72.,80.,60.,92.14]),
                             tmp_path/"deck.pptx",
                             uncertainty={5:(85.,99.),63:(72.,110.),252:(52.,120.)})
    prs=Presentation(path)
    assert len(prs.slides)==5
    texts=[" ".join(shape.text for shape in slide.shapes if shape.has_text_frame)
           for slide in prs.slides]
    assert "RNN" in texts[2] and "LSTM" in texts[2] and "GRU" in texts[2]
    assert "21.14" in texts[2]
    assert "92.14" in texts[3] and "78.27" in texts[3]
    assert "80%" in texts[3] and "85.00" in texts[3]

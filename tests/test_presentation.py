from pathlib import Path
import pandas as pd
from pptx import Presentation
from scripts.build_presentation import build_deck

def test_business_presentation_has_exactly_five_slides(tmp_path):
    metrics={"rmse":3.2,"relative_rmse":1.05}
    summary={
        "config":{"horizons":[5,63,252]},
        "selection":{"architecture":"gru"},
        "models":{kind:{"test":{str(h):metrics for h in (5,63,252)}}
                  for kind in ("rnn","lstm","gru")}
    }
    latest={"last_date":"2026-11-11","last_price":75.0,
            "prices":[76.0,77.0,78.0],
            "lower":[72.0,68.0,60.0],
            "upper":[80.0,86.0,95.0]}
    data=pd.DataFrame({"brent":[55.,100.,75.]})
    output=tmp_path/"presentation.pptx"
    result=build_deck(summary,latest,data,output)
    assert result.is_file()
    deck=Presentation(result)
    assert len(deck.slides)==5
    assert all(len(slide.shapes)>5 for slide in deck.slides)

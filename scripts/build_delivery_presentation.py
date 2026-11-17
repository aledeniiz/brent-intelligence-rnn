"""Five-slide executive presentation from the SAME v4 candidate as app_delivery.py.

No training. Historical metrics describe pre-2024 fitted evaluation models;
today's forecast comes from the separately refitted candidate.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from brent_intelligence.config import DATA_PATH, ARTIFACTS_DIR, REPORTS_DIR
from brent_intelligence.delivery_inference import forecast_candidate
from brent_intelligence.features import load_prices

NAVY=RGBColor(10,18,31)
PANEL=RGBColor(25,39,59)
CYAN=RGBColor(88,220,206)
WHITE=RGBColor(245,248,251)
MUTED=RGBColor(181,194,208)
AMBER=RGBColor(241,177,94)


def text(slide, value, x,y,w,h,size=16,bold=False,color=WHITE):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=shape.text_frame
    tf.word_wrap=True
    p=tf.paragraphs[0]
    p.text=str(value)
    p.font.name="Aptos"
    p.font.size=Pt(size)
    p.font.bold=bold
    p.font.color.rgb=color
    p.alignment=PP_ALIGN.LEFT
    return shape


def panel(slide,x,y,w,h):
    rect=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),Inches(y),Inches(w),Inches(h))
    rect.fill.solid()
    rect.fill.fore_color.rgb=PANEL
    rect.line.fill.background()
    return rect


def slide_new(prs,idx,title,subtitle):
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb=NAVY
    text(slide,"BRENT INTELLIGENCE · CASO DE USO III",.7,.35,10,.35,12,True,CYAN)
    text(slide,title,.7,.85,12,.63,29,True)
    text(slide,subtitle,.7,1.56,11.95,.6,13,False,MUTED)
    text(slide,f"Aprendizaje automático · {idx} / 5",.7,7.10,10,.25,10,False,MUTED)
    return slide


def metric_card(slide,x,y,w,label,value,note,highlight=False):
    panel(slide,x,y,w,1.65)
    text(slide,label,x+.2,y+.16,w-.4,.35,12,False,MUTED)
    text(slide,value,x+.2,y+.54,w-.4,.55,24,True,CYAN if highlight else WHITE)
    text(slide,note,x+.2,y+1.17,w-.4,.36,11,False,MUTED)


def build_delivery_deck(summary, forecast, price_series, output, uncertainty=None):
    if summary.get("mode")!="complete-delivery-candidates":
        raise ValueError("Se requiere summary.json de entrenamiento completo v4")
    selected=summary["selected_architecture"]
    comparison=summary["historical_test_diagnostic"]["models"]
    if selected not in comparison or selected not in summary["full_data_models"]:
        raise ValueError("Modelo seleccionado no esta en artefactos")
    horizons=(5,63,252)
    if set(forecast["predictions"])!={"5","63","252"}:
        raise ValueError("Se requieren los tres horizontes 5/63/252")
    prs=Presentation()
    prs.slide_width=Inches(13.333)
    prs.slide_height=Inches(7.5)
    base=float(forecast["last_observed_price"])

    slide=slide_new(prs,1,"La volatilidad del crudo exige planificar",
       "Pronóstico como apoyo a compras y coberturas energéticas; no garantía de rentabilidad.")
    metric_card(slide,.74,2.4,3.75,"Brent histórico · mínimo",
                f"USD {price_series.min():.2f}","EIA/FRED desde enero de 2000")
    metric_card(slide,4.81,2.4,3.75,"Brent histórico · máximo",
                f"USD {price_series.max():.2f}","Episodios de fuerte volatilidad")
    metric_card(slide,8.88,2.4,3.75,"Último Brent observado",
                f"USD {base:.2f}",forecast["latest_observed_date"],True)
    panel(slide,.74,4.58,11.88,1.30)
    text(slide,"DESAFÍO: anticipar escenarios de coste sin ocultar incertidumbre",
         1.0,4.91,11.1,.56,20,True)

    slide=slide_new(prs,2,"Tres redes. Mismos datos. Validación temporal.",
       "RNN simple, LSTM y GRU · ventana de 60 sesiones · horizontes 5, 63 y 252.")
    entries=[
        ("01 · FUENTES","Brent/WTI EIA-FRED y señales macroeconómicas y financieras"),
        ("02 · PREPROCESADO","Huecos, transformaciones y escalado ajustados solo al pasado"),
        ("03 · REDES","RNN, LSTM y GRU comparadas con el predictor ingenuo"),
        ("04 · VALIDACIÓN","Walk-forward; entrenamiento hasta 2023, evaluación posterior")
    ]
    for i,(name,note) in enumerate(entries):
        x=.76+(i%2)*6.10
        y=2.38+(i//2)*1.84
        panel(slide,x,y,5.77,1.58)
        text(slide,name,x+.22,y+.18,5.36,.40,17,True,CYAN)
        text(slide,note,x+.22,y+.68,5.30,.69,14)

    slide=slide_new(prs,3,"Resultados fuera del periodo de entrenamiento",
       "RMSE USD/barril y variación relativa al ingenuo · modelos ajustados hasta 2023.")
    positions=[.80,2.86,6.15,9.44]
    for x,title in zip(positions,["RED","5 SESIONES","63 SESIONES","252 SESIONES"]):
        text(slide,title,x,2.23,3.1,.38,13,True,CYAN)
    for i,arch in enumerate(("rnn","lstm","gru")):
        y=2.73+i*1.10
        panel(slide,.75,y,11.84,.98)
        text(slide,arch.upper()+(" *" if arch==selected else ""),
             .90,y+.25,1.88,.40,18,True,CYAN if arch==selected else WHITE)
        for j,h in enumerate(horizons,1):
            stats=comparison[arch]["metrics"][str(h)]
            text(slide,f"{stats['rmse']:.2f} USD",
                 positions[j],y+.15,3,.34,18,True)
            diff=stats["improvement_pct"]
            text(slide,f"{diff:+.2f}% vs ingenuo",
                 positions[j],y+.56,3,.31,12,False,CYAN if diff>0 else AMBER)
    text(slide,"* Arquitectura elegida por validación pre-2024: "
         +selected.upper()+". 2024+ ya fue examinado anteriormente.",
         .83,6.34,11.75,.41,12,False,MUTED)

    slide=slide_new(prs,4,"Demostración: previsiones en tres horizontes",
       "Modelo candidato final reajustado. No confundir su pronóstico con el test histórico.")
    for i,h in enumerate(horizons):
        y=2.24+1.28*i
        panel(slide,.75,y,11.83,1.09)
        p=float(forecast["predictions"][str(h)])
        text(slide,f"{h} sesiones",1.02,y+.24,2.25,.46,19,True,CYAN)
        text(slide,f"USD {p:.2f}",3.15,y+.20,3.04,.51,24,True)
        text(slide,f"{100*(p/base-1):+.2f}% vs hoy",
             7.03,y+.15,4.50,.35,16)
        if uncertainty and h in uncertainty:
            lo,hi=uncertainty[h]
            text(slide,f"Banda 80% orientativa: {lo:.2f}–{hi:.2f} USD",
                 7.03,y+.59,5.0,.35,12,False,MUTED)
        else:
            text(slide,"Sin banda empírica documentada",
                 7.03,y+.59,5.0,.35,12,False,MUTED)
    text(slide,"La app muestra bandas empíricas orientativas cuando existen residuos; "
         "no existe cobertura futura del 80% demostrada.",
         .82,6.22,11.76,.72,13,False,MUTED)

    slide=slide_new(prs,5,"Valor para compras de energía — con límites",
       "Una herramienta auditable, útil para escenarios, no un sistema de trading.")
    panel(slide,.77,2.27,5.71,3.74)
    panel(slide,6.80,2.27,5.72,3.74)
    text(slide,"OPORTUNIDADES",1.08,2.66,5.1,.5,18,True,CYAN)
    text(slide,"• Seguir precios actualizados\n"
         "• Comparar tres horizontes\n"
         "• Evaluar riesgo con bandas orientativas\n"
         "• Informar compras y presupuestos",
         1.08,3.29,5.07,2.18,16)
    text(slide,"LÍMITES REALES",7.12,2.66,5.1,.5,18,True,AMBER)
    text(slide,"• A 5 sesiones no supera al ingenuo\n"
         "• A 63 sesiones la mejora es pequeña\n"
         "• A 252 sesiones depende del periodo\n"
         "• Exógenas sin vintages certificados",
         7.12,3.29,5.07,2.18,16)
    text(slide,"Recomendación: usarlo como apoyo y medir sistemáticamente "
         "sus errores frente a los precios futuros observados.",
         .93,6.35,11.50,.48,14,False,MUTED)

    if len(prs.slides)!=5:
        raise AssertionError("La entrega no puede exceder cinco diapositivas")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    prs.save(output)
    return output


def retrospective_bands(csv_path,architecture,forecasts):
    """Display-only historical percentiles. NOT independently validated coverage."""
    path=Path(csv_path)
    if not path.is_file():
        return {}
    data=pd.read_csv(path)
    needed={"architecture","horizon","actual","predicted"}
    if not needed.issubset(data.columns):
        return {}
    result={}
    for h in (5,63,252):
        subset=data.loc[
            (data["architecture"].astype(str).str.lower()==architecture.lower()) &
            (pd.to_numeric(data["horizon"],errors="coerce")==h)
        ]
        a=pd.to_numeric(subset["actual"],errors="coerce").to_numpy(dtype=float)
        p=pd.to_numeric(subset["predicted"],errors="coerce").to_numpy(dtype=float)
        ok=np.isfinite(a)&np.isfinite(p)&(a>0)&(p>0)
        if np.count_nonzero(ok)<30:
            continue
        q=float(np.quantile(np.abs(np.log(a[ok]/p[ok])),.80))
        point=float(forecasts["predictions"][str(h)])
        result[h]=(float(point*np.exp(-q)),float(point*np.exp(q)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--data",type=Path,default=DATA_PATH)
    parser.add_argument("--weights-dir",type=Path,default=None)
    parser.add_argument("--output",type=Path,default=
                        REPORTS_DIR/"Brent_Intelligence_V4_5_slides.pptx")
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text(encoding="utf-8"))
    forecast=forecast_candidate(
        summary_path=args.summary,data_path=args.data,
        weights_dir=args.weights_dir or ARTIFACTS_DIR/"delivery_candidates"/args.summary.parent.name
    )
    raw=load_prices(args.data)
    bands=retrospective_bands(
        args.summary.parent/"holdout_predictions.csv",
        summary["selected_architecture"],forecast)
    output=build_delivery_deck(summary,forecast,raw["brent"],args.output,
                               uncertainty=bands)
    print(f"Presentación académica: {output} · 5 diapositivas")
    print("Nota: las métricas históricas no pertenecen al reajuste final")


if __name__=="__main__":
    main()

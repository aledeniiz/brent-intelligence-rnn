"""Genera exactamente 5 diapositivas comerciales desde resultados reales versionados."""
from __future__ import annotations
import json
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.util import Inches,Pt
from brent_intelligence.config import DATA_PATH,ARTIFACTS_DIR,REPORTS_DIR
from brent_intelligence.experiment import predict_latest
from brent_intelligence.features import load_prices

NAVY=RGBColor(11,20,36)
PANEL=RGBColor(22,36,60)
TEAL=RGBColor(59,224,199)
WHITE=RGBColor(246,249,253)
MUTED=RGBColor(158,176,197)
ORANGE=RGBColor(255,181,103)

def rect(slide,x,y,w,h,color=PANEL):
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb=color
    shape.line.fill.background()
    return shape

def text(slide,value,x,y,w,h,size=16,bold=False,color=WHITE):
    tb=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=tb.text_frame
    tf.word_wrap=True
    tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    p=tf.paragraphs[0]
    p.text=str(value)
    p.alignment=PP_ALIGN.LEFT
    p.font.name="Aptos"
    p.font.size=Pt(size)
    p.font.bold=bold
    p.font.color.rgb=color
    return tb

def new_slide(prs,n,title,subtitle):
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb=NAVY
    rect(slide,0,0,0.12,7.5,TEAL)
    text(slide,"BRENT INTELLIGENCE",0.65,0.33,4.5,0.37,13,True,TEAL)
    text(slide,title,0.65,0.83,12.0,0.65,30,True)
    text(slide,subtitle,0.68,1.49,11.95,0.55,13,False,MUTED)
    text(slide,f"CASO 2 · UAX     |     {n} / 5",0.65,7.07,12,0.23,10,False,MUTED)
    return slide

def card(slide,x,y,w,h,label,value,note,highlight=False):
    rect(slide,x,y,w,h)
    text(slide,label,x+0.25,y+0.18,w-0.5,0.37,13,False,MUTED)
    text(slide,value,x+0.25,y+0.63,w-0.5,0.62,26,True,TEAL if highlight else WHITE)
    text(slide,note,x+0.25,y+h-0.60,w-0.5,0.43,12,False,MUTED)

def build_deck(summary:dict,latest:dict,data,output:Path):
    horizons=[int(h) for h in summary["config"]["horizons"]]
    if horizons!=[5,63,252]:
        raise ValueError("Presentacion prevista para 5/63/252 sesiones")
    prs=Presentation()
    prs.slide_width=Inches(13.333)
    prs.slide_height=Inches(7.5)
    winner=summary["selection"]["architecture"]
    last=float(latest["last_price"])

    s=new_slide(prs,1,"Petróleo: volatilidad y decisiones",
                "Anticipar escenarios de coste ayuda a planificar; no elimina el riesgo de mercado.")
    card(s,0.7,2.25,3.75,2.12,"Mínimo histórico del dataset",
         f"USD {data['brent'].min():,.2f}","Barril · desde enero de 2000")
    card(s,4.78,2.25,3.75,2.12,"Máximo histórico del dataset",
         f"USD {data['brent'].max():,.2f}","Barril · desde enero de 2000")
    card(s,8.86,2.25,3.75,2.12,"Último Brent observado",
         f"USD {last:,.2f}",latest["last_date"],True)
    rect(s,0.7,4.78,11.9,1.38)
    text(s,"DESAFÍO",0.98,5.03,1.5,0.35,12,True,TEAL)
    text(s,"Convertir la incertidumbre energética en escenarios con evidencia cuantitativa.",
         2.36,4.96,9.85,0.66,18)

    s=new_slide(prs,2,"Una solución: tres redes, tres horizontes",
                "Datos diarios EIA/FRED, indicadores globales y validación temporal.")
    parts=[("01 · DATOS","Brent spot, WTI y señales de mercados internacionales"),
           ("02 · PREPARACIÓN","Ventana de 60 sesiones y escalado sin mirar al futuro"),
           ("03 · APRENDIZAJE","RNN simple / LSTM / GRU: 5, 63 y 252 sesiones"),
           ("04 · EVALUACIÓN","Cinco folds walk-forward, ingenuo y test reservado")]
    for i,(name,note) in enumerate(parts):
        x=0.72+(i%2)*6.12
        y=2.23+(i//2)*2.03
        rect(s,x,y,5.82,1.77)
        text(s,name,x+0.23,y+0.20,5.35,0.37,17,True,TEAL)
        text(s,note,x+0.23,y+0.68,5.38,0.85,15)

    s=new_slide(prs,3,"Evidencia fuera de muestra",
                "RMSE (USD/barril) y mejora respecto a pronosticar el último precio.")
    xs=[0.75,2.90,6.16,9.42]
    ws=[1.95,3.10,3.10,3.10]
    for j,label in enumerate(["MODELO","5 SESIONES","63 SESIONES","252 SESIONES"]):
        text(s,label,xs[j]+0.08,2.07,ws[j]-0.1,0.48,13,True,TEAL)
    for i,kind in enumerate(["rnn","lstm","gru"]):
        y=2.66+1.07*i
        rect(s,0.74,y,11.88,0.96)
        text(s,kind.upper()+(" *" if kind==winner else ""),0.9,y+0.20,1.83,0.44,19,True,
             TEAL if kind==winner else WHITE)
        for j,h in enumerate(horizons,1):
            m=summary["models"][kind]["test"][str(h)]
            ratio=m["relative_rmse"]
            relative="n/d" if ratio is None else f"{100*(1-ratio):+.1f}% vs ingenuo"
            text(s,f"RMSE USD {m['rmse']:.2f}",xs[j]+0.06,y+0.12,ws[j]-0.15,0.32,16,True)
            text(s,relative,xs[j]+0.06,y+0.53,ws[j]-0.15,0.25,12,False,
                 TEAL if ratio is not None and ratio<1 else ORANGE)
    text(s,f"* Modelo elegido por validación: {winner.upper()}. No se seleccionó con el test.",
         0.8,6.17,11.4,0.4,13,False,MUTED)

    s=new_slide(prs,4,"Pronósticos e incertidumbre",
                "Modelo de producción actualizado; no es una trayectoria diaria futura.")
    for i,h in enumerate(horizons):
        y=2.13+1.42*i
        rect(s,0.7,y,11.9,1.22)
        point=float(latest["prices"][i])
        low=float(latest["lower"][i])
        high=float(latest["upper"][i])
        text(s,f"{h} sesiones",0.98,y+0.26,2.14,0.50,19,True,TEAL)
        text(s,f"USD {point:,.2f}",3.20,y+0.20,2.80,0.55,23,True)
        text(s,f"{100*(point/last-1):+.1f}%",6.12,y+0.30,1.55,0.36,16,True)
        text(s,f"80% nominal: {low:,.2f} – {high:,.2f}",
             7.73,y+0.25,4.53,0.57,14)
    text(s,"Intervalos empíricos calibrados. Su cobertura futura no está garantizada.",
         0.78,6.57,11.7,0.33,12,False,MUTED)

    s=new_slide(prs,5,"Propuesta de valor y límites",
                "Apoyo a decisiones de compra, planificación y seguimiento; no es un sistema de trading.")
    rect(s,0.72,2.15,5.70,3.92)
    rect(s,6.73,2.15,5.85,3.92)
    text(s,"POTENCIAL DE USO",1.02,2.48,5.1,0.48,18,True,TEAL)
    text(s,"• Presupuestar compras energéticas\n• Comparar escenarios de coste\n• Medir incertidumbre por horizonte\n• Seguimiento reproducible",
         1.02,3.05,5.02,2.36,16)
    text(s,"LÍMITES IMPORTANTES",7.03,2.48,5.1,0.48,18,True,ORANGE)
    text(s,"• No promete beneficios económicos\n• No anticipa shocks geopolíticos\n• Exógenas sin vintage point-in-time\n• Cobertura futura no garantizada",
         7.03,3.05,5.0,2.36,16)
    text(s,"Próxima fase: validar la estabilidad por periodos y contrastar variables adicionales.",
         0.85,6.38,11.65,0.46,14,False,MUTED)
    if len(prs.slides)!=5:
        raise AssertionError("La entrega tiene un maximo de cinco diapositivas")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    prs.save(output)
    return output

def main():
    pointer=ARTIFACTS_DIR/"active.json"
    if not pointer.exists():
        raise FileNotFoundError("Entrena el modelo antes de generar las diapositivas")
    relative=json.loads(pointer.read_text(encoding="utf-8"))["run"]
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Ruta de modelo invalida")
    summary=json.loads((ARTIFACTS_DIR/relative/"summary.json").read_text(encoding="utf-8"))
    latest=predict_latest()
    data=load_prices(DATA_PATH)
    out=build_deck(summary,latest,data,REPORTS_DIR/"Brent_Intelligence_5_slides.pptx")
    print(f"Generadas exactamente cinco diapositivas: {out}")

if __name__=="__main__":
    main()

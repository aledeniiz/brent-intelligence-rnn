"""Exportar tablas verificables desde un entrenamiento completo, nunca de humo."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from brent_intelligence.config import ARTIFACTS_DIR,REPORTS_DIR

def export(artifacts:Path=ARTIFACTS_DIR,reports:Path=REPORTS_DIR)->dict:
    pointer=Path(artifacts)/"active.json"
    if not pointer.is_file():
        raise FileNotFoundError("Falta un entrenamiento completo (artifacts/active.json)")
    relative=json.loads(pointer.read_text(encoding="utf-8"))["run"]
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Ruta de ejecucion invalida")
    summary=json.loads((Path(artifacts)/relative/"summary.json").read_text(encoding="utf-8"))
    if summary.get("mode")!="historical-evaluation":
        raise ValueError("No se pueden exportar resultados de smoke")
    if len(summary["folds"])!=5:
        raise ValueError("Faltan folds de validacion")
    winner=summary["selection"]["architecture"]
    rows=[]
    for arch,detail in summary["models"].items():
        for h in summary["config"]["horizons"]:
            metric=detail["test"][str(h)]
            ratio=metric["relative_rmse"]
            rows.append({
                "modelo":arch.upper(),"horizonte_sesiones":h,
                "mae_usd":round(metric["mae"],5),
                "mse_usd2":round(metric["mse"],5),
                "rmse_usd":round(metric["rmse"],5),
                "rmse_ingenuo_usd":round(metric["naive_rmse"],5),
                "mejora_rmse_pct":round(100*(1-ratio),4) if ratio is not None else "",
                "parametros":detail["parameter_count"],
                "score_validacion":round(detail["validation_score"],6),
                "seleccionado":arch==winner
            })
    reports=Path(reports)
    reports.mkdir(parents=True,exist_ok=True)
    csv_file=reports/"comparison_table.csv"
    with csv_file.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    coverage=summary["uncertainty"]["test_coverage"]
    facts=[
        "# Datos verificables para la presentacion (NO es una presentacion)",
        "",
        f"Generado desde los artefactos: {relative}",
        f"Huella SHA256 dataset: {summary['data_sha256']}",
        f"Ultima fecha Brent: {summary['last_data_date']}",
        f"Red elegida solo por validacion: {winner.upper()}",
        f"Folds: {len(summary['folds'])}, test: {summary['test_period']['n_anchors']} anclajes.",
        "",
        "## Tabla comparativa",
        "Consulta reports/comparison_table.csv para las 9 filas (3 redes por 3 horizontes).",
        "",
        "## Bandas empiricas nominales del 80%"
    ]
    for h,rate in coverage.items():
        facts.append(f"- {h} sesiones: cobertura observada en test {100*rate:.2f}%.")
    facts.extend(["","## Notas metodologicas obligatorias",
                  "- El modelo de produccion actualizado no tiene el mismo test independiente.",
                  "- Los datos exogenos no estan certificados point-in-time.",
                  "- El precio futuro ingenuo se fija igual al ultimo precio observado.",
                  "- No atribuir ganancias economicas ni robustez ante shocks sin validacion."])
    (reports/"presentation_facts.md").write_text("\n".join(facts)+"\n",encoding="utf-8")
    result={"table":str(csv_file),"facts":str(reports/"presentation_facts.md"),
            "rows":len(rows),"winner":winner}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result

if __name__=="__main__":
    export()

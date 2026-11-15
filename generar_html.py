"""Genera index.html: la versión simple del dashboard.

Toma los resultados de data/ (creados por pipeline.py) y los incrusta en plantilla.html,
así la página funciona sola, sin servidor ni librerías.
"""
import json
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).parent
DATA = AQUI / "data"


def main() -> None:
    real = pd.read_csv(DATA / "semilla_tiendas_juan.csv")
    hist = pd.read_csv(DATA / "historial_sintetico_5_anios.csv")
    esc = pd.read_csv(DATA / "escenarios_2000_anios.csv")
    pron = pd.read_csv(DATA / "pronostico_2026_2028.csv")
    pred = pd.read_csv(DATA / "predicciones_2025.csv")
    val = pd.read_csv(DATA / "validacion.csv")
    resumen = json.loads((DATA / "resumen.json").read_text(encoding="utf-8"))

    q = esc.groupby("mes").ventas_usd.quantile([0.1, 0.5, 0.9]).unstack()
    anual = esc.groupby("anio_sintetico").agg(v=("ventas_usd", "sum"), u=("utilidad_usd", "sum"))
    cols = {c: c for c in pred.columns}
    sint_col = next(c for c in cols if "solo con datos sintéticos" in c)
    real_col = next(c for c in cols if "solo con datos reales" in c)
    naive_col = next(c for c in cols if "Ingenuo" in c)

    datos = {
        "real": [{"f": r.fecha[:7], "v": round(r.ventas_usd), "u": round(r.utilidad_usd)} for r in real.itertuples()],
        "generadoMes": [{"m": int(m), "lo": round(q.loc[m, 0.1]), "med": round(q.loc[m, 0.5]), "hi": round(q.loc[m, 0.9])}
                        for m in q.index],
        "generado5": hist.sort_values(["anio_sintetico", "mes"]).groupby("anio_sintetico").ventas_usd
                         .apply(lambda s: [round(x) for x in s]).tolist(),
        "contraste": {
            "real": [round(x) for x in pred.ventas_usd],
            "sintetico": [round(x) for x in pred[sint_col]],
            "soloReal": [round(x) for x in pred[real_col]],
            "ingenuo": [round(x) for x in pred[naive_col]],
        },
        "pron": [{"f": r.fecha[:7], "m": int(r.mes), "v": round(r.ventas_pronostico_usd),
                  "lo": round(q.loc[r.mes, 0.1]), "hi": round(q.loc[r.mes, 0.9])} for r in pron.itertuples()],
        "anual": {"v": [round(anual.v.quantile(p)) for p in (0.1, 0.5, 0.9)],
                  "u": [round(anual.u.quantile(p)) for p in (0.1, 0.5, 0.9)]},
        "nEscenarios": int(esc.anio_sintetico.nunique()),
        "validacion": val.to_dict(orient="records"),
        "resumen": resumen,
    }
    plantilla = (AQUI / "plantilla.html").read_text(encoding="utf-8")
    html = plantilla.replace("/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
    (AQUI / "index.html").write_text(html, encoding="utf-8")
    print(f"index.html: {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()

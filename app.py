"""Dashboard Streamlit: misma historia que index.html (real vs. generado, contraste 2025 y pronóstico)."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
REAL, GEN, ESP, ALT = "#1f4e79", "#d9822b", "#2e7d4f", "#9a9890"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MESES_LARGO = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
               "noviembre", "diciembre"]
st.set_page_config(page_title="Ventas de Tiendas_Juan", page_icon="📊", layout="centered")


def usd(x: float) -> str:
    return "$" + f"{x:,.0f}".replace(",", ".")


def info(texto: str) -> None:
    """st.info sin que Streamlit lea los '$' como fórmulas LaTeX."""
    st.info(texto.replace("$", "\\$"))


@st.cache_data
def cargar():
    return (
        pd.read_csv(DATA / "semilla_tiendas_juan.csv"),
        pd.read_csv(DATA / "historial_sintetico_5_anios.csv"),
        pd.read_csv(DATA / "escenarios_2000_anios.csv"),
        pd.read_csv(DATA / "predicciones_2025.csv"),
        pd.read_csv(DATA / "pronostico_2026_2028.csv"),
        pd.read_csv(DATA / "validacion.csv"),
        json.loads((DATA / "resumen.json").read_text(encoding="utf-8")),
    )


real, hist, esc, pred, pron, val, resumen = cargar()
col_sint = next(c for c in pred.columns if "solo con datos sintéticos" in c)
col_real = next(c for c in pred.columns if "solo con datos reales" in c)
col_ing = next(c for c in pred.columns if "Ingenuo" in c)
err = lambda c: (pred[c] - pred.ventas_usd).abs().mean()  # noqa: E731
base_layout = dict(separators=",.", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12, x=0),
                   yaxis=dict(tickprefix="$", separatethousands=True), hovermode="x unified")

st.title("Tiendas_Juan: ventas reales, generadas y esperadas")
st.markdown("Una tienda de barrio de Loja con solo 24 meses de registros. Con inteligencia artificial generamos "
            "5 años más de datos parecidos a los reales para estimar cómo podrían venir las ventas de 2026 a 2028. "
            "**Caso aproximado con datos ficticios.**")

c1, c2, c3 = st.columns(3)
c1.metric("Años de historial (2 reales + 5 generados)", "7 años")
c2.metric("Parecido con los datos reales",
          f"{round(resumen['generadores'][resumen['generador_elegido']]['calidad_global'] * 100)} de 100")
c3.metric("Error al predecir 2025", f"± {err(col_sint) / pred.ventas_usd.mean() * 100:.0f}% al mes",
          help="Modelo entrenado solo con datos generados")

# ---------------------------------------------------------------- Paso 1
st.divider()
st.caption("PASO 1")
st.subheader("¿Los datos generados se parecen a los reales?")
st.markdown("La IA aprendió de los 2 años reales y generó 5 años más. Aquí se compara cuánto se vende en promedio "
            "cada mes. Si las dos barras de un mes miden parecido, la IA aprendió bien ese mes.")
pr = real.groupby("mes").ventas_usd.mean()
pg = hist.groupby("mes").ventas_usd.mean()
fig = go.Figure()
fig.add_bar(y=MESES, x=pr.values, orientation="h", name="Real (promedio de 2024 y 2025)", marker_color=REAL,
            text=[usd(v) for v in pr.values], textposition="outside")
fig.add_bar(y=MESES, x=pg.values, orientation="h", name="Generado por IA (promedio de 5 años)", marker_color=GEN,
            text=[usd(v) for v in pg.values], textposition="outside")
fig.update_layout(**{**base_layout, "hovermode": "y unified", "yaxis": dict(autorange="reversed"),
                     "xaxis": dict(tickprefix="$", range=[0, max(pr.max(), pg.max()) * 1.18])},
                  barmode="group", height=620)
st.plotly_chart(fig, use_container_width=True)
difs = ((pg - pr).abs() / pr * 100)
peor = int(difs.idxmax())
info(f"En lo real, el mejor mes es **{MESES_LARGO[int(pr.idxmax()) - 1]}** y el más flojo "
        f"**{MESES_LARGO[int(pr.idxmin()) - 1]}**; en lo generado, **{MESES_LARGO[int(pg.idxmax()) - 1]}** y "
        f"**{MESES_LARGO[int(pg.idxmin()) - 1]}**. En promedio, cada mes generado se aleja **{difs.mean():.0f}%** del real. "
        f"La mayor diferencia está en **{MESES_LARGO[peor - 1]}** ({usd(pr[peor])} real contra {usd(pg[peor])} generado).")

# ---------------------------------------------------------------- Paso 2
st.divider()
st.caption("PASO 2")
st.subheader("¿Qué tan bien predice? Contraste con lo que pasó en 2025")
st.markdown("Le escondimos al modelo los 12 meses de 2025 y le pedimos predecirlos aprendiendo **solo de los datos "
            "generados**. Luego comparamos con lo que la tienda vendió de verdad.")
fig = go.Figure()
fig.add_scatter(x=MESES, y=pred[col_ing], name="Repetir el mismo mes del año pasado", line=dict(color=ALT, dash="dash"))
fig.add_scatter(x=MESES, y=pred[col_sint], name="Predicción con datos generados", line=dict(color=GEN, width=3),
                mode="lines+markers")
fig.add_scatter(x=MESES, y=pred.ventas_usd, name="Lo que se vendió de verdad", line=dict(color=REAL, width=3),
                mode="lines+markers")
fig.update_layout(**base_layout, height=380)
st.plotly_chart(fig, use_container_width=True)
tabla = pd.DataFrame({
    "Forma de predecir": ["Modelo que aprendió solo de los datos generados",
                          "Modelo que aprendió solo de los 11 meses reales de 2024",
                          "Repetir el mismo mes del año pasado (sin IA)"],
    "Se equivoca en promedio": [f"{usd(err(c))} al mes ({err(c) / pred.ventas_usd.mean() * 100:.1f}%)"
                                for c in (col_sint, col_real, col_ing)],
})
st.dataframe(tabla, hide_index=True, use_container_width=True)
info(f"Aprendiendo **solo** de datos generados, el modelo se equivocó en promedio **{usd(err(col_sint))}** por mes, "
        f"menos que repetir el año anterior ({usd(err(col_ing))}). El modelo con los datos reales fue algo mejor "
        f"({usd(err(col_real))}): en esta prueba, los datos generados no mejoran la predicción mes a mes; su aporte "
        "está en mostrar el rango de años posibles (paso 3).")

# ---------------------------------------------------------------- Paso 3
st.divider()
st.caption("PASO 3")
st.subheader("Ventas esperadas para 2026, 2027 y 2028")
st.markdown("El modelo entrenado con los datos reales y generados estima los próximos 36 meses. La zona sombreada "
            "muestra entre qué valores suele moverse cada mes según los datos generados.")
banda = esc.groupby("mes").ventas_usd.quantile([0.1, 0.9]).unstack()
fr, fp = pd.to_datetime(real.fecha), pd.to_datetime(pron.fecha)
fig = go.Figure()
fig.add_scatter(x=fp, y=banda.loc[pron.mes, 0.9].values, line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=fp, y=banda.loc[pron.mes, 0.1].values, line=dict(width=0), fill="tonexty",
                fillcolor="rgba(46,125,79,0.18)", name="Rango habitual (mes flojo a mes bueno)", hoverinfo="skip")
fig.add_scatter(x=fr, y=real.ventas_usd, name="Ventas reales", line=dict(color=REAL, width=2.5), mode="lines+markers")
fig.add_scatter(x=[fr.iloc[-1], *fp], y=[real.ventas_usd.iloc[-1], *pron.ventas_pronostico_usd],
                name="Ventas esperadas", line=dict(color=ESP, width=2.5, dash="dash"))
fig.update_layout(**base_layout, height=380)
st.plotly_chart(fig, use_container_width=True)

anual_esc = esc.groupby("anio_sintetico").agg(v=("ventas_usd", "sum"), u=("utilidad_usd", "sum"))
vlo, vhi = anual_esc.v.quantile(0.1), anual_esc.v.quantile(0.9)
ulo, umed, uhi = (anual_esc.u.quantile(q) for q in (0.1, 0.5, 0.9))
# se redondea cada mes antes de sumar, igual que index.html, para que ambas versiones den la misma cifra
por_anio = pd.concat([real.assign(a=real.fecha.str[:4], v=real.ventas_usd.round()).groupby("a").v.sum(),
                      pron.assign(a=pron.fecha.str[:4], v=pron.ventas_pronostico_usd.round()).groupby("a").v.sum()])
st.dataframe(pd.DataFrame({
    "Año": [f"{a} ({'esperado' if int(a) >= 2026 else 'real'})" for a in por_anio.index],
    "Ventas del año": [usd(v) for v in por_anio.values],
    "Rango habitual de un año": [f"{usd(vlo)} – {usd(vhi)}" if int(a) >= 2026 else "—" for a in por_anio.index],
}), hide_index=True, use_container_width=True)
cambio = (por_anio["2026"] - por_anio["2025"]) / por_anio["2025"] * 100
info(f"Para 2026 se esperan unos **{usd(por_anio['2026'])}** en ventas ({cambio:+.1f}% frente a 2025). Después de "
        f"pagar mercadería, arriendo y servicios, en un año típico a la tienda le quedarían unos **{usd(umed)}** "
        f"(entre {usd(ulo)} en un año flojo y {usd(uhi)} en uno bueno), es decir, unos **{usd(umed / 12)} al mes**. "
        "Ojo: el modelo no supone crecimiento ni caída, porque con solo 2 años reales no hay base para estimar una tendencia.")

# ---------------------------------------------------------------- Detalles técnicos
st.divider()
with st.expander("Detalles técnicos (para la presentación)"):
    st.markdown(f"**Generación.** Con los {resumen['filas_semilla']} meses se entrenaron dos modelos generativos de SDV: "
                "TVAE (autoencoder variacional) y CTGAN (red generativa adversarial). Se eligió el de mayor calidad "
                "según SDMetrics.")
    gen = pd.DataFrame(resumen["generadores"]).T
    gen = (gen * 100).round(1).astype(str) + "%"
    gen.columns = ["Calidad", "Forma de columnas", "Relación entre columnas", "Rechazadas por umbral"]
    st.dataframe(gen, use_container_width=True)
    st.markdown(f"**Umbrales.** Se descartó toda fila generada fuera del rango observado ± "
                f"{int(resumen['margen_umbral'] * 100)}%.")
    st.dataframe(pd.DataFrame(resumen["umbrales"]).T.rename(columns={"min": "Mínimo", "max": "Máximo"}),
                 use_container_width=True)
    st.markdown("**Validación TSTR** (Train on Synthetic, Test on Real) contra los 12 meses reales de 2025:")
    st.dataframe(val.rename(columns={"metodo": "Método", "MAE_usd": "MAE (USD)", "MAPE_pct": "MAPE (%)"}),
                 hide_index=True, use_container_width=True)
    st.markdown(f"**Pronóstico.** Gradient Boosting con mes del año y ventas del mes anterior, entrenado con reales + "
                f"sintéticos, aplicado de forma recursiva por 36 meses. Rango: percentiles 10 y 90 de "
                f"{esc.anio_sintetico.nunique()} años simulados con el modelo generativo.")

st.caption("Esta página no recomienda mantener ni cerrar el negocio: muestra lo que los datos sugieren para que el "
           "dueño decida con más información. Los datos generados solo saben lo que la tienda ya vivió; si algo "
           "cambia mucho (nueva competencia, obras en la calle, una crisis), el cálculo no lo anticipa.")

"""Dashboard Streamlit: misma historia que index.html.

Flujo: pocos datos reales -> umbrales -> datos generados por IA -> modelo -> próximos años.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
REAL, GEN, ESP, ALT, UMB = "#1f4e79", "#d9822b", "#2e7d4f", "#9a9890", "#b3261e"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MESES_LARGO = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
               "noviembre", "diciembre"]
st.set_page_config(page_title="Ventas de Tiendas_Juan", page_icon="📊", layout="centered")


def usd(x: float) -> str:
    return "$" + f"{x:,.0f}".replace(",", ".")


def info(texto: str) -> None:
    """st.info sin que Streamlit lea los '$' como fórmulas LaTeX."""
    st.info(texto.replace("$", "\\$"))


def md(texto: str) -> None:
    st.markdown(texto.replace("$", "\\$"))


@st.cache_data
def cargar():
    return (
        pd.read_csv(DATA / "semilla_tiendas_juan.csv"),
        pd.read_csv(DATA / "historial_sintetico_5_anios.csv"),
        pd.read_csv(DATA / "escenarios_2000_anios.csv"),
        pd.read_csv(DATA / "predicciones_2025.csv"),
        pd.read_csv(DATA / "pronostico_2026_2028.csv"),
        json.loads((DATA / "resumen.json").read_text(encoding="utf-8")),
    )


real, hist, esc, pred, pron, R = cargar()
U = R["umbrales"]
base_layout = dict(separators=",.", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12, x=0),
                   yaxis=dict(tickprefix="$", separatethousands=True), hovermode="x unified")
# se redondea cada mes antes de sumar, igual que index.html, para que ambas versiones den las mismas cifras
tot_real = real.assign(a=real.fecha.str[:4], v=real.ventas_usd.round()).groupby("a").v.sum()
tot_pron = pron.assign(a=pron.fecha.str[:4], v=pron.ventas_pronostico_usd.round()).groupby("a").v.sum()

st.title("Tiendas_Juan: ¿cómo podrían venir las ventas?")
md("Una tienda de barrio de Loja con solo 24 meses de registros. Con inteligencia artificial generamos más datos "
   "parecidos a los reales para estimar si los próximos años pintan buenos o malos. **Caso aproximado con datos ficticios.**")
st.caption("1\\. Pocos datos reales → umbrales → 2. Datos generados por IA → modelo → 3. Próximos años")
c1, c2, c3 = st.columns(3)
c1.metric("Años de historial (2 reales + 5 generados)", "7 años")
c2.metric("Parecido con los datos reales", f"{round(R['generadores'][R['generador_elegido']]['calidad_global'] * 100)} de 100")
c3.metric("Ventas esperadas en 2026", usd(tot_pron["2026"]))

# ---------------------------------------------------------------- Paso 1
st.divider()
st.caption("PASO 1")
st.subheader("Los datos de la tienda y sus umbrales")
md(f"La tienda solo tiene 2 años de ventas. Con ellos fijamos **umbrales**: un mínimo y un máximo para cada dato, a partir "
   f"de lo que la tienda ya vivió (con un margen de {round(R['margen_umbral'] * 100)} %). Así, la IA no puede inventar "
   "cifras absurdas, como una tienda que de pronto vende $80.000 al mes.")
fr = pd.to_datetime(real.fecha)
fig = go.Figure()
for lado, nombre in (("max", "Umbral máximo"), ("min", "Umbral mínimo")):
    fig.add_scatter(x=fr, y=[U["ventas_usd"][lado]] * len(fr), name=f"{nombre}: {usd(U['ventas_usd'][lado])}",
                    line=dict(color=UMB, dash="dash", width=1.5), hoverinfo="skip")
fig.add_scatter(x=fr, y=real.ventas_usd, name="Ventas reales de cada mes", line=dict(color=REAL, width=2.5),
                mode="lines+markers")
fig.update_layout(**base_layout, height=340)
fig.update_xaxes(tickvals=pd.to_datetime(["2024-07-01", "2025-07-01"]), ticktext=["2024", "2025"])
st.plotly_chart(fig, use_container_width=True)
u1, u2 = st.columns(2)
u1.metric("Ventas del mes", f"{usd(U['ventas_usd']['min'])} a {usd(U['ventas_usd']['max'])}")
u2.metric("Clientes del mes", f"{U['clientes']['min']:.0f} a {U['clientes']['max']:.0f}")
u3, u4 = st.columns(2)
u3.metric("Gasto promedio por cliente",
          f"${U['ticket_promedio']['min']:.2f} a ${U['ticket_promedio']['max']:.2f}".replace(".", ","))
u4.metric("Gasto en mercadería del mes", f"{usd(U['gasto_stock_usd']['min'])} a {usd(U['gasto_stock_usd']['max'])}")

# ---------------------------------------------------------------- Paso 2
st.divider()
st.caption("PASO 2")
st.subheader("Los datos generados por la IA")
st.markdown("La IA aprendió de los 2 años reales y generó más años parecidos. Aquí se compara cuánto se vende en "
            "promedio cada mes: si las dos barras de un mes miden parecido, la IA aprendió bien ese mes.")
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
difs = (pg - pr).abs() / pr * 100
peor = int(difs.idxmax())
rechazo = round(R["generadores"][R["generador_elegido"]]["tasa_rechazo_umbrales"] * 100)
info(f"En lo real, el mejor mes es **{MESES_LARGO[int(pr.idxmax()) - 1]}** y el más flojo "
     f"**{MESES_LARGO[int(pr.idxmin()) - 1]}**; en lo generado, **{MESES_LARGO[int(pg.idxmax()) - 1]}** y "
     f"**{MESES_LARGO[int(pg.idxmin()) - 1]}**. En promedio, cada mes generado se aleja **{difs.mean():.0f}%** del real; "
     f"la mayor diferencia está en **{MESES_LARGO[peor - 1]}** ({usd(pr[peor])} real contra {usd(pg[peor])} generado). "
     f"Los umbrales hicieron su trabajo: se descartó el **{rechazo}%** de lo que generó la IA por salirse de ellos.")

# ---------------------------------------------------------------- Paso 3
st.divider()
st.caption("PASO 3")
st.subheader("¿Cómo podrían venir 2026, 2027 y 2028?")
st.markdown("Un modelo entrenado con los datos reales y los generados estima los próximos 36 meses. La zona sombreada "
            "muestra entre qué valores suele moverse cada mes.")
banda = esc.groupby("mes").ventas_usd.quantile([0.1, 0.9]).unstack()
fp = pd.to_datetime(pron.fecha)
fig = go.Figure()
fig.add_scatter(x=fp, y=banda.loc[pron.mes, 0.9].values, line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=fp, y=banda.loc[pron.mes, 0.1].values, line=dict(width=0), fill="tonexty",
                fillcolor="rgba(46,125,79,0.18)", name="Rango habitual (mes flojo a mes bueno)", hoverinfo="skip")
fig.add_scatter(x=fr, y=real.ventas_usd, name="Ventas reales", line=dict(color=REAL, width=2.5), mode="lines+markers")
fig.add_scatter(x=[fr.iloc[-1], *fp], y=[real.ventas_usd.iloc[-1], *pron.ventas_pronostico_usd],
                name="Ventas esperadas", line=dict(color=ESP, width=2.5, dash="dash"))
fig.update_layout(**base_layout, height=380)
st.plotly_chart(fig, use_container_width=True)

anual = esc.groupby("anio_sintetico").agg(v=("ventas_usd", "sum"), u=("utilidad_usd", "sum"))
vlo, vhi = anual.v.quantile(0.1), anual.v.quantile(0.9)
ulo, umed, uhi = (anual.u.quantile(q) for q in (0.1, 0.5, 0.9))
t25 = tot_real["2025"]
de10 = round(float((anual.v > real[real.fecha.str.startswith("2025")].ventas_usd.sum()).mean()) * 10)
for col, a in zip(st.columns(3), ("2026", "2027", "2028")):
    cambio = (tot_pron[a] - t25) / t25 * 100
    col.metric(f"{a} (esperado)", usd(tot_pron[a]), f"{cambio:+.1f}% frente a 2025")
    col.caption(f"Rango: {usd(vlo)} – {usd(vhi)}. En **{de10} de cada 10** escenarios se vende más que en 2025."
                .replace("$", "\\$"))
info(f"2025 fue un año fuerte para la tienda ({usd(t25)}), así que superarlo no es lo más probable: ocurre en {de10} de "
     f"cada 10 escenarios. Lo más probable es un año **parecido a 2024** ({usd(tot_real['2024'])}). Después de pagar "
     f"mercadería, arriendo y servicios, en un año típico a la tienda le quedarían unos **{usd(umed)}** (entre {usd(ulo)} "
     f"en un año flojo y {usd(uhi)} en uno bueno), es decir, unos **{usd(umed / 12)} al mes**. Ojo: con solo 2 años "
     "reales no hay base para estimar si el negocio crece o decrece, por eso los tres años se ven parecidos.")

# ---------------------------------------------------------------- Detalles técnicos
st.divider()
with st.expander("Detalles técnicos (para la presentación)"):
    st.markdown(f"**Generación.** Con los {R['filas_semilla']} meses se entrenaron dos modelos generativos de SDV: "
                "TVAE (autoencoder variacional) y CTGAN (red generativa adversarial). Se eligió el de mayor calidad "
                "según SDMetrics.")
    gen = pd.DataFrame(R["generadores"]).T
    gen = (gen * 100).round(1).astype(str) + "%"
    gen.columns = ["Calidad", "Forma de columnas", "Relación entre columnas", "Rechazadas por umbral"]
    st.dataframe(gen, use_container_width=True)

    st.markdown("**Examen del método (TSTR: Train on Synthetic, Test on Real).** Para saber si el método merece "
                "confianza, se hizo como si solo existiera 2024: la IA generó datos a partir de 2024, un modelo aprendió "
                "de ellos y predijo 2025, que luego se comparó con las ventas registradas.")
    col_sint = next(c for c in pred.columns if "solo con datos sintéticos" in c)
    col_real = next(c for c in pred.columns if "solo con datos reales" in c)
    col_ing = next(c for c in pred.columns if "Ingenuo" in c)
    err = lambda c: (pred[c] - pred.ventas_usd).abs().mean()  # noqa: E731
    st.dataframe(pd.DataFrame({
        "Forma de predecir 2025": ["Modelo que aprendió solo de los 11 meses reales de 2024",
                                   "Repetir el mismo mes del año pasado (sin IA)",
                                   "Modelo que aprendió solo de datos generados a partir de 2024"],
        "Error promedio por mes": [f"{usd(err(c))} ({err(c) / pred.ventas_usd.mean() * 100:.1f}%)"
                                   for c in (col_real, col_ing, col_sint)],
    }), hide_index=True, use_container_width=True)
    st.markdown("Con solo 12 meses para aprender, los datos generados **no superaron** a repetir el año anterior ni al "
                "modelo con datos reales. Su aporte no está en acertar un número, sino en mostrar el **rango** de lo que "
                "podría pasar (paso 3), algo que repetir el año anterior no da. El pronóstico final usa los 24 meses.")
    n_esc = f"{esc.anio_sintetico.nunique():,}".replace(",", ".")
    md(f"**Pronóstico.** Gradient Boosting con el mes del año y las ventas del mes anterior, entrenado con datos reales y "
       f"generados, aplicado de forma recursiva por 36 meses. El rango y la comparación con 2025 salen de {n_esc} "
       f"años simulados con el modelo generativo. Costos fijos supuestos: {usd(R['costos_fijos_mensuales'])} al mes.")

st.caption("Esta página no recomienda mantener ni cerrar el negocio: muestra lo que los datos sugieren para que el "
           "dueño decida con más información. Los datos generados solo saben lo que la tienda ya vivió; si algo "
           "cambia mucho (nueva competencia, obras en la calle, una crisis), el cálculo no lo anticipa.")

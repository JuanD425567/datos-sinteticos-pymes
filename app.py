"""Dashboard de la prueba de concepto: datos sintéticos para decisiones de pymes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="Datos sintéticos para pymes", page_icon="📊", layout="wide")


@st.cache_data
def cargar():
    return (
        pd.read_csv(DATA / "semilla_tiendas_juan.csv"),
        pd.read_csv(DATA / "historial_sintetico_5_anios.csv"),
        pd.read_csv(DATA / "escenarios_2000_anios.csv"),
        pd.read_csv(DATA / "validacion.csv"),
        pd.read_csv(DATA / "predicciones_2025.csv"),
        pd.read_csv(DATA / "pronostico_2026_2028.csv").head(12),
        json.loads((DATA / "resumen.json").read_text(encoding="utf-8")),
    )


real, historial, escenarios, validacion, pred25, pron26, resumen = cargar()
historial["mes"] = historial["mes"].astype(int)
escenarios["mes"] = escenarios["mes"].astype(int)
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

st.title("📊 Tiendas_Juan: ¿me alcanza para invertir?")
st.caption(
    "Prueba de concepto: IA generativa (GAN / VAE con SDV) para ampliar los pocos datos de una pyme de Loja "
    "y simular escenarios de ventas. Todos los datos son ficticios."
)
st.warning(
    "Herramienta de **apoyo** a la decisión, no un oráculo. Los datos sintéticos no crean información nueva: "
    "cuantifican la incertidumbre de lo que la tienda ya registró.",
    icon="⚠️",
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["1 · Datos de la tienda", "2 · Datos sintéticos", "3 · Predicción y validación", "4 · Decisión de inversión"]
)

# ------------------------------------------------------------------ 1
with tab1:
    st.subheader("Lo que la tienda tiene: 24 meses de registros")
    c1, c2, c3 = st.columns(3)
    c1.metric("Meses registrados", len(real))
    c2.metric("Venta mensual promedio", f"${real.ventas_usd.mean():,.0f}")
    c3.metric("Utilidad mensual promedio", f"${real.utilidad_usd.mean():,.0f}")
    fig = px.line(real, x="fecha", y=["ventas_usd", "gasto_stock_usd", "utilidad_usd"], markers=True)
    fig.update_layout(legend_title_text="", yaxis_title="USD", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Con 24 puntos casi no se puede entrenar un modelo ni estimar qué tan malo puede ser un mal año.")
    st.dataframe(real, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ 2
with tab2:
    st.subheader("Ampliación con modelos generativos")
    st.markdown(
        f"Se entrenaron dos modelos generativos de la librería **SDV** con los 24 meses y se eligió el de mejor "
        f"calidad: **{resumen['generador_elegido']}**. Luego se descartó toda fila fuera de los **umbrales** "
        f"(rango observado ± {int(resumen['margen_umbral'] * 100)} %)."
    )
    gen = pd.DataFrame(resumen["generadores"]).T.reset_index().rename(columns={"index": "modelo"})
    st.dataframe(gen, use_container_width=True, hide_index=True)

    umb = pd.DataFrame(resumen["umbrales"]).T.reset_index().rename(columns={"index": "variable"})
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("**Umbrales aplicados**")
        st.dataframe(umb, use_container_width=True, hide_index=True)
    with col_b:
        var = st.selectbox("Comparar distribución", ["ventas_usd", "clientes", "ticket_promedio", "gasto_stock_usd"])
        comp = pd.concat(
            [real[[var]].assign(origen="Real (24)"), historial[[var]].assign(origen="Sintético (5 años)")]
        )
        fig = px.histogram(comp, x=var, color="origen", barmode="overlay", histnorm="probability density", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Estacionalidad: real vs. sintético** (promedio por mes)")
    est = pd.concat(
        [
            real.groupby("mes").ventas_usd.mean().rename("Real"),
            historial.groupby("mes").ventas_usd.mean().rename("Sintético"),
        ],
        axis=1,
    ).reset_index()
    est["mes"] = est["mes"].map(lambda m: MESES[m - 1])
    st.plotly_chart(px.bar(est, x="mes", y=["Real", "Sintético"], barmode="group"), use_container_width=True)
    with st.expander("Ver historial sintético de 5 años"):
        st.dataframe(historial, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ 3
with tab3:
    st.subheader("¿Sirven los datos sintéticos para predecir?")
    st.markdown(
        "Validación **TSTR** (*Train on Synthetic, Test on Real*): se entrena con datos sintéticos y se mide el "
        "error contra los 12 meses **reales** de 2025, que el modelo nunca vio. Se compara con métodos sin IA."
    )
    st.dataframe(validacion, use_container_width=True, hide_index=True)
    cols = [c for c in pred25.columns if c not in ("fecha", "mes")]
    fig = px.line(pred25, x="fecha", y=cols, markers=True)
    fig.update_layout(legend_title_text="", yaxis_title="Ventas USD", xaxis_title="", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pronóstico 2026")
    fig = go.Figure()
    fig.add_scatter(x=real.fecha, y=real.ventas_usd, name="Real 2024-2025", mode="lines+markers")
    fig.add_scatter(x=pron26.fecha, y=pron26.ventas_pronostico_usd, name="Pronóstico 2026", mode="lines+markers",
                    line=dict(dash="dash"))
    banda = escenarios.groupby("mes").ventas_usd.quantile([0.1, 0.9]).unstack()
    fig.add_scatter(x=pron26.fecha, y=banda[0.9].values, mode="lines", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=pron26.fecha, y=banda[0.1].values, mode="lines", line=dict(width=0), fill="tonexty",
                    name="Rango probable (P10–P90)")
    fig.update_layout(yaxis_title="Ventas USD")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ 4
with tab4:
    st.subheader("¿Puedo pagar una nueva inversión?")
    st.markdown("Se simulan **2000 años posibles** con el modelo generativo. Mueve los valores de la inversión.")
    c1, c2, c3 = st.columns(3)
    nombre = c1.text_input("¿Qué quieres comprar o pagar?", "Congelador nuevo")
    cuota = c2.number_input("Cuota mensual (USD)", 0.0, 2000.0, 180.0, 10.0)
    colchon = c3.number_input("Colchón mínimo que quieres conservar al mes (USD)", 0.0, 2000.0, 150.0, 10.0)

    anual = escenarios.groupby("anio_sintetico").agg(ventas=("ventas_usd", "sum"), utilidad=("utilidad_usd", "sum"))
    meses_ok = (escenarios.utilidad_usd - cuota >= colchon).groupby(escenarios.anio_sintetico).sum()
    p_anual = float((anual.utilidad - 12 * cuota >= 12 * colchon).mean())
    p_mes = float((escenarios.utilidad_usd - cuota >= colchon).mean())
    p_10 = float((meses_ok >= 10).mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ventas anuales (P50)", f"${anual.ventas.median():,.0f}")
    k2.metric("Año malo (P10) – bueno (P90)",
              f"{anual.ventas.quantile(.1) / 1000:,.1f}k – {anual.ventas.quantile(.9) / 1000:,.1f}k")
    k3.metric("Prob. de cubrirlo en el año", f"{p_anual:.0%}")
    k4.metric("Prob. de cubrirlo ≥10 de 12 meses", f"{p_10:.0%}")

    if p_10 >= 0.8:
        st.success(f"**{nombre}**: en la mayoría de escenarios la tienda puede pagar la cuota y mantener el colchón.")
    elif p_10 >= 0.5:
        st.warning(f"**{nombre}**: es viable pero con riesgo; habrá meses ajustados (enero, febrero). "
                   "Considera una cuota menor o ahorrar antes.")
    else:
        st.error(f"**{nombre}**: en la mayoría de escenarios no alcanza. Mejor esperar o buscar una cuota menor.")

    st.markdown("**Utilidad mensual que queda después de la cuota, por mes** (2000 escenarios)")
    tmp = escenarios.assign(restante=escenarios.utilidad_usd - cuota, mes_nombre=escenarios.mes.map(lambda m: MESES[m - 1]))
    fig = px.box(tmp, x="mes_nombre", y="restante", points=False, category_orders={"mes_nombre": MESES})
    fig.add_hline(y=colchon, line_dash="dash", annotation_text="Colchón mínimo")
    fig.update_layout(xaxis_title="", yaxis_title="USD que quedan")
    st.plotly_chart(fig, use_container_width=True)

    riesgo = tmp.groupby("mes_nombre").restante.apply(lambda s: (s < colchon).mean()).reindex(MESES)
    st.markdown("**Probabilidad de quedar bajo el colchón por mes**")
    st.plotly_chart(px.bar(x=riesgo.index, y=riesgo.values, labels={"x": "", "y": "probabilidad"}),
                    use_container_width=True)
    st.caption(f"Probabilidad de que un mes cualquiera cubra cuota + colchón: {p_mes:.0%}.")

"""Pasos 2 a 4 de la prueba de concepto.

2. Amplía los 24 meses semilla con modelos generativos de SDV (TVAE = VAE, CTGAN = GAN)
   y aplica umbrales (mínimo/máximo por variable) para que no aparezcan valores absurdos.
3. Entrena un modelo de predicción de ventas con los datos sintéticos y lo valida
   contra los meses reales de 2025 (TSTR: Train on Synthetic, Test on Real),
   comparándolo con métodos sin IA.
4. Genera miles de "años posibles" para el análisis de escenarios del dashboard.

Todo lo que necesita el dashboard queda en data/ (el dashboard no reentrena nada).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sdv.evaluation.evaluation import evaluate_quality
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")
np.random.seed(425)

DATA = Path(__file__).parent / "data"
ARRIENDO, SERVICIOS = 250.0, 65.0
MARGEN_UMBRAL = 0.15  # los umbrales permiten 15 % más allá de lo observado
FERIADOS = {1: 1, 2: 2, 3: 0, 4: 1, 5: 2, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1, 11: 3, 12: 1}
COLUMNAS_GEN = ["mes", "clientes", "ticket_promedio", "ventas_usd", "gasto_stock_usd"]
NUMERICAS = ["clientes", "ticket_promedio", "ventas_usd", "gasto_stock_usd"]


# ---------------------------------------------------------------- 2. datos sintéticos
def calcular_umbrales(real: pd.DataFrame) -> dict:
    umbrales = {}
    for col in NUMERICAS:
        lo, hi = real[col].min(), real[col].max()
        umbrales[col] = {"min": round(lo * (1 - MARGEN_UMBRAL), 2), "max": round(hi * (1 + MARGEN_UMBRAL), 2)}
    ratio = real["gasto_stock_usd"] / real["ventas_usd"]
    umbrales["ratio_stock_ventas"] = {"min": round(ratio.min() - 0.05, 3), "max": round(ratio.max() + 0.05, 3)}
    return umbrales


def aplicar_umbrales(df: pd.DataFrame, umbrales: dict) -> pd.DataFrame:
    ok = pd.Series(True, index=df.index)
    for col in NUMERICAS:
        ok &= df[col].between(umbrales[col]["min"], umbrales[col]["max"])
    ratio = df["gasto_stock_usd"] / df["ventas_usd"]
    ok &= ratio.between(umbrales["ratio_stock_ventas"]["min"], umbrales["ratio_stock_ventas"]["max"])
    return df[ok].copy()


def completar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mes"] = df["mes"].astype(int)
    df["clientes"] = df["clientes"].round().astype(int)
    df["feriados"] = df["mes"].map(FERIADOS)
    df["utilidad_usd"] = (df["ventas_usd"] - df["gasto_stock_usd"] - ARRIENDO - SERVICIOS).round(2)
    return df


def muestrear_valido(synth, n: int, umbrales: dict) -> tuple[pd.DataFrame, float]:
    """Muestrea hasta tener n filas que respetan los umbrales; devuelve también la tasa de rechazo."""
    validas, generadas = [], 0
    while sum(len(v) for v in validas) < n:
        lote = synth.sample(num_rows=max(n, 2000))["tienda"]
        generadas += len(lote)
        validas.append(aplicar_umbrales(lote, umbrales))
    df = pd.concat(validas, ignore_index=True)
    rechazo = 1 - len(df) / generadas
    return completar(df.iloc[:n]), rechazo


def armar_anios(pool: pd.DataFrame, n_anios: int, rng) -> pd.DataFrame:
    """Arma años sintéticos completos tomando, para cada mes, una fila generada de ese mes.

    Cada año recibe un "nivel" aleatorio (año bueno o malo) que se comparte entre sus meses,
    para que los meses no sean independientes: un año malo tiende a ser malo en varios meses.
    """
    from scipy.stats import norm

    por_mes = {m: g.sort_values("ventas_usd").reset_index(drop=True) for m, g in pool.groupby("mes")}
    filas = []
    for a in range(n_anios):
        nivel = rng.normal()
        for m in range(1, 13):
            g = por_mes[m]
            q = norm.cdf(0.6 * nivel + 0.8 * rng.normal())
            fila = g.iloc[min(int(q * len(g)), len(g) - 1)].to_dict()
            fila["mes"] = int(fila["mes"])
            fila["anio_sintetico"] = a + 1
            filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- 3. predicción
def features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    x["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    x["ventas_mes_anterior"] = df["ventas_mes_anterior"]
    return x


def con_lag(df: pd.DataFrame, grupo: str | None = None) -> pd.DataFrame:
    df = df.copy()
    if grupo:
        df["ventas_mes_anterior"] = df.groupby(grupo)["ventas_usd"].shift(1)
    else:
        df["ventas_mes_anterior"] = df["ventas_usd"].shift(1)
    return df.dropna(subset=["ventas_mes_anterior"])


def modelo() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=425)


def main() -> None:
    import torch

    torch.manual_seed(425)  # resultados reproducibles
    rng = np.random.default_rng(425)
    real =pd.read_csv(DATA / "semilla_tiendas_juan.csv")
    umbrales = calcular_umbrales(real)

    metadata = Metadata.detect_from_dataframes({"tienda": real[COLUMNAS_GEN]}, infer_keys=None)
    metadata.update_column("mes", sdtype="categorical", table_name="tienda")

    # Entrenamos dos modelos generativos y nos quedamos con el de mejor calidad.
    candidatos = {
        "TVAE (autoencoder variacional)": TVAESynthesizer(
            metadata, epochs=1500, batch_size=24, enforce_min_max_values=False
        ),
        "CTGAN (red generativa adversarial)": CTGANSynthesizer(
            metadata, epochs=1500, batch_size=20, enforce_min_max_values=False
        ),
    }
    resultados_gen, pools = {}, {}
    for nombre, synth in candidatos.items():
        print(f"Entrenando {nombre}...")
        synth.fit({"tienda": real[COLUMNAS_GEN]})
        pool, rechazo = muestrear_valido(synth, 30000, umbrales)
        rep = evaluate_quality(real[COLUMNAS_GEN], pool[COLUMNAS_GEN], metadata, verbose=False)
        props = rep.get_properties().set_index("Property")["Score"]
        resultados_gen[nombre] = {
            "calidad_global": round(rep.get_score(), 4),
            "forma_columnas": round(float(props.get("Column Shapes", np.nan)), 4),
            "relacion_entre_columnas": round(float(props.get("Column Pair Trends", np.nan)), 4),
            "tasa_rechazo_umbrales": round(rechazo, 4),
        }
        pools[nombre] = pool
        print(f"  {resultados_gen[nombre]}")

    elegido = max(resultados_gen, key=lambda k: resultados_gen[k]["calidad_global"])
    pool = pools[elegido]
    print(f"Generador elegido: {elegido}")

    # Historial sintético de 5 años (lo que se muestra como "datos ampliados")
    historial = armar_anios(pool, 5, rng)
    historial.to_csv(DATA / "historial_sintetico_5_anios.csv", index=False)

    # --------- validación TSTR: probamos contra los 12 meses reales de 2025
    real_lag = con_lag(real)
    test = real_lag[real_lag["anio"] == 2025]
    train_real = real_lag[real_lag["anio"] == 2024]
    entrenamiento_sint = con_lag(armar_anios(pool, 200, rng), grupo="anio_sintetico")

    m_real = modelo().fit(features(train_real), train_real["ventas_usd"])
    m_sint = modelo().fit(features(entrenamiento_sint), entrenamiento_sint["ventas_usd"])
    aumentado = pd.concat([train_real] * 20 + [entrenamiento_sint], ignore_index=True)
    m_aum = modelo().fit(features(aumentado), aumentado["ventas_usd"])

    y = test["ventas_usd"].values
    real_2024 = real[real["anio"] == 2024].set_index("mes")["ventas_usd"]
    predicciones = {
        "Ingenuo estacional (mismo mes del año pasado) - sin IA": test["mes"].map(real_2024).values,
        "Promedio móvil 3 meses - sin IA": real["ventas_usd"].rolling(3).mean().shift(1)[test.index].values,
        "Modelo entrenado solo con datos reales (11 meses)": m_real.predict(features(test)),
        "Modelo entrenado solo con datos sintéticos (TSTR)": m_sint.predict(features(test)),
        "Modelo reales + sintéticos": m_aum.predict(features(test)),
    }
    validacion = []
    for nombre, pred in predicciones.items():
        validacion.append(
            {
                "metodo": nombre,
                "MAE_usd": round(mean_absolute_error(y, pred), 2),
                "MAPE_pct": round(100 * mean_absolute_percentage_error(y, pred), 2),
            }
        )
    val_df = pd.DataFrame(validacion).sort_values("MAE_usd")
    val_df.to_csv(DATA / "validacion.csv", index=False)
    print(val_df.to_string(index=False))

    pred_test = test[["fecha", "mes", "ventas_usd"]].copy()
    for nombre, pred in predicciones.items():
        pred_test[nombre] = np.round(pred, 2)
    pred_test.to_csv(DATA / "predicciones_2025.csv", index=False)

    # --------- pronóstico 2026: modelo reales(24 meses)+sintéticos, de forma recursiva
    aumentado_full = pd.concat([real_lag] * 10 + [entrenamiento_sint], ignore_index=True)
    m_final = modelo().fit(features(aumentado_full), aumentado_full["ventas_usd"])
    anterior, filas = real["ventas_usd"].iloc[-1], []
    for mes in range(1, 13):
        x = features(pd.DataFrame({"mes": [mes], "ventas_mes_anterior": [anterior]}))
        p = float(m_final.predict(x)[0])
        filas.append({"fecha": f"2026-{mes:02d}-01", "mes": mes, "ventas_pronostico_usd": round(p, 2)})
        anterior = p
    pd.DataFrame(filas).to_csv(DATA / "pronostico_2026.csv", index=False)

    # --------- escenarios: 2000 años posibles para el análisis de inversión
    escenarios = armar_anios(pool, 2000, rng)
    escenarios[["anio_sintetico", "mes", "ventas_usd", "gasto_stock_usd", "utilidad_usd"]].round(2).to_csv(
        DATA / "escenarios_2000_anios.csv", index=False
    )

    resumen = {
        "generador_elegido": elegido,
        "generadores": resultados_gen,
        "umbrales": umbrales,
        "margen_umbral": MARGEN_UMBRAL,
        "costos_fijos_mensuales": ARRIENDO + SERVICIOS,
        "filas_semilla": len(real),
        "filas_sinteticas_generadas": len(pool),
    }
    (DATA / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

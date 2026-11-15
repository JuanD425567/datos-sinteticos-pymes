"""Paso 1: genera los datos semilla de "Tiendas_Juan", una tienda de barrio ficticia de Loja.

Simula lo que una pyme tendría en su cuaderno: 24 meses (ene-2024 a dic-2025) de registros mensuales.
Estos datos hacen el papel de "datos reales" en la prueba de concepto.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(425)
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

# Estacionalidad típica de una tienda de barrio en Loja (multiplicador sobre la venta base)
ESTACIONALIDAD = {
    1: 0.85,   # enero: cuesta de enero
    2: 0.88,   # febrero: carnaval, pero bolsillo flojo
    3: 0.95,
    4: 0.97,
    5: 1.00,   # día de la madre
    6: 0.96,
    7: 0.98,
    8: 1.08,   # útiles y regreso a clases (régimen Sierra)
    9: 1.18,   # Feria de Loja / Virgen del Cisne
    10: 1.00,
    11: 1.05,  # finados / independencia de Loja
    12: 1.42,  # navidad
}
FERIADOS = {1: 1, 2: 2, 3: 0, 4: 1, 5: 2, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1, 11: 3, 12: 1}

VENTA_BASE = 2300.0
CRECIMIENTO_MENSUAL = 0.004  # ~5 % anual
ARRIENDO, SERVICIOS = 250.0, 65.0


def main() -> None:
    filas = []
    for i, fecha in enumerate(pd.date_range("2024-01-01", "2025-12-01", freq="MS")):
        mes = fecha.month
        ventas = VENTA_BASE * ESTACIONALIDAD[mes] * (1 + CRECIMIENTO_MENSUAL) ** i
        ventas *= RNG.normal(1.0, 0.07)
        ticket = RNG.normal(5.3, 0.35) * (1.08 if mes == 12 else 1.0)
        clientes = round(ventas / ticket)
        ventas = round(clientes * ticket, 2)
        gasto_stock = round(ventas * RNG.normal(0.68, 0.025), 2)
        utilidad = round(ventas - gasto_stock - ARRIENDO - SERVICIOS, 2)
        filas.append(
            {
                "fecha": fecha.date().isoformat(),
                "anio": fecha.year,
                "mes": mes,
                "feriados": FERIADOS[mes],
                "clientes": clientes,
                "ticket_promedio": round(ticket, 2),
                "ventas_usd": ventas,
                "gasto_stock_usd": gasto_stock,
                "utilidad_usd": utilidad,
            }
        )
    df = pd.DataFrame(filas)
    df.to_csv(DATA / "semilla_tiendas_juan.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

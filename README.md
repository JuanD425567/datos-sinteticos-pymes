# Datos sintéticos para pymes: prueba de concepto

Tiendas_Juan, una tienda de barrio ficticia de Loja, tiene apenas 24 meses de registros.
Esta prueba de concepto amplía esos datos con modelos generativos (TVAE y CTGAN de SDV),
aplica umbrales para que los valores sean plausibles, entrena un modelo de predicción y
simula 2000 años posibles para responder: **¿puedo pagar una nueva inversión?**

## Archivos
- `generar_semilla.py`: crea los 24 meses "reales" (ficticios) → `data/semilla_tiendas_juan.csv`
- `pipeline.py`: entrena SDV, aplica umbrales, valida (TSTR), pronostica 2026 y crea escenarios → `data/`
- `app.py`: dashboard Streamlit (solo lee `data/`; no necesita SDV ni torch)

## Correr localmente
```
pip install sdv scikit-learn streamlit plotly
python generar_semilla.py
python pipeline.py
streamlit run app.py
```

## Publicar en Streamlit Community Cloud
Sube esta carpeta (sin `.venv`) a un repositorio de GitHub, entra a share.streamlit.io,
elige el repo y `app.py` como archivo principal. `requirements.txt` solo trae lo que el dashboard usa.

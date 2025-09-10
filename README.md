# Clasificador SPAM / No-SPAM con Regresión Logística (scikit-learn)

## Descripción
Implementación reproducible para clasificar correos como SPAM o No-SPAM usando un pipeline `StandardScaler → LogisticRegression (liblinear)`.
La app incluye interfaz con Streamlit para cargar el CSV, seleccionar hasta 10 features, entrenar, evaluar F1, ajustar umbral, ver matrices de confusión, curvas PR/ROC, importancias (%) y correlaciones.

## Estructura
- `app.py`: aplicación Streamlit con todo el flujo.
- `requirements.txt`: dependencias.

## Requisitos
- Python 3.9+ recomendado.
- Instalar dependencias:
```bash
pip install -r requirements.txt

# app.py
# Clasificador SPAM/No-SPAM (Streamlit)
# Flujo: carga CSV -> selección de features -> entrenamiento (Logistic Regression) ->
# métricas (F1, matriz de confusión) -> umbral óptimo por F1 -> importancias (%) ->
# correlaciones -> curvas PR/ROC
# Requisitos: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn (solo para heatmap)
# Fabian Valero - Esteban Fonseca

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc,
    classification_report
)

# ---------- Interfaz ----------
st.set_page_config(page_title="Clasificador SPAM", page_icon="📧", layout="wide")
st.markdown("""
<style>
/* Fuente y layout limpio */
html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; }
.block-container{ padding-top: 1.2rem; padding-bottom: 2rem; }
h1,h2,h3 { font-weight: 700; letter-spacing: -0.01em; }
.small-note { opacity: 0.7; font-size: 0.9rem; }
.metric-card { border: 1px solid #2223; border-radius: 14px; padding: 14px 16px; }
kbd { background: #1113; border-radius: 6px; padding: 2px 6px; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("📧 Clasificador SPAM")
st.sidebar.write("Fabian Valero - Esteban Fonseca")
uploaded = st.sidebar.file_uploader("Cargar CSV de correos", type=["csv"])

st.sidebar.markdown("---")
test_size = st.sidebar.slider("Proporción Test (%)", 10, 40, 25, step=5)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

optimize_threshold = st.sidebar.checkbox("Optimizar umbral para F1", value=True)
custom_threshold = st.sidebar.slider("Umbral de decisión (si no optimiza)", 0.0, 1.0, 0.50, 0.01)

st.sidebar.markdown("---")
st.sidebar.caption("Sugerencia: presiona **R** para recargar tras ajustes.")

# ---------- Carga de datos ----------
DEFAULT_FEATURES = [
    "BodyLength",
    "HasLink",
    "NumExclamations",
    "HasSuspiciousWord",
    "SenderDomainReputation",
    "NumUppercaseWords",
    "HasAttachment"
]
TARGET_COL = "Spam"

def load_df(file) -> pd.DataFrame:
    return pd.read_csv(file)

if uploaded is not None:
    df = load_df(uploaded)
    st.success("CSV cargado correctamente.")
else:
    st.info("Cargar un CSV con una columna objetivo `Spam` (0/1) y features numéricas/binarias.")
    st.stop()

# ---------- Selección de features ----------
all_cols = list(df.columns)
if TARGET_COL not in all_cols:
    st.error("No se encuentra la columna objetivo `Spam` en el CSV.")
    st.stop()

suggested = [c for c in DEFAULT_FEATURES if c in df.columns]
with st.expander("Seleccionar características (features)", expanded=True):
    chosen_features = st.multiselect(
        "Seleccionar hasta 10 features:",
        options=[c for c in all_cols if c != TARGET_COL],
        default=suggested[:10],
        max_selections=10
    )
    if len(chosen_features) == 0:
        st.error("Seleccionar al menos 1 feature.")
        st.stop()

# ---------- Preparación ----------
clean_df = df.dropna(subset=chosen_features + [TARGET_COL]).copy()
X = clean_df[chosen_features].copy()
y = clean_df[TARGET_COL].astype(int)

# Verificar que los features sean numéricos o binarios
non_numeric = [c for c in chosen_features if not np.issubdtype(X[c].dtype, np.number)]
if non_numeric:
    st.warning(f"Se encontraron columnas no numéricas: {non_numeric}. Considerar preprocesarlas.")
    # Intento suave de conversión booleana
    for c in non_numeric:
        X[c] = X[c].astype("category").cat.codes

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100.0, random_state=random_state, stratify=y
)

# ---------- Modelo: Regresión Logística ----------
# (Para clasificación de SPAM es el enfoque correcto; la "regresión lineal" no aplica a target binario)
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("logreg", LogisticRegression(solver="liblinear", random_state=random_state))
])

pipe.fit(X_train, y_train)
proba_test = pipe.predict_proba(X_test)[:, 1]

# ---------- Umbral ----------
if optimize_threshold:
    prec, rec, thr = precision_recall_curve(y_test, proba_test)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
    # El arreglo thr tiene len = len(prec/rec)-1
    best_idx = int(np.nanargmax(f1s))
    best_f1 = f1s[best_idx]
    # Alinear índice
    if best_idx == 0:
        best_threshold = 0.5
    else:
        best_threshold = float(thr[best_idx-1])
    threshold = best_threshold
else:
    threshold = float(custom_threshold)

y_pred_05 = (proba_test >= 0.5).astype(int)
y_pred_thr = (proba_test >= threshold).astype(int)

f1_05 = f1_score(y_test, y_pred_05)
f1_thr = f1_score(y_test, y_pred_thr)

cm_05 = confusion_matrix(y_test, y_pred_05, labels=[0, 1])
cm_thr = confusion_matrix(y_test, y_pred_thr, labels=[0, 1])

# ---------- Importancias (%): coef estandarizados ----------
logreg = pipe.named_steps["logreg"]
coefs = logreg.coef_[0]
abs_coefs = np.abs(coefs)
if abs_coefs.sum() == 0:
    importances_pct = np.zeros_like(abs_coefs)
else:
    importances_pct = 100 * abs_coefs / abs_coefs.sum()

imp_df = (pd.DataFrame({
    "Feature": chosen_features,
    "Coef": coefs,
    "Importancia_%": importances_pct
})
.sort_values("Importancia_%", ascending=False)
.reset_index(drop=True))

# ---------- Correlaciones ----------
corr_df = clean_df[chosen_features + [TARGET_COL]].corr(numeric_only=True).round(3)
corr_with_target = corr_df[TARGET_COL].drop(TARGET_COL, errors="ignore").sort_values(ascending=False)

# ---------- Curvas PR/ROC ----------
prec, rec, _ = precision_recall_curve(y_test, proba_test)
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_auc = auc(fpr, tpr)

# ---------- UI ----------
st.title("Clasificador SPAM / No-SPAM - ML - UDEC")
st.caption("Pipeline: StandardScaler → Regresión logística (liblinear)")

# Métricas principales
colA, colB, colC = st.columns(3)
with colA:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("F1 @ 0.5", f"{f1_05:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)
with colB:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Umbral usado", f"{threshold:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)
with colC:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("F1 @ umbral", f"{f1_thr:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

# Matrices de confusión
col1, col2 = st.columns(2)
with col1:
    st.subheader("Matriz de confusión @ 0.5")
    st.dataframe(pd.DataFrame(cm_05, index=["Real:0","Real:1"], columns=["Pred:0","Pred:1"]))
with col2:
    st.subheader(f"Matriz de confusión @ {threshold:.4f}")
    st.dataframe(pd.DataFrame(cm_thr, index=["Real:0","Real:1"], columns=["Pred:0","Pred:1"]))

# Importancias
st.subheader("Importancia de características (%)")
st.dataframe(imp_df.style.format({"Importancia_%": "{:.2f}"}))

# Correlación con el target
st.subheader("Correlación (Pearson) de cada feature con Spam")
st.dataframe(corr_with_target.to_frame("Correlación con Spam"))

# Heatmap de correlaciones
st.subheader("Matriz de correlación (Pearson) - Features + Spam")
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_df, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(corr_df.columns)))
ax.set_yticks(range(len(corr_df.index)))
ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
ax.set_yticklabels(corr_df.index)
ax.set_title("Matriz de correlación")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
st.pyplot(fig, use_container_width=True)

# PR Curve
st.subheader("Curva Precisión-Recall")
fig2, ax2 = plt.subplots()
ax2.plot(rec, prec)
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall")
st.pyplot(fig2, use_container_width=True)

# ROC Curve
st.subheader(f"Curva ROC (AUC = {roc_auc:.4f})")
fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr)
ax3.plot([0,1], [0,1], linestyle="--")
ax3.set_xlabel("FPR")
ax3.set_ylabel("TPR")
ax3.set_title("ROC")
st.pyplot(fig3, use_container_width=True)

# Reporte textual
with st.expander("Reporte de clasificación (sklearn)", expanded=False):
    report = classification_report(y_test, y_pred_thr, digits=4, output_dict=False)
    st.text(report)

# Justificación de features no usados (si el CSV tiene más de 10)
not_used = [c for c in all_cols if c not in chosen_features + [TARGET_COL]]
if len(not_used) > 0:
    st.subheader("Justificación de features no utilizados")
    st.markdown("""
- Se priorizó un máximo de 10 variables, tal como exige la actividad.
- Variables textuales (e.g., `Subject`, `Body`) requerirían vectorización (TF-IDF, n-grams) generando miles de columnas y dificultando la interpretación en porcentajes.
- Se preservó interpretabilidad directa en el modelo lineal con señales ya capturadas por indicadores estructurados (palabras sospechosas, exclamaciones, links, reputación de dominio).
    """)

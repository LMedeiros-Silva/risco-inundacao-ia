"""Interface web do alerta de inundação. Rode o notebook antes (gera modelo_inundacao.joblib).

    streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Alerta de Inundação", page_icon="🌊")

try:
    b = joblib.load("modelo_inundacao.joblib")
except FileNotFoundError:
    st.error("modelo_inundacao.joblib não encontrado. Execute o notebook risco_inundacao.ipynb primeiro.")
    st.stop()

baixo, alto = b["limites"]
CORES = {"BAIXO": "green", "MÉDIO": "orange", "ALTO": "red"}

st.title("🌊 Estimativa de Risco de Inundação")
st.caption(f"Modelo: {b['nome_modelo']} · MAE {b['metricas']['MAE']:.4f} · RMSE {b['metricas']['RMSE']:.4f} (teste)")

escolha = st.selectbox("Cenário", ["(manual)"] + list(b["cenarios"]))
base = b["cenarios"].get(escolha, b["medianas"])

st.subheader("Fatores (0 = baixo, valores maiores = pior condição)")
cols = st.columns(2)
entradas = {
    f: cols[i % 2].slider(f, 0, b["maximos"][f], int(base[f]), key=f"{escolha}-{f}")
    for i, f in enumerate(b["features"])
}

prob = float(b["modelo"].predict(pd.DataFrame([entradas])[b["features"]])[0])
risco = "BAIXO" if prob < baixo else "MÉDIO" if prob < alto else "ALTO"

st.divider()
c1, c2 = st.columns(2)
c1.metric("Probabilidade prevista", f"{prob:.3f}")
c2.markdown(f"### Risco: :{CORES[risco]}[{risco}]")
st.progress(min(max(prob, 0.0), 1.0))
st.write(f"Faixas: baixo < {baixo} ≤ médio < {alto} ≤ alto (tercis do conjunto de treino).")
st.warning("Dados sintéticos (Kaggle) e faixas didáticas, sem validação para uso operacional. "
           f"Erro típico do modelo ±{b['metricas']['RMSE']:.3f}: previsões perto dos limites podem cair na faixa vizinha.")

# Estimativa de Risco de Inundação com IA

Protótipo da disciplina **Inteligência Computacional e Artificial**: prevê `FloodProbability` (dataset sintético
*Regression with a Flood Prediction Dataset* – Kaggle Playground S4E5) e converte a previsão em alerta **baixo / médio / alto**.

🎥 **Vídeo de demonstração:** https://youtu.be/XnceOwrEZ3I

| Arquivo | Conteúdo |
|---|---|
| `risco_inundacao.ipynb` | Exploração, preparação, treino (Regressão Linear x Random Forest), MAE/RMSE, gráficos, alertas, 3 cenários, interface com widgets e limitações |
| `app.py` | Interface web (Streamlit) que usa o modelo salvo pelo notebook |
| `requirements.txt` | Dependências |

## Como executar

1. Baixe `train.csv` em https://www.kaggle.com/competitions/playground-series-s4e5/data (é preciso login e aceitar as regras da competição) e salve em `data/train.csv`.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Abra `risco_inundacao.ipynb` e execute todas as células (gera `modelo_inundacao.joblib`).
4. Interface web:
   ```bash
   streamlit run app.py
   ```

Reprodutibilidade: amostra de 100 mil linhas, divisão 80/20 e modelos com `random_state=42`.

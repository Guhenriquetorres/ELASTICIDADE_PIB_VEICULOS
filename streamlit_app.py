# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE INTERFACE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Elasticidades Bayes – Gamma", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"]{
    font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif
}
.section{
    border-radius:18px;
    border:1px solid rgba(0,0,0,.06);
    padding:18px;
    background:#fff;
    box-shadow:0 8px 24px rgba(0,0,0,.04)
}
.metric-card{
    border-radius:16px;
    padding:16px 18px;
    border:1px solid rgba(0,0,0,.06);
    background:linear-gradient(180deg,rgba(0,0,0,.03),rgba(0,0,0,.015));
    box-shadow:0 4px 12px rgba(0,0,0,.05)
}
.small{color:#666;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"

# -----------------------------------------------------------------------------
# CARREGAMENTO DOS ARQUIVOS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("df_2021.csv")

    # Carrega o modelo pequeno
    with open("trace_small.pkl", "rb") as f:
        small = pickle.load(f)

    # Carrega metadata
    with open("metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    # Atualiza meta com médias
    meta["beta_mean"] = small["beta_mean"]
    meta["alpha0_mean"] = small["alpha0_mean"]
    meta["veics"] = small["veics"]

    return df, meta

df, meta = load_data()

# Estes são os coeficientes finais do modelo gamma
post_beta_mean = np.array(meta["beta_mean"])  # vetor de betas
alpha0_mean = float(meta["alpha0_mean"])      # intercepto
veics = meta["veics"]

mu_beta_prior = meta["mu_beta_prior"]
sigma_beta_prior = meta["sigma_beta_prior"]

# -----------------------------------------------------------------------------
# TÍTULO
# -----------------------------------------------------------------------------
st.title("Elasticidades Bayes — Modelo Gamma")
st.markdown("""
Painel completo com análise exploratória, prior × posterior reduzido, diagnósticos e interpretação.
""")

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_eda, tab_betas, tab_diag, tab_interp = st.tabs([
    "EDA – Exploração",
    "Prior × Posterior",
    "Diagnósticos",
    "Interpretação"
])

# =============================================================================
# TAB 1 — EDA
# =============================================================================
with tab_eda:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Exploração dos Dados — EDA")

    df["log_caminhao"]   = np.log1p(df["CAMINHAO"])
    df["log_ciclomotor"] = np.log1p(df["CICLOMOTOR"])
    df["log_automovel"]  = np.log1p(df["AUTOMOVEL"])
    df["log_industria"]  = np.log(df["vl_industria"])

    vals = df["vl_industria"]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="small">Observações</div><h3>{len(vals):,}</h3></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="small">Média</div><h3>{vals.mean():,.0f}</h3></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="small">Mediana</div><h3>{vals.median():,.0f}</h3></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="small">Desvio-padrão</div><h3>{vals.std():,.0f}</h3></div>', unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Histogramas
    fig_hist = make_subplots(rows=4, cols=2)
    vars_original = ["CAMINHAO", "CICLOMOTOR", "AUTOMOVEL", "vl_industria"]
    vars_log = ["log_caminhao", "log_ciclomotor", "log_automovel", "log_industria"]

    for i in range(4):
        fig_hist.add_trace(go.Histogram(x=df[vars_original[i]], nbinsx=40), row=i+1, col=1)
        fig_hist.add_trace(go.Histogram(x=df[vars_log[i]], nbinsx=40), row=i+1, col=2)

    fig_hist.update_layout(height=1200, template=PLOTLY_TEMPLATE, title="Distribuição Original e Log")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 2 — PRIOR × POSTERIOR (versão reduzida)
# =============================================================================
with tab_betas:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Comparação Prior × Posterior (reduzida)")

    fig = make_subplots(rows=1, cols=len(veics), subplot_titles=veics)

    for i, v in enumerate(veics):
        prior_vals = np.random.normal(mu_beta_prior[i], sigma_beta_prior[i], 5000)
        kde_prior = gaussian_kde(prior_vals)
        x_grid = np.linspace(prior_vals.min(), prior_vals.max(), 300)
        prior_density = kde_prior(x_grid)

        fig.add_trace(go.Scatter(x=x_grid, y=prior_density, mode="lines", line=dict(color="gray")), row=1, col=i+1)
        fig.add_trace(go.Scatter(x=[post_beta_mean[i]], y=[0], mode="markers", marker=dict(color="red", size=12)), row=1, col=i+1)

    fig.update_layout(height=450, template=PLOTLY_TEMPLATE, title="Prior × Posterior (médias)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 3 — DIAGNÓSTICOS
# =============================================================================
with tab_diag:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Diagnósticos do Modelo")

    df_final = df.copy()

    X = df_final[["log_caminhao", "log_ciclomotor", "log_automovel"]].values
    log_mu = alpha0_mean + X @ post_beta_mean
    df_final["y_hat"] = np.exp(log_mu)
    df_final["resid"] = df_final["vl_industria"] - df_final["y_hat"]
    df_final["erro_abs"] = df_final["resid"].abs()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_final["vl_industria"],
        y=df_final["y_hat"],
        mode="markers",
        marker=dict(color=df_final["erro_abs"], colorscale="RdBu_r"),
    ))
    max_val = df_final[["vl_industria", "y_hat"]].max().max()
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(dash="dash")))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Observado × Previsto")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 4 — INTERPRETAÇÃO
# =============================================================================
with tab_interp:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Interpretação Acadêmica")

    b1, b2, b3 = post_beta_mean

    st.markdown(f"""
### Elasticidades

• **Caminhão:** β ≈ {b1:.2f}  
Representa logística regional e capacidade de escoamento da produção.

• **Ciclomotor:** β ≈ {b2:.2f}  
Atividade econômica urbana de baixa renda.

• **Automóvel:** β ≈ {b3:.2f}  
Maior elasticidade — proxy de renda, urbanização e complexidade econômica.

### Por que Gamma?

O PIB industrial é:
- assimétrico à direita  
- estritamente positivo  
- heterocedástico  

→ A verossimilhança Gamma modela exatamente esse comportamento.
""")

    st.markdown("</div>", unsafe_allow_html=True)

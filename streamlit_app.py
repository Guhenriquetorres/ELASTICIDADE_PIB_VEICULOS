# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go   
import math
import plotly.express as px
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


def _download(df, label, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


PLOTLY_TEMPLATE = "plotly_white"   
# =============================================================================
# CONFIGURAÇÃO INICIAL
# =============================================================================
st.set_page_config(page_title="Bayes – Elasticidades (Gamma)", layout="wide")

# =====================================================
# CSS
# =====================================================
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
h1,h2,h3{letter-spacing:.2px}
hr{margin:.6rem 0 1rem 0;border-color:rgba(0,0,0,.08)}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CARREGAMENTO DOS ARQUIVOS (SEM RODAR O MODELO)
# =============================================================================
@st.cache_data
def load_data():
    # Carregar base
    df = pd.read_csv("df_2021.csv")

    # Carregar trace compactado
    import bz2, pickle
    with bz2.BZ2File("trace_multi.pbz2", "rb") as f:
        trace_multi = pickle.load(f)

    # Carregar metadata
    with open("metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    return df, trace_multi, meta

# --- CHAMAR A FUNÇÃO PARA CARREGAR OS ARQUIVOS ---
df, trace_multi, meta = load_data()

# --- AGORA SIM: USAR meta E trace_multi ---
veics = meta["veics"]
mu_beta_prior = meta["mu_beta_prior"]
sigma_beta_prior = meta["sigma_beta_prior"]

post_beta = trace_multi.posterior["beta"].stack(draws=("chain","draw")).values



# =============================================================================
# TÍTULO
# =============================================================================
st.title("Elasticidades Bayes – Modelo Gamma")
st.markdown("""
Este painel apresenta a análise exploratória e os resultados do modelo Bayesiano com verossimilhança Gamma,
estimado sobre o PIB industrial municipal de 2021.
""")


# =============================================================================
# TABS PRINCIPAIS
# =============================================================================
tab_eda, tab_betas, tab_diag, tab_interp = st.tabs([
    "EDA – Análise exploratória",
    "Prior × Posterior (Betas)",
    "Diagnósticos do modelo",
    "Interpretação"
])


# =============================================================================
# TAB 1 – EDA COMPLETA (APENAS PLOTLY)
# =============================================================================
with tab_eda:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Exploração dos Dados — EDA")

    # --------------------------------------------------------------
    # Transformações log
    # --------------------------------------------------------------
    df["log_caminhao"]   = np.log1p(df["CAMINHAO"])
    df["log_ciclomotor"] = np.log1p(df["CICLOMOTOR"])
    df["log_automovel"]  = np.log1p(df["AUTOMOVEL"])
    df["log_industria"]  = np.log(df["vl_industria"])

    vars_original = ["CAMINHAO", "CICLOMOTOR", "AUTOMOVEL", "vl_industria"]
    vars_log      = ["log_caminhao", "log_ciclomotor", "log_automovel", "log_industria"]

    # --------------------------------------------------------------
    # MÉTRICAS (baseadas apenas no PIB Industrial)
    # --------------------------------------------------------------
    vals = df["vl_industria"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><div class="small">Observações</div>'
        f'<h3>{len(vals):,}</h3></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><div class="small">Média</div>'
        f'<h3>{vals.mean():,.0f}</h3></div>'.replace(",", "."),
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card"><div class="small">Mediana</div>'
        f'<h3>{vals.median():,.0f}</h3></div>'.replace(",", "."),
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="metric-card"><div class="small">Desvio-padrão</div>'
        f'<h3>{vals.std(ddof=0):,.0f}</h3></div>'.replace(",", "."),
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # =============================================================================
    # HISTOGRAMAS (Original × Log)
    # =============================================================================
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig_hist = make_subplots(
        rows=4, cols=2,
        subplot_titles=[f"Original — {v}" for v in vars_original] +
                       [f"Log-transformado — {v}" for v in vars_log]
    )

    for i in range(4):
        # original
        fig_hist.add_trace(
            go.Histogram(
                x=df[vars_original[i]],
                nbinsx=40,
                marker=dict(color="#4C72B0"),
                opacity=0.75,
                name=f"orig_{vars_original[i]}"
            ),
            row=i+1, col=1
        )
        # log
        fig_hist.add_trace(
            go.Histogram(
                x=df[vars_log[i]],
                nbinsx=40,
                marker=dict(color="#55A868"),
                opacity=0.75,
                name=f"log_{vars_log[i]}"
            ),
            row=i+1, col=2
        )

    fig_hist.update_layout(
        height=1200,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        title="Distribuição Original vs Log-transformada"
    )

    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # =============================================================================
    # CORRELAÇÃO (apenas com Plotly)
    # =============================================================================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Correlação entre Veículos e PIB Industrial")

    cols_corr = ["CAMINHAO", "CICLOMOTOR", "AUTOMOVEL", "vl_industria"]
    corr = df[cols_corr].corr()

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            reversescale=True,
            zmid=0,
            colorbar=dict(title="corr")
        )
    )

    fig_corr.update_layout(
        height=500,
        template=PLOTLY_TEMPLATE,
        title="Correlação — Frotas × PIB Industrial (2021)"
    )

    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # =============================================================================
    # REGRESSÕES (log X log) — Scatter + linha de regressão
    # =============================================================================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Relação log(veículos) × log(PIB Industrial)")

    relacoes = [
        ("log_caminhao",   "log(CAMINHAO) × log(PIB Industrial)"),
        ("log_ciclomotor", "log(CICLOMOTOR) × log(PIB Industrial)"),
        ("log_automovel",  "log(AUTOMOVEL) × log(PIB Industrial)")
    ]

    fig_reg = make_subplots(rows=1, cols=3,
                            subplot_titles=[t for _, t in relacoes])

    for i, (col, _) in enumerate(relacoes, start=1):

        x = df[col]
        y = df["log_industria"]

        fig_reg.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=4, opacity=0.55, color="#4C72B0"),
                name=col
            ),
            row=1, col=i
        )

        # Regressão
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = coef[0] * x_line + coef[1]

        fig_reg.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="red", width=2),
                showlegend=False
            ),
            row=1, col=i
        )

    fig_reg.update_layout(
        height=450,
        template=PLOTLY_TEMPLATE,
        title="Regressões log(veículos) × log(PIB Industrial)"
    )

    st.plotly_chart(fig_reg, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)






# =============================================================================
# TAB 2 – PRIOR × POSTERIOR (versão Plotly White)
# =============================================================================
with tab_betas:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Comparação Prior × Posterior dos Betas")

    import scipy.stats as st_kde
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=len(veics),
        subplot_titles=[f"{v}" for v in veics]
    )

    for i, v in enumerate(veics):

        # -----------------------------
        # PRIOR & POSTERIOR VALUES
        # -----------------------------
        prior_vals = np.random.normal(mu_beta_prior[i], sigma_beta_prior[i], 5000)
        post_vals  = post_beta[i]

        ci_low, ci_high = np.percentile(post_vals, [2.5, 97.5])
        post_mean = np.mean(post_vals)

        # -----------------------------
        # KDE ESTIMATION (PRIOR)
        # -----------------------------
        kde_prior = st_kde.gaussian_kde(prior_vals)
        x_grid = np.linspace(
            min(prior_vals.min(), post_vals.min()),
            max(prior_vals.max(), post_vals.max()),
            300
        )
        prior_density = kde_prior(x_grid)

        # -----------------------------
        # KDE ESTIMATION (POSTERIOR)
        # -----------------------------
        kde_post = st_kde.gaussian_kde(post_vals)
        post_density = kde_post(x_grid)

        # -----------------------------
        # PLOT PRIOR
        # -----------------------------
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=prior_density,
                mode="lines",
                line=dict(color="gray", width=2),
                fill="tozeroy",
                fillcolor="rgba(150,150,150,0.25)",
                name="Prior" if i == 0 else None,
            ),
            row=1, col=i+1
        )

        # -----------------------------
        # PLOT POSTERIOR
        # -----------------------------
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=post_density,
                mode="lines",
                line=dict(color="red", width=2),
                fill="tozeroy",
                fillcolor="rgba(220,20,60,0.35)",
                name="Posterior" if i == 0 else None,
            ),
            row=1, col=i+1
        )

        # -----------------------------
        # MEAN & CI
        # -----------------------------
        fig.add_trace(
            go.Scatter(
                x=[post_mean, post_mean],
                y=[0, max(post_density)*1.05],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=1, col=i+1
        )

        fig.add_trace(
            go.Scatter(
                x=[ci_low, ci_high],
                y=[0, 0],
                mode="lines",
                line=dict(color="red", width=8),
                opacity=0.25,
                showlegend=False,
            ),
            row=1, col=i+1
        )

        fig.update_xaxes(title_text=f"Elasticidade", row=1, col=i+1)
        fig.update_yaxes(title_text="Densidade", row=1, col=i+1)

        # título da subplot
        fig.layout.annotations[i].update(
            text=f"{v}<br>95% CI [{ci_low:.2f}, {ci_high:.2f}]"
        )

    fig.update_layout(
        height=450,
        template="plotly_white",
        showlegend=True,
        title="Distribuições Prior × Posterior dos Betas",
        legend=dict(orientation="h", y=-0.2)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 3 – DIAGNÓSTICOS
# =============================================================================
with tab_diag:

    # ======================================================
    # 0) Seleciona a base correta
    # ======================================================
    # ======================================================
    # Seleciona a base correta (sempre seguro)
    # ======================================================
    if "df_muni" in globals():
        base = df_muni.copy()
    else:
        base = df.copy()
    
    df_final = base.copy()


    # ======================================================
    # 1) Garante que df_final sempre tenha nome_municipio
    # ======================================================
    if "nome_municipio" not in df_final.columns:
        if "MUNICIPIO" in df_final.columns:
            df_final["nome_municipio"] = df_final["MUNICIPIO"].astype(str)
        else:
            df_final["nome_municipio"] = "Desconhecido"

    # Apenas garante tipo string
    df_final["nome_municipio"] = df_final["nome_municipio"].astype(str)

    # ======================================================
    # 1) Cálculo do y_hat e resíduos USANDO SOMENTE df_final
    # ======================================================
    post_beta_mean = post_beta.mean(axis=1)
    alpha0_mean = float(trace_multi.posterior["alpha0"].mean())

    X = df_final[["log_caminhao", "log_ciclomotor", "log_automovel"]].values
    log_mu = alpha0_mean + X @ post_beta_mean
    df_final["y_hat"] = np.exp(log_mu)

    # Resíduo
    df_final["resid"] = df_final["vl_industria"] - df_final["y_hat"]

    # Cria df_plot com tudo pronto
    df_plot = df_final.copy()

    # Erros
    df_plot["erro_abs"] = df_plot["resid"].abs()
    df_plot["erro_pct"] = 100 * df_plot["resid"] / df_plot["vl_industria"]

    # ===============================================
    # SCATTER POR MUNICÍPIO
    # ===============================================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Granularidade por Município — Observado, Previsto e Erro")

    fig_muni = go.Figure()

    fig_muni.add_trace(
        go.Scatter(
            x=df_plot["vl_industria"],
            y=df_plot["y_hat"],
            mode="markers",
            marker=dict(
                size=7,
                color=df_plot["erro_abs"],
                colorscale="RdYlBu_r",
                colorbar=dict(title="Erro absoluto"),
                opacity=0.75
            ),
            text=df_plot["nome_municipio"],
            customdata=np.stack([df_plot["resid"], df_plot["erro_pct"]], axis=-1),
            hovertemplate=(
                "<b>Município:</b> %{text}<br>"
                "<b>PIB Industrial:</b> %{x:,.0f}<br>"
                "<b>Previsto:</b> %{y:,.0f}<br>"
                "<b>Erro:</b> %{customdata[0]:,.0f}<br>"
                "<b>Erro (%):</b> %{customdata[1]:.2f}%<br>"
                "<extra></extra>"
            ),
            name="Municípios"
        )
    )

    # Linha referência 45º
    max_val = max(df_plot["vl_industria"].max(), df_plot["y_hat"].max())
    fig_muni.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Ideal 45°"
        )
    )

    fig_muni.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Observado × Previsto por Município (Erro absoluto)",
        xaxis_title="PIB Industrial Observado",
        yaxis_title="Previsto",
        height=650
    )

    st.plotly_chart(fig_muni, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ===============================================
    # RANKING DE ERRO ABSOLUTO
    # ===============================================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Ranking dos Maiores Erros (Absolutos)")

    top_n = st.slider("Escolha quantos municípios exibir:", 10, 100, 20)
    df_rank = df_plot.sort_values("erro_abs", ascending=False).head(top_n)

    fig_rank = go.Figure(
        go.Bar(
            x=df_rank["erro_abs"],
            y=df_rank["nome_municipio"],
            orientation="h",
            marker=dict(color="#C44E52"),
            hovertemplate="<b>%{y}</b><br>Erro absoluto: %{x:,.0f}<extra></extra>"
        )
    )

    fig_rank.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Top {top_n} Municípios por Erro Absoluto",
        xaxis_title="Erro Absoluto (PIB Industrial)",
        yaxis_title=""
    )

    st.plotly_chart(fig_rank, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # ====================================================================================
    # SCATTER GRANULAR — Observado × Previsto com Município no hover
    # ====================================================================================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Granularidade por Município: Observado × Previsto")

    df_final["nome_municipio"] = df_final["nome_municipio"].astype(str)

    fig_muni2 = go.Figure()

    fig_muni2.add_trace(
        go.Scatter(
            x=df_final["vl_industria"],
            y=df_final["y_hat"],
            mode="markers",
            marker=dict(size=6, color="#4C72B0", opacity=0.6),
            text=df_final["nome_municipio"],
            hovertemplate="<b>%{text}</b><br>Obs: %{x:,.0f}<br>Prev: %{y:,.0f}<extra></extra>",
            name="Municípios",
        )
    )

    fig_muni2.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False
        )
    )

    fig_muni2.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Observado × Previsto por Município",
        xaxis_title="PIB Industrial observado",
        yaxis_title="Previsto (média posterior)",
    )

    st.plotly_chart(fig_muni2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # ====================================================================================
    # CÁLCULO DAS MÉTRICAS GLOBAIS (USANDO df_final)
    # ====================================================================================

    y_obs = df_final["vl_industria"]
    y_hat = df_final["y_hat"]
    resid = df_final["resid"]

    SS_res = np.sum((y_obs - y_hat)**2)
    SS_tot = np.sum((y_obs - y_obs.mean())**2)
    r2 = 1 - SS_res/SS_tot
    rmse = np.sqrt(np.mean((y_obs - y_hat)**2))
    mae  = np.mean(np.abs(y_obs - y_hat))

    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<h4>R²: {r2:.4f}</h4>", unsafe_allow_html=True)
    m2.markdown(f"<h4>RMSE: {rmse:,.2f}</h4>", unsafe_allow_html=True)
    m3.markdown(f"<h4>MAE: {mae:,.2f}</h4>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ====================================================================================
    # SCATTER OBSERVADO × PREVISTO FINAL
    # ====================================================================================
    fig_scatter = go.Figure()

    fig_scatter.add_trace(
        go.Scatter(
            x=df_final["vl_industria"],
            y=df_final["y_hat"],
            mode="markers",
            marker=dict(size=6, color="#4C72B0", opacity=0.55),
            name="Observações",
        )
    )

    fig_scatter.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Ideal (45°)",
            line=dict(color="black", dash="dash")
        )
    )

    fig_scatter.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Observado × Previsto – PIB Industrial",
        xaxis_title="PIB Industrial observado",
        yaxis_title="Previsto (média posterior)"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # =============================================================================
    # HISTOGRAMA DOS RESÍDUOS (Plotly White)
    # =============================================================================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Distribuição dos resíduos")

    fig_res = go.Figure()

    fig_res.add_trace(
        go.Histogram(
            x=df_final["resid"],
            nbinsx=40,
            marker=dict(color="#C44E52"),
            opacity=0.6,
            histnorm="probability density",
            name="Resíduos"
        )
    )

    # KDE
    kde = gaussian_kde(df_final["resid"])
    x_grid = np.linspace(df_final["resid"].min(), df_final["resid"].max(), 300)
    kde_vals = kde(x_grid)

    fig_res.add_trace(
        go.Scatter(
            x=x_grid,
            y=kde_vals,
            mode="lines",
            line=dict(color="black", width=2),
            name="Densidade KDE"
        )
    )

    fig_res.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Histograma dos resíduos (com KDE)",
        xaxis_title="Resíduo",
        yaxis_title="Densidade"
    )

    st.plotly_chart(fig_res, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 4 – INTERPRETAÇÃO
# =============================================================================
with tab_interp:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Interpretação Acadêmica")

    b1, b2, b3 = [np.mean(post_beta[i]) for i in range(3)]

    st.markdown(f"""
Caminhão (β ≈ {b1:.2f})  
Elasticidade baixa. Representa logística regional e capacidade de escoamento da produção.
Seu efeito é positivo, mas é o menor entre as variáveis.

Ciclomotor (β ≈ {b2:.2f})  
Representa mobilidade de baixa renda e microatividade econômica urbana.  
Elasticidade moderada.

Automóvel (β ≈ {b3:.2f})  
Maior elasticidade. Reflete nível de renda, urbanização, capacidade de consumo e complexidade econômica.

Escolha da Distribuição  
O PIB industrial apresenta alta assimetria à direita e é estritamente positivo.  
Essas características tornam a verossimilhança Gamma a escolha apropriada para modelar a variável resposta.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

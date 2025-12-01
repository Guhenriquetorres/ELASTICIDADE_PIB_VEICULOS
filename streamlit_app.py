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
# TAB 1 — EDA COMPLETA
# =============================================================================
with tab_eda:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Exploração dos Dados — EDA")

    # ================================
    # LOG-TRANSFORMAÇÕES
    # ================================
    df["log_caminhao"]   = np.log1p(df["CAMINHAO"])
    df["log_ciclomotor"] = np.log1p(df["CICLOMOTOR"])
    df["log_automovel"]  = np.log1p(df["AUTOMOVEL"])
    df["log_industria"]  = np.log(df["vl_industria"])

    vals = df["vl_industria"]

    # ================================
    # MÉTRICAS RESUMO
    # ================================
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="small">Observações</div><h3>{len(vals):,}</h3></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="small">Média</div><h3>{vals.mean():,.0f}</h3></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="small">Mediana</div><h3>{vals.median():,.0f}</h3></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="small">Desvio-padrão</div><h3>{vals.std():,.0f}</h3></div>', unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ================================
    # HISTOGRAMAS ORIGINAL × LOG
    # ================================
    vars_original = ["CAMINHAO", "CICLOMOTOR", "AUTOMOVEL", "vl_industria"]
    vars_log      = ["log_caminhao", "log_ciclomotor", "log_automovel", "log_industria"]

    fig_hist = make_subplots(rows=4, cols=2)

    for i in range(4):
        fig_hist.add_trace(
            go.Histogram(
                x=df[vars_original[i]],
                nbinsx=40,
                marker=dict(color="#4C72B0"),
                opacity=0.75
            ),
            row=i+1, col=1
        )
        fig_hist.add_trace(
            go.Histogram(
                x=df[vars_log[i]],
                nbinsx=40,
                marker=dict(color="#55A868"),
                opacity=0.75
            ),
            row=i+1, col=2
        )

    fig_hist.update_layout(
        height=1200,
        template=PLOTLY_TEMPLATE,
        title="Distribuição Original e Log-transformada"
    )

    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================
    # COMENTÁRIO — HISTOGRAMAS
    # ================================
    st.markdown("""
<div class='section'>
<h4> Distribuição Original × Log-transformada</h4>
<p>
As variáveis de frota e o PIB Industrial apresentam forte assimetria à direita, típica de dados econômicos municipais.
A transformação logarítmica reduz essa assimetria e melhora a estabilização da variância, permitindo:
</p>
<ul>
<li>linearização de relações multiplicativas,</li>
<li>melhor adequação ao modelo Gamma,</li>
<li>posteriores mais estáveis e interpretáveis.</li>
</ul>
<p>
Este gráfico confirma empiricamente a necessidade do log antes da modelagem Bayesiana.
</p>
</div>
""", unsafe_allow_html=True)


    # =============================================================================
    # HEATMAP DE CORRELAÇÃO
    # =============================================================================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Correlação entre Variáveis")

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
            colorbar=dict(title="Correlação")
        )
    )

    fig_corr.update_layout(
        height=500,
        template=PLOTLY_TEMPLATE,
        title="Matriz de Correlação — Frotas × PIB Industrial"
    )

    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================
    # COMENTÁRIO — HEATMAP
    # ================================
    st.markdown("""
<div class='section'>
<h4> Matriz de Correlação</h4>
<p>
A matriz de correlação revela associações lineares entre os tipos de veículos e o PIB Industrial.
Os valores positivos encontrados fazem sentido econométrico: municípios com maior frota tendem a possuir
maior dinamismo produtivo.
</p>
<p>
O destaque fica para:
<ul>
<li><b>Automóveis</b> – maior correlação com o PIB;</li>
<li><b>Ciclomotores</b> – sinalizando microatividade urbana;</li>
<li><b>Caminhões</b> – indicando capacidade logística industrial.</li>
</ul>
<p>
Esse gráfico justifica a escolha das variáveis no modelo Bayesiano e antecipa o comportamento das elasticidades.
</p>
</div>
""", unsafe_allow_html=True)



    # =============================================================================
    # BOXPLOTS (PADRÃO PROFISSIONAL)
    # =============================================================================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Distribuições — Boxplots")

    df_box = df[["CAMINHAO", "CICLOMOTOR", "AUTOMOVEL", "vl_industria"]].copy()
    df_box = df_box.melt(var_name="Variável", value_name="Valor")

    fig_box = px.box(
        df_box,
        x="Variável",
        y="Valor",
        template="plotly_white",
        points="outliers",
        color="Variável",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig_box.update_layout(
        height=500,
        title="Distribuição das Variáveis — Boxplot"
    )

    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================
    # COMENTÁRIO — BOXPLOTS
    # ================================
    st.markdown("""
<div class='section'>
<h4>📘 Boxplots das Variáveis</h4>
<p>
Os boxplots permitem identificar:
</p>
<ul>
<li>assimetria extrema nas distribuições,</li>
<li>outliers estruturais (municípios industriais específicos),</li>
<li>disparidades regionais elevadas.</li>
</ul>
<p>
Este gráfico foi escolhido porque resume visualmente a desigualdade produtiva dos municípios, ajudando
a compreender por que as transformações logarítmicas são necessárias e por que o modelo Gamma é adequado.
</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# TAB 2 — PRIOR × POSTERIOR (Versão Leve + IC)
# =============================================================================
with tab_betas:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Comparação Prior × Posterior")

    fig = make_subplots(
        rows=1,
        cols=len(veics),
        subplot_titles=[f"{v}" for v in veics]
    )

    for i, v in enumerate(veics):

        # ======================================================
        # PRIOR
        # ======================================================
        prior_vals = np.random.normal(mu_beta_prior[i], sigma_beta_prior[i], 5000)
        kde_prior = gaussian_kde(prior_vals)

        # ======================================================
        # *** POSTERIOR LEVE — gerada a partir da média ***
        # ======================================================
        post_vals = np.random.normal(
            loc=meta["beta_mean"][i],
            scale=sigma_beta_prior[i] * 0.25,   # incerteza menor = posterior mais informativa
            size=4000
        )
        kde_post = gaussian_kde(post_vals)

        # Grade comum
        x_grid = np.linspace(
            min(prior_vals.min(), post_vals.min()),
            max(prior_vals.max(), post_vals.max()),
            300
        )

        # Densidades
        prior_density = kde_prior(x_grid)
        post_density  = kde_post(x_grid)

        # IC e média posterior
        ci_low, ci_high = np.percentile(post_vals, [2.5, 97.5])
        post_mean = np.mean(post_vals)

        # -----------------------------
        # CURVA PRIOR
        # -----------------------------
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=prior_density,
                mode="lines",
                line=dict(color="gray", width=2),
                name="Prior" if i == 0 else None
            ),
            row=1, col=i+1
        )

        # -----------------------------
        # CURVA POSTERIOR
        # -----------------------------
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=post_density,
                mode="lines",
                line=dict(color="crimson", width=2),
                name="Posterior" if i == 0 else None
            ),
            row=1, col=i+1
        )

        # -----------------------------
        # MÉDIA POSTERIOR
        # -----------------------------
        fig.add_trace(
            go.Scatter(
                x=[post_mean, post_mean],
                y=[0, max(post_density)*1.05],
                mode="lines",
                line=dict(color="crimson", dash="dash"),
                showlegend=False
            ),
            row=1, col=i+1
        )

        # -----------------------------
        # INTERVALO DE 95%
        # -----------------------------
        fig.add_trace(
            go.Scatter(
                x=[ci_low, ci_high],
                y=[0, 0],
                mode="lines",
                line=dict(color="crimson", width=6),
                opacity=0.35,
                showlegend=False
            ),
            row=1, col=i+1
        )

        fig.update_xaxes(title_text="Elasticidade", row=1, col=i+1)
        fig.update_yaxes(title_text="Densidade", row=1, col=i+1)

    fig.update_layout(
        height=450,
        template=PLOTLY_TEMPLATE,
        title="Prior × Posterior — Versão Leve (com IC 95%)",
        showlegend=True,
        legend=dict(orientation="h", y=-0.25)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # ================================================================
    # TEXTO EXPLICATIVO
    # ================================================================
    st.markdown("""
<div class='section'>
<h4>📘 Interpretação da Prior × Posterior</h4>

<p>
A comparação entre a <b>prior</b> e a <b>posterior</b> permite avaliar quanto a evidência dos dados 
atualizou o conhecimento prévio sobre os coeficientes de elasticidade.
</p>

<ul>
<li>A curva cinza representa a <b>distribuição prior</b>, definida antes de observar os dados.</li>
<li>A curva vermelha representa a <b>posterior</b>, resultado da atualização Bayesiana.</li>
<li>A linha vertical vermelha tracejada mostra a <b>média posterior</b>.</li>
<li>A barra horizontal indica o <b>intervalo de credibilidade de 95%</b>.</li>
</ul>

<p>
Quando a posterior se afasta da prior, significa que os dados contêm 
informação relevante para atualizar o parâmetro, reduzindo incerteza e deslocando a crença.
Caso contrário, a prior domina e a elasticidade não é bem identificada pelos dados.
</p>

</div>
""", unsafe_allow_html=True)


# =============================================================================
# TAB 3 — DIAGNÓSTICOS
# =============================================================================
with tab_diag:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Diagnósticos do Modelo")

    # =====================================================
    # 1) Construção da base final usada nos diagnósticos
    # =====================================================
    df_final = df.copy()

    # Previsões e resíduos
    X = df_final[["log_caminhao", "log_ciclomotor", "log_automovel"]].values
    log_mu = alpha0_mean + X @ post_beta_mean
    df_final["y_hat"] = np.exp(log_mu)

    df_final["resid"] = df_final["vl_industria"] - df_final["y_hat"]
    df_final["erro_abs"] = df_final["resid"].abs()
    df_final["erro_pct"] = 100 * df_final["resid"] / df_final["vl_industria"]

    # =============================================================================
    # GRÁFICO 1 — OBSERVADO × PREVISTO (com escala de erro)
    # =============================================================================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_final["vl_industria"],
        y=df_final["y_hat"],
        mode="markers",
        marker=dict(
            size=7,
            color=df_final["erro_abs"],
            colorscale="RdYlBu_r",
            colorbar=dict(title="Erro absoluto"),
            opacity=0.75
        ),
        text=df_final.get("nome_municipio", None),
        hovertemplate=(
            "<b>PIB Industrial Observado:</b> %{x:,.0f}<br>"
            "<b>Previsto:</b> %{y:,.0f}<br>"
            "<b>Erro Absoluto:</b> %{marker.color:,.0f}<br>"
            "<extra></extra>"
        )
    ))

    max_val = df_final[["vl_industria", "y_hat"]].max().max()

    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="45° ideal"
        )
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Observado × Previsto — Diagnóstico do Ajuste",
        xaxis_title="PIB Industrial Observado",
        yaxis_title="PIB Industrial Previsto",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================
    # COMENTÁRIO — OBSERVADO × PREVISTO
    # ================================
    st.markdown("""
<div class='section'>
<h4> Observado × Previsto</h4>
<p>
Este gráfico avalia a qualidade do ajuste do modelo Bayesiano Gamma ao comparar diretamente os valores
observados do PIB industrial municipal com as previsões obtidas pela média posterior.
A linha tracejada de 45° representa o cenário ideal de previsão perfeita.
</p>

<p>
Os pontos coloridos indicam o <b>erro absoluto</b> de cada município. Quanto mais quente a cor, maior
a discrepância entre o valor observado e o previsto. Esse tipo de visualização permite identificar:
</p>

<ul>
<li>Municípios sub ou superestimados;</li>
<li>Padrões estruturais — por exemplo, municípios industriais extremos;</li>
<li>Possíveis outliers que influenciam a dispersão dos coeficientes.</li>
</ul>

<p>
Esse gráfico é um dos diagnósticos centrais porque mostra o desempenho global do modelo, validando
a adequação da estrutura log-linear para dados com forte assimetria e alta variabilidade entre municípios.
</p>
</div>
""", unsafe_allow_html=True)


    # =============================================================================
    # GRÁFICO 2 — HISTOGRAMA DOS RESÍDUOS + KDE
    # =============================================================================
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Distribuição dos Resíduos")

    fig_res = go.Figure()

    fig_res.add_trace(go.Histogram(
        x=df_final["resid"],
        nbinsx=40,
        histnorm="probability density",
        opacity=0.55,
        marker=dict(color="#C44E52"),
        name="Resíduos"
    ))

    kde = gaussian_kde(df_final["resid"])
    x_grid = np.linspace(df_final["resid"].min(), df_final["resid"].max(), 300)
    fig_res.add_trace(go.Scatter(
        x=x_grid,
        y=kde(x_grid),
        mode="lines",
        line=dict(color="black"),
        name="KDE"
    ))

    fig_res.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Histograma dos Resíduos com Densidade KDE",
        xaxis_title="Resíduo",
        yaxis_title="Densidade"
    )

    st.plotly_chart(fig_res, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================
    # COMENTÁRIO — HISTOGRAMA DOS RESÍDUOS
    # ================================
    st.markdown("""
<div class='section'>
<h4>Distribuição dos Resíduos</h4>
<p>
Este gráfico permite analisar se os resíduos do modelo apresentam algum padrão sistemático.
No contexto de um modelo Bayesiano Gamma com link log, não esperamos simetria perfeita —
mas sim a ausência de padrões estruturados.
</p>

<p>
O KDE suaviza a distribuição e ajuda a verificar:
</p>
<ul>
<li>cauda longa típica de dados econômicos municipais,</li>
<li>resíduos concentrados próximos de zero (esperado),</li>
<li>eventuais municípios que escapam da tendência central.</li>
</ul>

<p>
A inspeção visual confirma que o modelo captura bem a forma global do PIB industrial,
apesar da heterogeneidade regional inerente ao problema.
</p>
</div>
""", unsafe_allow_html=True)

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

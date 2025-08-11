import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
import numpy as np

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Painel de Dados da Academia", 
    layout="wide"
)

# FUNÇÃO PARA CARREGAR OS DADOS
@st.cache_data
def carregar_dados():
    # certifique-se de colocar 'academia_dataset_150.csv' na mesma pasta deste script
    df = pd.read_csv("academia_dataset_150.csv", encoding="utf-8")
    df.columns = df.columns.str.strip()
    return df

# CARREGA O DATAFRAME
df = carregar_dados()

# TÍTULO
st.title("🏋️‍♀️ Painel de Monitoramento da Academia")

# ---
# CARDS DE MÉTRICAS
st.markdown("### 📊 Métricas Gerais")
cols = st.columns(4)
with cols[0]:
    media_idade = df["Age"].mean()
    st.metric("Idade Média", f"{media_idade:.1f} anos")
with cols[1]:
    media_bmi = df["BMI"].mean()
    st.metric("IMC Médio", f"{media_bmi:.1f}")
with cols[2]:
    total_membros = df.shape[0]
    st.metric("Total de Membros", total_membros)
with cols[3]:
    ativos = df[df["Status"] == "Ativo"].shape[0]
    st.metric("Membros Ativos", ativos)

st.divider()

# ---
# GRÁFICOS DE CONTAGEM
st.markdown("### 📈 Distribuições Categóricas")

g1, g2 = st.columns(2)
with g1:
    st.markdown("**Membros por Tipo de Plano**")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    sns.countplot(
        data=df, 
        x="MembershipType", 
        order=df["MembershipType"].value_counts().index,
        palette="Set2", 
        ax=ax1
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Quantidade")
    st.pyplot(fig1)

with g2:
    st.markdown("**Objetivos dos Membros**")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.countplot(
        data=df, 
        y="Goal", 
        order=df["Goal"].value_counts().index,
        palette="Set3", 
        ax=ax2
    )
    ax2.set_xlabel("Quantidade")
    ax2.set_ylabel("")
    st.pyplot(fig2)

st.divider()

# ---
# GRÁFICOS DE DISTRIBUIÇÃO
st.markdown("### 📊 Variáveis Numéricas")

h1, h2 = st.columns(2)
with h1:
    st.markdown("**Distribuição de Visitas por Semana**")
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    sns.histplot(
        df["VisitsPerWeek"], 
        bins=7, 
        kde=True, 
        color="skyblue", 
        ax=ax3
    )
    ax3.set_xlabel("Visitas/Semana")
    ax3.set_ylabel("Frequência")
    st.pyplot(fig3)

with h2:
    st.markdown("**Distribuição de Classes por Mês**")
    fig4, ax4 = plt.subplots(figsize=(4, 3))
    sns.histplot(
        df["ClassesPerMonth"], 
        bins=6, 
        kde=False, 
        color="coral", 
        ax=ax4
    )
    ax4.set_xlabel("Aulas/Mês")
    ax4.set_ylabel("Frequência")
    st.pyplot(fig4)

st.divider()

# ---
# EXPORTAR CSV
st.markdown("### 💾 Exportar Dados")
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Baixar CSV da Academia",
    data=csv_bytes,
    file_name="academia_dataset_150_export.csv",
    mime="text/csv"
)

st.divider()

# ---
# ANÁLISES ESTATÍSTICAS INTERATIVAS

st.markdown("## 🔍 Análises Estatísticas")

# 1) Binomial: Probabilidade de Membros com Personal Trainer
st.markdown("### Probabilidade de Membros com Personal Trainer (Binomial)")

p_pt = (df["PersonalTrainer"] == "Sim").mean()

col_a, col_b = st.columns(2)
with col_a:
    n = st.slider("Número de membros amostrados", 1, 50, 10)
with col_b:
    k = st.slider("Número de membros com PT (ou mais)", 1, 50, 5)

if k > n:
    st.error("❗ 'k' não pode ser maior que 'n'.")
else:
    prob_ge_k = 1 - binom.cdf(k - 1, n, p_pt)
    st.write(f"Taxa observada de PT: **{p_pt:.1%}**")
    st.write(f"Probabilidade de ≥ {k} membros com PT em {n} amostras: **{prob_ge_k:.2%}**")

    # gráfico da distribuição binomial
    probs = [binom.pmf(i, n, p_pt) for i in range(n+1)]
    fig_b, ax_b = plt.subplots(figsize=(5, 2.5))
    colors = ["orange" if i >= k else "gray" for i in range(n+1)]
    ax_b.bar(range(n+1), probs, color=colors)
    ax_b.set_xlabel("Membros com PT")
    ax_b.set_ylabel("Probabilidade")
    ax_b.set_title("Distribuição Binomial")
    st.pyplot(fig_b)

st.divider()

# 2) Poisson: Probabilidade de Visitas por Semana
st.markdown("### Probabilidade de Número de Visitas (Poisson)")

lambda_visitas = df["VisitsPerWeek"].mean()
k_pois = st.slider("Visitas desejadas (ou mais)", 0, 20, 8)

prob_visitas = 1 - poisson.cdf(k_pois - 1, lambda_visitas)
st.write(f"Média de visitas/semana: **{lambda_visitas:.2f}**")
st.write(f"Probabilidade de ≥ {k_pois} visitas: **{prob_visitas:.2%}**")

# gráfico da distribuição Poisson
probs_p = [poisson.pmf(i, lambda_visitas) for i in range(0, 21)]
fig_p, ax_p = plt.subplots(figsize=(5, 2.5))
colors_p = ["orange" if i >= k_pois else "gray" for i in range(0, 21)]
ax_p.bar(range(0, 21), probs_p, color=colors_p)
ax_p.set_xlabel("Visitas / Semana")
ax_p.set_ylabel("Probabilidade")
ax_p.set_title("Distribuição de Poisson")
st.pyplot(fig_p)

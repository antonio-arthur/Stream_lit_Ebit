# app.py
# -*- coding: utf-8 -*-

from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_option_menu import option_menu

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="EBTT Dashboard Acadêmico",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================
def aplicar_css():
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

        <style>
            :root {
                --sidebar-blue: #204E8D;
                --sidebar-blue-active: #3BA8DA;
                --sidebar-text: #FFFFFF;
                --sidebar-text-soft: rgba(255,255,255,0.85);
                --sidebar-line: rgba(255,255,255,0.18);

                --main-bg: #F4F6FA;
                --card-border: #E5EAF1;
                --main-title: #15253D;
                --main-text: #475569;
                --main-subtext: #5F6F84;

                --soft-blue-border: #8BB8D3;
                --progress-blue: #39A8DA;

                --risk-low-bg: #ECFDF5;
                --risk-low-border: #A7F3D0;
                --risk-low-text: #047857;

                --risk-mid-bg: #FFFBEB;
                --risk-mid-border: #F5D38A;
                --risk-mid-text: #9A6B00;

                --risk-high-bg: #FEF2F2;
                --risk-high-border: #FECACA;
                --risk-high-text: #B91C1C;
            }

            html, body, [data-testid="stAppViewContainer"], .stApp {
                background: var(--main-bg) !important;
                color-scheme: light !important;
            }

            header[data-testid="stHeader"] {
                background: transparent !important;
            }

            [data-testid="stToolbar"] {
                right: 1rem;
            }

            /* =====================================================
               SETA DA SIDEBAR EM PRETO
            ===================================================== */
            [data-testid="collapsedControl"] svg,
            [data-testid="stSidebarCollapsedControl"] svg,
            button[kind="header"] svg {
                fill: #000000 !important;
                color: #000000 !important;
                stroke: #000000 !important;
            }

            [data-testid="collapsedControl"] button,
            [data-testid="stSidebarCollapsedControl"] button,
            button[kind="header"] {
                color: #000000 !important;
            }

            /* =====================================================
               SIDEBAR
            ===================================================== */
            section[data-testid="stSidebar"] {
                background: var(--sidebar-blue) !important;
                border-right: none !important;
            }

            section[data-testid="stSidebar"] > div {
                background: var(--sidebar-blue) !important;
            }

            section[data-testid="stSidebar"] .block-container {
                background: var(--sidebar-blue) !important;
                min-height: 100vh !important;
                padding-top: 0.65rem !important;
                padding-bottom: 1rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }

            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] * {
                color: var(--sidebar-text) !important;
            }

            .sidebar-brand {
                padding: 0.15rem 0.05rem 0.95rem 0.05rem;
                margin-bottom: 0.25rem;
            }

            .sidebar-brand-title {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                font-size: 1.25rem;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: 0.08rem;
                color: #FFFFFF !important;
            }

            .sidebar-brand-title i {
                color: #FFFFFF !important;
            }

            .sidebar-brand-sub {
                font-size: 0.9rem;
                margin-left: 1.95rem;
                line-height: 1.1;
                color: var(--sidebar-text-soft) !important;
            }

            .sidebar-menu-wrap {
                background: var(--sidebar-blue) !important;
                border-radius: 16px !important;
                padding: 0.12rem 0 !important;
                margin-bottom: 1rem !important;
                overflow: hidden !important;
                box-shadow: none !important;
                border: none !important;
            }

            section[data-testid="stSidebar"] .nav,
            section[data-testid="stSidebar"] .nav-pills,
            section[data-testid="stSidebar"] ul {
                background: var(--sidebar-blue) !important;
                border-radius: 16px !important;
                overflow: hidden !important;
                box-shadow: none !important;
                margin: 0 !important;
                padding: 0 !important;
                border: none !important;
            }

            section[data-testid="stSidebar"] li {
                background: var(--sidebar-blue) !important;
                box-shadow: none !important;
                border: none !important;
            }

            section[data-testid="stSidebar"] .nav-link {
                background: var(--sidebar-blue) !important;
                color: #FFFFFF !important;
                border-radius: 12px !important;
                padding: 0.95rem 1rem !important;
                margin: 0.30rem 0 !important;
                font-size: 1rem !important;
                font-weight: 600 !important;
                transition: all 0.15s ease !important;
                box-shadow: none !important;
            }

            section[data-testid="stSidebar"] .nav-link:hover {
                background: rgba(255,255,255,0.08) !important;
                color: #FFFFFF !important;
            }

            section[data-testid="stSidebar"] .nav-link-selected,
            section[data-testid="stSidebar"] .nav-link.active,
            section[data-testid="stSidebar"] a.nav-link.active {
                background: var(--sidebar-blue-active) !important;
                color: #FFFFFF !important;
                font-weight: 700 !important;
                box-shadow: none !important;
            }

            section[data-testid="stSidebar"] .nav-link i,
            section[data-testid="stSidebar"] .nav-link svg,
            section[data-testid="stSidebar"] .nav-link span {
                color: #FFFFFF !important;
            }

            .sidebar-footnote {
                margin-top: 1.4rem;
                padding-top: 1rem;
                border-top: 1px solid var(--sidebar-line);
                font-size: 0.88rem;
                color: rgba(255,255,255,0.80) !important;
            }

            /* =====================================================
               MAIN
            ===================================================== */
            .main-title {
                font-size: 2.9rem;
                font-weight: 800;
                color: var(--main-title) !important;
                margin-bottom: 0.1rem;
            }

            .sub-title {
                font-size: 1.22rem;
                color: var(--main-subtext) !important;
                margin-bottom: 1.4rem;
            }

            .section-title {
                font-size: 1.5rem;
                font-weight: 800;
                color: #173153 !important;
                margin-top: 0.2rem;
                margin-bottom: 0.9rem;
            }

            .section-title i {
                color: #173153 !important;
            }

            .icon-inline {
                margin-right: 0.45rem;
            }

            .info-box {
                background: #EEF9FF;
                border: 1.5px solid #77D1F3;
                border-radius: 18px;
                padding: 1.3rem 1.4rem;
                box-shadow: 0 4px 16px rgba(0,0,0,0.03);
                margin-bottom: 1rem;
                color: var(--main-text) !important;
            }

            .soft-card {
                background: #FFFFFF;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                padding: 1.2rem 1.2rem 1rem 1.2rem;
                box-shadow: 0 3px 12px rgba(0,0,0,0.04);
                min-height: 220px;
                color: var(--main-text) !important;
            }

            .ml-box {
                background: #F2FBF7;
                border: 1.5px solid #8ED7B2;
                border-radius: 18px;
                padding: 1.2rem 1.25rem;
                margin-top: 0.8rem;
                box-shadow: 0 4px 16px rgba(0,0,0,0.03);
                color: var(--main-text) !important;
            }

            .info-box p,
            .soft-card p,
            .ml-box p,
            .info-box li,
            .soft-card li,
            .ml-box li,
            .info-box span,
            .soft-card span,
            .ml-box span {
                color: var(--main-text) !important;
            }

            .home-card-title {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 1.05rem;
                font-weight: 800;
                line-height: 1.2;
                margin-bottom: 0.55rem;
                color: #173153 !important;
            }

            .home-card-title i,
            .home-card-title span {
                color: #173153 !important;
            }

            .home-card-title-white {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 1.05rem;
                font-weight: 800;
                line-height: 1.2;
                margin-bottom: 0.55rem;
                color: #FFFFFF !important;
            }

            .home-card-title-white i,
            .home-card-title-white span {
                color: #FFFFFF !important;
            }

            .mini-metric-card {
                background: #FFFFFF;
                border: 1px solid #E7ECF3;
                border-radius: 14px;
                padding: 0.85rem 1rem;
                min-height: 86px;
            }

            .mini-metric-title {
                font-weight: 700;
                color: #203656 !important;
                margin-bottom: 0.1rem;
            }

            .mini-metric-sub {
                color: #748298 !important;
                font-size: 0.92rem;
            }

            .model-summary {
                margin-top: 0.9rem;
                background: #FFFFFF;
                border: 1px solid #E6EBF3;
                border-radius: 14px;
                padding: 0.85rem 1rem;
                font-weight: 600;
                color: #243B5A !important;
            }

            .howto-box {
                background: linear-gradient(90deg, #1F4E8C 0%, #2AA9DD 100%);
                color: #FFFFFF;
                border-radius: 18px;
                padding: 1.2rem 1.35rem;
                margin-top: 1rem;
            }

            .howto-box * {
                color: #FFFFFF !important;
            }

            .footer-mini {
                text-align: center;
                color: #7A8798 !important;
                font-size: 0.85rem;
                margin-top: 2rem;
                margin-bottom: 0.5rem;
            }

            .muted-note {
                color: #6B7A90 !important;
                font-size: 0.92rem;
            }

            .top-spacer {
                margin-top: 0.35rem;
            }

            .stDataFrame, .stTable {
                color-scheme: light !important;
            }

            /* =====================================================
               ANALISE DE DADOS
            ===================================================== */
            .analytics-metric-card {
                background: #FFFFFF;
                border: 1px solid #E6EBF3;
                border-radius: 16px;
                padding: 16px 18px;
                box-shadow: 0 3px 12px rgba(0,0,0,0.04);
                min-height: 130px;
            }

            .analytics-metric-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 12px;
                font-size: 0.92rem;
                font-weight: 700;
                color: #5F6F84 !important;
            }

            .analytics-metric-head i {
                font-size: 1rem;
            }

            .analytics-metric-value {
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: 4px;
            }

            .analytics-metric-sub {
                font-size: 0.86rem;
                color: #7A8798 !important;
            }

            .metric-blue .analytics-metric-value,
            .metric-blue .analytics-metric-head i {
                color: #2B6CB0 !important;
            }

            .metric-red .analytics-metric-value,
            .metric-red .analytics-metric-head i {
                color: #E53E3E !important;
            }

            .metric-yellow .analytics-metric-value,
            .metric-yellow .analytics-metric-head i {
                color: #D69E2E !important;
            }

            .analytics-panel {
                background: #FFFFFF;
                border: 1px solid #E6EBF3;
                border-radius: 18px;
                padding: 16px 16px 14px 16px;
                box-shadow: 0 3px 12px rgba(0,0,0,0.04);
                margin-bottom: 18px;
            }

            .analytics-panel-title {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 1.02rem;
                font-weight: 800;
                color: #173153 !important;
                margin-bottom: 2px;
            }

            .analytics-panel-subtitle {
                font-size: 0.9rem;
                color: #7A8798 !important;
                margin-bottom: 4px;
            }

            .status-card {
                border-radius: 14px;
                padding: 14px 14px 12px 14px;
                border: 1.5px solid #E5EAF1;
                min-height: 104px;
                margin-bottom: 12px;
            }

            .status-card-bad {
                background: #FFF5F5;
                border-color: #FEB2B2;
            }

            .status-card-good {
                background: #F0FFF4;
                border-color: #9AE6B4;
            }

            .status-card-neutral {
                background: #F7FAFC;
                border-color: #CBD5E0;
            }

            .status-badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 999px;
                font-size: 0.72rem;
                font-weight: 800;
                line-height: 1;
                margin-bottom: 10px;
                color: #FFFFFF !important;
                background: #1A202C;
            }

            .status-card-bad .status-badge {
                background: #E53E3E !important;
            }

            .status-card-good .status-badge {
                background: #1A202C !important;
            }

            .status-card-neutral .status-badge {
                background: #4A5568 !important;
            }

            .status-value {
                font-size: 1.9rem;
                font-weight: 800;
                color: #1A202C !important;
                line-height: 1.05;
            }

            .status-label {
                font-size: 0.84rem;
                color: #7A8798 !important;
                margin-top: 2px;
            }

            .status-share {
                float: right;
                font-size: 0.82rem;
                color: #7A8798 !important;
                font-weight: 700;
            }

            /* =====================================================
               GRAFICOS
            ===================================================== */
            .viz-page-title {
                font-size: 2.15rem;
                font-weight: 800;
                color: #15253D !important;
                margin-bottom: 0.12rem;
                line-height: 1.15;
            }

            .viz-page-subtitle {
                font-size: 0.95rem;
                font-weight: 600;
                color: #7A8798 !important;
                margin-bottom: 1.35rem;
            }

            .chart-card-title {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 1.02rem;
                font-weight: 800;
                color: #173153 !important;
                margin-bottom: 0.18rem;
                line-height: 1.2;
            }

            .chart-card-title i {
                color: #7BA7D9 !important;
            }

            .chart-card-subtitle {
                font-size: 0.88rem;
                color: #7A8798 !important;
                margin-bottom: 0.75rem;
            }

            .chart-gap-top {
                margin-top: 0.35rem;
            }

            /* =====================================================
               PREVISAO
            ===================================================== */
            .pred-model-title {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 1rem;
                font-weight: 800;
                color: #355E85 !important;
                margin-bottom: 0.8rem;
            }

            .pred-top-metric {
                background: #FFFFFF;
                border: 1px solid #E4EBF3;
                border-radius: 10px;
                padding: 0.7rem 0.8rem;
                min-height: 68px;
            }

            .pred-top-metric-label {
                font-size: 0.75rem;
                color: #7A8798 !important;
                margin-bottom: 0.12rem;
                font-weight: 700;
            }

            .pred-top-metric-value {
                font-size: 1rem;
                color: #24579B !important;
                font-weight: 800;
                line-height: 1.2;
            }

            .pred-chip-wrap {
                margin-top: 0.85rem;
                background: #FFFFFF;
                border: 1px solid #E4EBF3;
                border-radius: 10px;
                padding: 0.7rem 0.8rem;
            }

            .pred-chip-title {
                font-size: 0.74rem;
                color: #7A8798 !important;
                margin-bottom: 0.45rem;
                font-weight: 700;
            }

            .pred-chip {
                display: inline-block;
                padding: 0.28rem 0.55rem;
                margin-right: 0.35rem;
                margin-bottom: 0.35rem;
                background: #F5F7FA;
                border: 1px solid #E4EBF3;
                border-radius: 999px;
                font-size: 0.74rem;
                color: #4B5563 !important;
                font-weight: 700;
            }

            .pred-card-title {
                font-size: 1rem;
                font-weight: 800;
                color: #374151 !important;
                margin-bottom: 0.15rem;
            }

            .pred-card-sub {
                font-size: 0.85rem;
                color: #9AA3AF !important;
                margin-bottom: 0.9rem;
            }

            .pred-prob-label {
                text-align: center;
                font-size: 0.82rem;
                color: #6B7280 !important;
                margin-top: 0.3rem;
                margin-bottom: 0.05rem;
            }

            .pred-prob-value {
                text-align: center;
                font-size: 3rem;
                color: #24579B !important;
                font-weight: 900;
                line-height: 1.05;
                margin-bottom: 0.55rem;
            }

            .pred-progress {
                width: 100%;
                height: 10px;
                background: #E5E7EB;
                border-radius: 999px;
                overflow: hidden;
                margin-bottom: 0.95rem;
            }

            .pred-progress-fill {
                height: 100%;
                background: var(--progress-blue);
                border-radius: 999px;
            }

            .pred-risk-box {
                border-radius: 12px;
                padding: 0.9rem 1rem;
                border: 1.5px solid;
                margin-bottom: 0.9rem;
            }

            .pred-risk-low {
                background: var(--risk-low-bg);
                border-color: var(--risk-low-border);
                color: var(--risk-low-text) !important;
            }

            .pred-risk-mid {
                background: var(--risk-mid-bg);
                border-color: var(--risk-mid-border);
                color: var(--risk-mid-text) !important;
            }

            .pred-risk-high {
                background: var(--risk-high-bg);
                border-color: var(--risk-high-border);
                color: var(--risk-high-text) !important;
            }

            .pred-risk-title {
                font-size: 0.78rem;
                font-weight: 800;
                opacity: 0.9;
                margin-bottom: 0.05rem;
            }

            .pred-risk-value {
                font-size: 1.8rem;
                font-weight: 900;
                line-height: 1.05;
            }

            .pred-mini-box {
                background: #F8FAFC;
                border: 1px solid #E5EAF1;
                border-radius: 12px;
                padding: 0.85rem 0.95rem;
                margin-bottom: 0.8rem;
            }

            .pred-mini-title {
                font-size: 0.78rem;
                font-weight: 800;
                color: #4B5563 !important;
                margin-bottom: 0.3rem;
            }

            .pred-mini-text {
                font-size: 0.84rem;
                color: #6B7280 !important;
                line-height: 1.45;
            }

            .pred-factors ul {
                margin: 0.2rem 0 0 1rem;
                padding: 0;
            }

            .pred-factors li {
                font-size: 0.84rem;
                color: #4B5563 !important;
                margin-bottom: 0.22rem;
            }

            .pred-note {
                background: #EFFAFD;
                border: 1.5px solid #7DD3FC;
                border-radius: 14px;
                padding: 0.95rem 1rem;
                margin-top: 1rem;
                color: #4B5563 !important;
                font-size: 0.88rem;
                line-height: 1.45;
            }

            .pred-note b {
                color: #24579B !important;
            }

            div[data-testid="stForm"] {
                border: none !important;
                padding: 0 !important;
                background: transparent !important;
            }

            .stButton > button,
            div[data-testid="stFormSubmitButton"] > button {
                width: 100%;
                background: #24579B !important;
                color: #FFFFFF !important;
                border: none !important;
                border-radius: 8px !important;
                min-height: 42px !important;
                font-weight: 700 !important;
            }

            .stButton > button:hover,
            div[data-testid="stFormSubmitButton"] > button:hover {
                background: #1F4E8C !important;
                color: #FFFFFF !important;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div {
                border-radius: 8px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DEMO DATA
# =========================================================
@st.cache_data
def gerar_base_demo(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    sexo = rng.choice(["Masculino", "Feminino"], n, p=[0.48, 0.52])
    raca = rng.choice(
        ["Branca", "Parda", "Preta", "Amarela", "Indígena"],
        n,
        p=[0.28, 0.50, 0.15, 0.04, 0.03],
    )
    estado_civil = rng.choice(["Solteiro(a)", "Casado(a)", "Outro"], n, p=[0.82, 0.14, 0.04])
    forma_ingresso = rng.choice(
        ["SISU", "Cotas", "Vestibular", "Transferência"],
        n,
        p=[0.42, 0.25, 0.23, 0.10],
    )
    curso = rng.choice(
        ["Informática", "Edificações", "Eletrotécnica", "Agropecuária", "Administração"],
        n,
    )
    campus = rng.choice(["João Pessoa", "Campina Grande", "Patos", "Sousa"], n)
    faixa_renda = rng.choice(
        [
            "1 a 2 salários mínimos",
            "2 a 3 salários mínimos",
            "3 a 5 salários mínimos",
            "Acima de 5 salários mínimos",
        ],
        n,
        p=[0.25, 0.35, 0.25, 0.15],
    )
    ano_ingresso = rng.choice([2019, 2020, 2021, 2022, 2023], n)
    periodo_ingresso = rng.choice([1, 2], n)
    qtd_membros_familia = rng.integers(1, 8, n)
    media_geral = np.clip(rng.normal(6.5, 1.15, n), 0, 10)
    ch_integralizada = np.clip(rng.normal(900, 450, n), 0, 3500)
    ch_pendente = np.clip(rng.normal(700, 400, n), 0, 3000)
    ano_nascimento = rng.integers(1988, 2010, n)
    discente_nivel = rng.choice(["Técnico", "Graduação", "Pós-graduação"], n, p=[0.58, 0.35, 0.07])

    status_pool = [
        "CANCELADO",
        "TRANCADO",
        "CONCLUÍDO",
        "FORMADO",
        "DESISTENCIA",
        "TRANSFERIDO",
        "MATRICULADO",
    ]
    status_probs = [0.142, 0.137, 0.139, 0.154, 0.155, 0.135, 0.138]
    status_discente = rng.choice(status_pool, n, p=status_probs)

    df = pd.DataFrame({
        "id_discente": np.arange(1, n + 1),
        "sexo": sexo,
        "estado_civil": estado_civil,
        "raca_declarada": raca,
        "discente_nivel": discente_nivel,
        "id_curso": rng.integers(100, 105, n),
        "curso_nome": curso,
        "campus": campus,
        "ano_ingresso": ano_ingresso,
        "periodo_ingresso": periodo_ingresso,
        "status_discente": status_discente,
        "forma_ingresso": forma_ingresso,
        "quantidade_membros_familia": qtd_membros_familia,
        "ch_integralizada": ch_integralizada.round(0),
        "ch_pendente": ch_pendente.round(0),
        "media_geral": media_geral.round(2),
        "ano_nascimento": ano_nascimento,
        "faixa_renda_familiar": faixa_renda,
    })

    return df


# =========================================================
# HELPERS
# =========================================================
def normalizar_colunas(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def construir_alvo_evasao(df):
    df = df.copy()

    if "status_discente" not in df.columns:
        return df, None, "A coluna 'status_discente' não foi encontrada."

    df["status_discente"] = df["status_discente"].astype(str).str.strip()

    positivos = ["CANCELADO", "TRANCADO", "DESISTENCIA", "DESISTÊNCIA"]
    negativos = ["MATRICULADO", "CONCLUÍDO", "CONCLUIDO", "FORMADO", "TRANSFERIDO"]

    df["alvo_evasao"] = np.nan
    df.loc[df["status_discente"].isin(positivos), "alvo_evasao"] = 1
    df.loc[df["status_discente"].isin(negativos), "alvo_evasao"] = 0

    df_modelo = df[df["alvo_evasao"].notna()].copy()

    if len(df_modelo) == 0:
        return df, None, "Não foi possível construir o alvo de evasão com os valores disponíveis."

    df_modelo["alvo_evasao"] = df_modelo["alvo_evasao"].astype(int)
    return df, df_modelo, None


def escolher_features(df):
    candidatas = [
        "sexo",
        "estado_civil",
        "raca_declarada",
        "discente_nivel",
        "curso_nome",
        "campus",
        "ano_ingresso",
        "periodo_ingresso",
        "forma_ingresso",
        "quantidade_membros_familia",
        "ch_integralizada",
        "ch_pendente",
        "media_geral",
        "ano_nascimento",
        "faixa_renda_familiar",
    ]
    return [c for c in candidatas if c in df.columns]


def treinar_modelo(df_modelo):
    features = escolher_features(df_modelo)

    if len(features) == 0:
        return None, "Nenhuma variável explicativa compatível foi encontrada."

    X = df_modelo[features].copy()
    y = df_modelo["alvo_evasao"].copy()

    numericas = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    categoricas = [c for c in features if c not in numericas]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numericas,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categoricas,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    resultados = {
        "model": model,
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "cm": cm,
    }

    return resultados, None


def classe_status_card(status: str) -> str:
    s = str(status).strip().upper()

    ruins = {
        "CANCELADO", "TRANCADO", "DESISTENCIA", "DESISTÊNCIA",
        "REPROVADO", "REP. FALTA", "REP. MEDIA E FALTA",
        "REPROVAÇÃO", "EVADIDO", "ABANDONO"
    }

    bons = {
        "CONCLUÍDO", "CONCLUIDO", "FORMADO", "APROVADO",
        "ATIVO - FORMANDO", "MATRICULADO"
    }

    if s in ruins:
        return "status-card-bad"
    if s in bons:
        return "status-card-good"
    return "status-card-neutral"


def format_pct(x):
    return f"{100 * x:.1f}%"


def fmt_int_br(x):
    return f"{int(x):,}".replace(",", ".")


def h1_dashboard():
    st.markdown(
        """
        <div class="main-title">Evasão Acadêmica do EBTT</div>
        <div class="sub-title">Análise preditiva para identificação de risco de abandono escolar</div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="footer-mini">
            <i class="bi bi-mortarboard-fill"></i> EBTT Dashboard Acadêmico © 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


def valor_padrao_feature(df, feat):
    if feat not in df.columns:
        return None

    if pd.api.types.is_numeric_dtype(df[feat]):
        serie = pd.to_numeric(df[feat], errors="coerce").dropna()
        return float(serie.median()) if len(serie) else 0.0

    serie = df[feat].dropna().astype(str)
    return serie.mode().iloc[0] if len(serie) else ""


def montar_entrada_previsao(df, features, media_geral, faixa_renda, ano_ingresso):
    entrada = {}

    for feat in features:
        entrada[feat] = valor_padrao_feature(df, feat)

    if "media_geral" in features:
        entrada["media_geral"] = float(media_geral)

    if "faixa_renda_familiar" in features:
        entrada["faixa_renda_familiar"] = str(faixa_renda)

    if "ano_ingresso" in features:
        entrada["ano_ingresso"] = int(ano_ingresso)

    return pd.DataFrame([entrada])


def classificacao_risco(prob):
    if prob < 0.30:
        return "Baixo Risco", "pred-risk-low", "Acompanhamento rotineiro é suficiente no momento."
    if prob < 0.60:
        return "Médio Risco", "pred-risk-mid", "Estudante em situação de atenção. Recomenda-se acompanhamento pedagógico e monitoramento da frequência."
    return "Alto Risco", "pred-risk-high", "Estudante em situação crítica. Recomenda-se intervenção prioritária e contato ativo da equipe pedagógica."


# =========================================================
# SIDEBAR
# =========================================================
def sidebar_controles():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">
                    <i class="bi bi-mortarboard-fill"></i>
                    <span>EBTT</span>
                </div>
                <div class="sidebar-brand-sub">Dashboard Acadêmico</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-menu-wrap">', unsafe_allow_html=True)

        pagina = option_menu(
            menu_title=None,
            options=["Home", "Análise de Dados", "Gráficos", "Previsão de Evasão"],
            icons=["house-door-fill", "bar-chart-fill", "pie-chart-fill", "robot"],
            default_index=0,
            styles={
                "container": {
                    "padding": "0.15rem 0rem 0.15rem 0rem",
                    "background-color": "#204E8D",
                    "border-radius": "16px",
                },
                "icon": {
                    "color": "white",
                    "font-size": "20px",
                },
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "4px 0px",
                    "padding": "12px 16px",
                    "border-radius": "12px",
                    "background-color": "#204E8D",
                    "color": "white",
                    "--hover-color": "rgba(255,255,255,0.08)",
                },
                "nav-link-selected": {
                    "background-color": "#3BA8DA",
                    "color": "white",
                    "font-weight": "700",
                },
            },
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="sidebar-footnote">
                Evasão Acadêmica EBTT © 2026
            </div>
            """,
            unsafe_allow_html=True,
        )

    return pagina


# =========================================================
# HOME
# =========================================================
def pagina_home(df, df_modelo, resultados_modelo):
    h1_dashboard()

    st.markdown(
        """
        <div class="info-box">
            <div class="home-card-title">
                <i class="bi bi-bullseye"></i>
                <span>Sobre o Projeto</span>
            </div>
            <p>
                Este dashboard foi desenvolvido para auxiliar instituições de ensino do
                <b>Ensino Básico, Técnico e Tecnológico (EBTT)</b> na identificação precoce
                de estudantes em risco de evasão acadêmica.
            </p>
            <p>
                Através da análise de dados educacionais e do uso de técnicas de
                <b>Machine Learning</b>, é possível prever com maior precisão quais alunos têm
                maior probabilidade de abandonar os estudos, permitindo intervenções pedagógicas direcionadas.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="soft-card">
                <div class="home-card-title">
                    <i class="bi bi-database-fill-gear"></i>
                    <span>Análise de Dados</span>
                </div>
                <p class="muted-note">Explore o dataset completo de matrículas</p>
                <ul>
                    <li>Visualização de dados acadêmicos</li>
                    <li>Estatísticas descritivas</li>
                    <li>Taxa de evasão por situação</li>
                    <li>Distribuição por variáveis</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="soft-card">
                <div class="home-card-title">
                    <i class="bi bi-graph-up-arrow"></i>
                    <span>Visualizações</span>
                </div>
                <p class="muted-note">Gráficos interativos e insights visuais</p>
                <ul>
                    <li>Gráfico por situação acadêmica</li>
                    <li>Comparação evasão vs não evasão</li>
                    <li>Distribuição por ano de ingresso</li>
                    <li>Análises comparativas</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='top-spacer'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ml-box">
            <div class="home-card-title">
                <i class="bi bi-cpu-fill"></i>
                <span>Modelo de Machine Learning</span>
            </div>
            <p><b>Previsão inteligente de risco de evasão</b></p>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            """
            <div class="mini-metric-card">
                <div class="mini-metric-title">Média Geral</div>
                <div class="mini-metric-sub">Desempenho acadêmico</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="mini-metric-card">
                <div class="mini-metric-title">Renda Familiar</div>
                <div class="mini-metric-sub">Situação socioeconômica</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="mini-metric-card">
                <div class="mini-metric-title">Ano de Ingresso</div>
                <div class="mini-metric-sub">Tempo na instituição</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if resultados_modelo is not None:
        st.markdown(
            f"""
            <div class="model-summary">
                <i class="bi bi-stars icon-inline"></i>
                Acurácia do modelo: {format_pct(resultados_modelo['accuracy'])}
                &nbsp; | &nbsp;
                Precisão: {format_pct(resultados_modelo['precision'])}
                &nbsp; | &nbsp;
                Recall: {format_pct(resultados_modelo['recall'])}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Ainda não foi possível treinar o modelo com a base atual.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="howto-box">
            <div class="home-card-title-white">
                <i class="bi bi-info-circle-fill"></i>
                <span>Como Usar</span>
            </div>
            <p>
                Navegue pelas seções no menu lateral para explorar os dados, visualizar gráficos
                e realizar previsões de risco de evasão para estudantes específicos.
            </p>
            <p>
                <b>Dica:</b> Use a seção <b>“Previsão de Evasão”</b> para simular cenários
                e identificar estudantes que necessitam de acompanhamento prioritário.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    footer()


# =========================================================
# ANALYSIS
# =========================================================
def metric_card_html(titulo, icone, valor, subtitulo, classe):
    return dedent(f"""
    <div class="analytics-metric-card {classe}">
        <div class="analytics-metric-head">
            <span>{titulo}</span>
            <i class="bi {icone}"></i>
        </div>
        <div class="analytics-metric-value">{valor}</div>
        <div class="analytics-metric-sub">{subtitulo}</div>
    </div>
    """)


def status_card_html(situacao, qtd, pct, classe):
    return dedent(f"""
    <div class="status-card {classe}">
        <div>
            <span class="status-badge">{situacao}</span>
            <span class="status-share">{pct:.1f}%</span>
        </div>
        <div class="status-value">{fmt_int_br(qtd)}</div>
        <div class="status-label">alunos</div>
    </div>
    """)


def pagina_analise(df, df_modelo):
    h1_dashboard()

    st.markdown(
        dedent("""
        <div class="section-title">
            <i class="bi bi-database-fill-check icon-inline"></i>Análise de Dados
        </div>
        """),
        unsafe_allow_html=True,
    )

    total_alunos = len(df)

    if "status_discente" in df.columns:
        status_series = df["status_discente"].astype(str).str.strip()
        status_evasao_ampla = {
            "CANCELADO", "TRANCADO", "DESISTENCIA", "DESISTÊNCIA", "EVADIDO", "ABANDONO"
        }
        evadidos = status_series.str.upper().isin(status_evasao_ampla).sum()
        taxa_evasao = (evadidos / total_alunos * 100) if total_alunos > 0 else 0
    else:
        status_series = pd.Series(dtype=str)
        evadidos = 0
        taxa_evasao = 0

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            metric_card_html(
                "Total de Alunos",
                "bi-people-fill",
                fmt_int_br(total_alunos),
                "Registros no dataset",
                "metric-blue",
            ),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            metric_card_html(
                "Alunos Evadidos",
                "bi-activity",
                fmt_int_br(evadidos),
                "Desistência, cancelamento e trancamento",
                "metric-red",
            ),
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            metric_card_html(
                "Taxa de Evasão",
                "bi-graph-up-arrow",
                f"{taxa_evasao:.1f}%",
                "Percentual de evasão total",
                "metric-yellow",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if "status_discente" in df.columns:
        dist = (
            status_series.value_counts(dropna=False)
            .rename_axis("situacao")
            .reset_index(name="qtd")
        )
        dist["pct"] = dist["qtd"] / dist["qtd"].sum() * 100

        st.markdown(
            dedent("""
            <div class="analytics-panel">
                <div class="analytics-panel-title">
                    <i class="bi bi-collection-fill"></i>
                    <span>Distribuição por Situação</span>
                </div>
                <div class="analytics-panel-subtitle">
                    Quantidade de alunos em cada situação de matrícula
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        cards = []
        for _, row in dist.iterrows():
            situacao = str(row["situacao"])
            qtd = int(row["qtd"])
            pct = float(row["pct"])
            classe = classe_status_card(situacao)
            cards.append((situacao, qtd, pct, classe))

        for i in range(0, len(cards), 3):
            cols = st.columns(3)
            bloco = cards[i:i + 3]

            for col, (situacao, qtd, pct, classe) in zip(cols, bloco):
                with col:
                    st.markdown(
                        status_card_html(situacao, qtd, pct, classe),
                        unsafe_allow_html=True,
                    )

    preview_cols_prioridade = [
        "id_discente",
        "status_discente",
        "media_geral",
        "faixa_renda_familiar",
        "ano_ingresso",
        "alvo_evasao",
    ]
    preview_cols = [c for c in preview_cols_prioridade if c in df.columns]

    preview_df = df.copy()
    if "alvo_evasao" in preview_df.columns:
        preview_df["alvo_evasao"] = preview_df["alvo_evasao"].map({1: "Sim", 0: "Não"})

    st.markdown(
        dedent("""
        <div class="analytics-panel">
            <div class="analytics-panel-title">
                <i class="bi bi-table"></i>
                <span>Preview do Dataset</span>
            </div>
            <div class="analytics-panel-subtitle">
                Primeiras 10 linhas da base de matrículas
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.caption(f"Total de registros: {fmt_int_br(len(df))}")

    if preview_cols:
        st.dataframe(
            preview_df[preview_cols].head(10),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(
            preview_df.head(10),
            use_container_width=True,
            hide_index=True,
        )

    footer()


# =========================================================
# CHARTS
# =========================================================
def pagina_graficos(df):
    st.markdown(
        dedent("""
        <div class="viz-page-title">Gráficos e Visualizações</div>
        <div class="viz-page-subtitle">Análise visual dos dados de evasão acadêmica</div>
        """),
        unsafe_allow_html=True,
    )

    if "status_discente" not in df.columns:
        st.warning("A coluna 'status_discente' não foi encontrada na base.")
        footer()
        return

    status_limpo = df["status_discente"].astype(str).str.strip()

    status_df = (
        status_limpo.value_counts()
        .rename_axis("Situação")
        .reset_index(name="Quantidade")
    )

    cores_barras = [
        "#27579B", "#37A0CF", "#3E8A3E", "#E7AE49",
        "#7F5AE6", "#E04499", "#1FB686", "#718096", "#A0AEC0",
    ]

    status_evasao = {"CANCELADO", "TRANCADO", "DESISTENCIA", "DESISTÊNCIA", "EVADIDO", "ABANDONO"}
    evadidos = int(status_limpo.str.upper().isin(status_evasao).sum())
    nao_evadidos = int(len(df) - evadidos)

    with st.container(border=True):
        st.markdown(
            dedent("""
            <div class="chart-card-title">
                <i class="bi bi-bar-chart-line-fill"></i>
                <span>Distribuição por Situação de Matrícula</span>
            </div>
            <div class="chart-card-subtitle">Quantidade de alunos em cada situação</div>
            """),
            unsafe_allow_html=True,
        )

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=status_df["Situação"],
                y=status_df["Quantidade"],
                name="Quantidade de Alunos",
                marker=dict(
                    color=cores_barras[:len(status_df)],
                    line=dict(color="#FFFFFF", width=1.2),
                ),
                hovertemplate="<b>%{x}</b><br>Quantidade: %{y}<extra></extra>",
            )
        )

        fig_bar.update_layout(
            height=350,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=25, r=20, t=10, b=70),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="top", y=-0.18,
                xanchor="center", x=0.5, title=None, font=dict(size=12),
            ),
            xaxis=dict(
                title=None, tickangle=-45,
                showgrid=True, gridcolor="#E9EEF5", zeroline=False,
                tickfont=dict(size=11, color="#6B7280"),
            ),
            yaxis=dict(
                title=None, showgrid=True, gridcolor="#E9EEF5",
                zeroline=False, rangemode="tozero",
                tickfont=dict(size=11, color="#6B7280"),
            ),
            font=dict(family="sans-serif", color="#334155"),
        )

        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    st.markdown("<div class='chart-gap-top'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown(
                dedent("""
                <div class="chart-card-title">
                    <i class="bi bi-pie-chart-fill"></i>
                    <span>Evasão vs Não Evasão</span>
                </div>
                <div class="chart-card-subtitle">Proporção de alunos evadidos</div>
                """),
                unsafe_allow_html=True,
            )

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Não Evadidos", "Evadidos"],
                        values=[nao_evadidos, evadidos],
                        sort=False,
                        textinfo="label+percent",
                        textposition="outside",
                        marker=dict(
                            colors=["#3E7F3C", "#F44343"],
                            line=dict(color="#FFFFFF", width=2),
                        ),
                        hovertemplate="<b>%{label}</b><br>Quantidade: %{value}<br>%{percent}<extra></extra>",
                    )
                ]
            )

            fig_pie.update_layout(
                height=320,
                paper_bgcolor="#FFFFFF",
                margin=dict(l=20, r=20, t=10, b=50),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="top", y=-0.08,
                    xanchor="center", x=0.5, title=None, font=dict(size=12),
                ),
                font=dict(family="sans-serif", color="#334155"),
            )

            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    with col2:
        with st.container(border=True):
            st.markdown(
                dedent("""
                <div class="chart-card-title">
                    <i class="bi bi-graph-up-arrow"></i>
                    <span>Evolução da Taxa de Evasão</span>
                </div>
                <div class="chart-card-subtitle">Taxa de evasão por ano de ingresso</div>
                """),
                unsafe_allow_html=True,
            )

            if "ano_ingresso" in df.columns:
                aux = df.copy()
                aux["status_discente"] = aux["status_discente"].astype(str).str.strip()
                aux["evasao_flag"] = aux["status_discente"].str.upper().isin(status_evasao).astype(int)

                evol_df = (
                    aux.groupby("ano_ingresso", as_index=False)["evasao_flag"]
                    .mean()
                    .sort_values("ano_ingresso")
                )
                evol_df["taxa_pct"] = evol_df["evasao_flag"] * 100
                ymax = max(80, float(evol_df["taxa_pct"].max()) * 1.15 if len(evol_df) else 80)

                fig_line = go.Figure()
                fig_line.add_trace(
                    go.Scatter(
                        x=evol_df["ano_ingresso"],
                        y=evol_df["taxa_pct"],
                        mode="lines+markers",
                        name="Taxa de Evasão (%)",
                        line=dict(color="#E7AE49", width=2.5),
                        marker=dict(color="#E7AE49", size=8, line=dict(color="#E7AE49", width=1)),
                        hovertemplate="<b>Ano %{x}</b><br>Taxa: %{y:.1f}%<extra></extra>",
                    )
                )

                fig_line.update_layout(
                    height=320,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    margin=dict(l=25, r=15, t=10, b=35),
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="top", y=-0.12,
                        xanchor="center", x=0.5, title=None, font=dict(size=12),
                    ),
                    xaxis=dict(
                        title=None, showgrid=True, gridcolor="#E9EEF5",
                        zeroline=False, tickfont=dict(size=11, color="#6B7280"),
                    ),
                    yaxis=dict(
                        title="Taxa (%)", showgrid=True, gridcolor="#E9EEF5", zeroline=False,
                        range=[0, ymax], tickfont=dict(size=11, color="#6B7280"),
                        title_font=dict(size=12, color="#6B7280"),
                    ),
                    font=dict(family="sans-serif", color="#334155"),
                )

                st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False, "responsive": True})
            else:
                st.info("A coluna 'ano_ingresso' não está disponível para montar a série temporal.")

    footer()


# =========================================================
# PREDICTION
# =========================================================
def pagina_previsao(df, df_modelo, resultados_modelo):
    h1_dashboard()

    st.markdown(
        """
        <div class="section-title">
            <i class="bi bi-robot icon-inline"></i>Previsão de Evasão
        </div>
        """,
        unsafe_allow_html=True,
    )

    if resultados_modelo is None:
        st.error("Não foi possível treinar o modelo com a base atual.")
        footer()
        return

    acc = format_pct(resultados_modelo["accuracy"])
    prec = format_pct(resultados_modelo["precision"])
    rec = format_pct(resultados_modelo["recall"])

    # =====================================================
    # BOX SUPERIOR - INFORMAÇÕES DO MODELO
    # =====================================================
    with st.container(border=True):
        st.markdown(
            """
            <div class="pred-model-title">
                <i class="bi bi-info-circle-fill"></i>
                <span>Informações do Modelo</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        t1, t2, t3, t4 = st.columns(4)

        with t1:
            st.markdown(
                """
                <div class="pred-top-metric">
                    <div class="pred-top-metric-label">Algoritmo</div>
                    <div class="pred-top-metric-value">Logistic Regression</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with t2:
            st.markdown(
                f"""
                <div class="pred-top-metric">
                    <div class="pred-top-metric-label">Acurácia</div>
                    <div class="pred-top-metric-value">{acc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with t3:
            st.markdown(
                f"""
                <div class="pred-top-metric">
                    <div class="pred-top-metric-label">Precisão</div>
                    <div class="pred-top-metric-value">{prec}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with t4:
            st.markdown(
                f"""
                <div class="pred-top-metric">
                    <div class="pred-top-metric-label">Recall</div>
                    <div class="pred-top-metric-value">{rec}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="pred-chip-wrap">
                <div class="pred-chip-title">Variáveis Utilizadas:</div>
                <span class="pred-chip">Média Geral</span>
                <span class="pred-chip">Faixa de Renda Familiar</span>
                <span class="pred-chip">Ano de Ingresso</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # =====================================================
    # VALORES PADRÃO
    # =====================================================
    opcoes_renda = (
        df["faixa_renda_familiar"].dropna().astype(str).sort_values().unique().tolist()
        if "faixa_renda_familiar" in df.columns
        else ["1 a 2 salários mínimos", "2 a 3 salários mínimos", "3 a 5 salários mínimos", "Acima de 5 salários mínimos"]
    )

    ano_min = int(pd.to_numeric(df["ano_ingresso"], errors="coerce").min()) if "ano_ingresso" in df.columns else 2019
    ano_max = int(pd.to_numeric(df["ano_ingresso"], errors="coerce").max()) if "ano_ingresso" in df.columns else 2025
    media_padrao = float(pd.to_numeric(df["media_geral"], errors="coerce").median()) if "media_geral" in df.columns else 7.0
    renda_padrao = opcoes_renda[0] if len(opcoes_renda) else "1 a 2 salários mínimos"
    ano_padrao = int(pd.to_numeric(df["ano_ingresso"], errors="coerce").median()) if "ano_ingresso" in df.columns else 2022

    if "pred_prob" not in st.session_state:
        entrada_inicial = montar_entrada_previsao(
            df=df,
            features=resultados_modelo["features"],
            media_geral=media_padrao,
            faixa_renda=renda_padrao,
            ano_ingresso=ano_padrao,
        )
        st.session_state.pred_prob = float(resultados_modelo["model"].predict_proba(entrada_inicial)[0, 1])
        st.session_state.pred_media = float(round(media_padrao, 1))
        st.session_state.pred_renda = renda_padrao
        st.session_state.pred_ano = int(ano_padrao)

    # =====================================================
    # DUAS COLUNAS
    # =====================================================
    col_form, col_result = st.columns(2)

    with col_form:
        with st.container(border=True):
            st.markdown(
                """
                <div class="pred-card-title">Dados do Estudante</div>
                <div class="pred-card-sub">Informe os dados para realizar a previsão</div>
                """,
                unsafe_allow_html=True,
            )

            with st.form("form_previsao_estilo", clear_on_submit=False):
                media_geral = st.slider(
                    "Média Geral",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(st.session_state.pred_media),
                    step=0.1,
                )

                faixa_renda = st.selectbox(
                    "Faixa de Renda Familiar",
                    opcoes_renda,
                    index=opcoes_renda.index(st.session_state.pred_renda) if st.session_state.pred_renda in opcoes_renda else 0,
                )

                anos_disponiveis = list(range(ano_min, ano_max + 1))
                ano_ingresso = st.selectbox(
                    "Ano de Ingresso",
                    anos_disponiveis,
                    index=anos_disponiveis.index(st.session_state.pred_ano) if st.session_state.pred_ano in anos_disponiveis else 0,
                )

                submitted = st.form_submit_button(
                    "🔍 Prever Risco de Evasão",
                    use_container_width=True,
                )

            if submitted:
                entrada_df = montar_entrada_previsao(
                    df=df,
                    features=resultados_modelo["features"],
                    media_geral=media_geral,
                    faixa_renda=faixa_renda,
                    ano_ingresso=ano_ingresso,
                )

                prob = float(resultados_modelo["model"].predict_proba(entrada_df)[0, 1])

                st.session_state.pred_prob = prob
                st.session_state.pred_media = float(media_geral)
                st.session_state.pred_renda = str(faixa_renda)
                st.session_state.pred_ano = int(ano_ingresso)

    with col_result:
        prob = float(st.session_state.pred_prob)
        classe_texto, classe_css, recomendacao = classificacao_risco(prob)

        ano_atual = pd.Timestamp.today().year
        tempo_instituicao = max(0, ano_atual - int(st.session_state.pred_ano))

        with st.container(border=True):
            st.markdown(
                """
                <div class="pred-card-title">Resultado da Previsão</div>
                <div class="pred-card-sub">Análise do risco de evasão</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="pred-prob-label">Probabilidade de Evasão</div>
                <div class="pred-prob-value">{prob * 100:.1f}%</div>
                <div class="pred-progress">
                    <div class="pred-progress-fill" style="width: {prob * 100:.1f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="pred-risk-box {classe_css}">
                    <div class="pred-risk-title">Classificação de Risco</div>
                    <div class="pred-risk-value">{classe_texto}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="pred-mini-box">
                    <div class="pred-mini-title">
                        <i class="bi bi-activity"></i> &nbsp;Recomendação:
                    </div>
                    <div class="pred-mini-text">{recomendacao}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="pred-mini-box pred-factors">
                    <div class="pred-mini-title">Fatores Analisados:</div>
                    <ul>
                        <li>Média Geral: {st.session_state.pred_media:.1f}</li>
                        <li>Renda Familiar: {st.session_state.pred_renda}</li>
                        <li>Ano de Ingresso: {st.session_state.pred_ano}</li>
                        <li>Tempo na Instituição: {tempo_instituicao} anos</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="pred-note">
            <b>Como interpretar:</b> O modelo calcula a probabilidade de evasão com base em padrões históricos.
            Probabilidades acima de 60% indicam alto risco e requerem atenção prioritária.
            Use estas previsões como ferramenta de apoio à decisão pedagógica, não como determinante absoluto.
        </div>
        """,
        unsafe_allow_html=True,
    )

    footer()


# =========================================================
# MAIN
# =========================================================
def main():
    aplicar_css()
    pagina = sidebar_controles()

    try:
        df = gerar_base_demo()
        df = normalizar_colunas(df)
        df, df_modelo, erro_alvo = construir_alvo_evasao(df)

        resultados_modelo = None
        if df_modelo is not None and erro_alvo is None:
            resultados_modelo, erro_modelo = treinar_modelo(df_modelo)
            if erro_modelo is not None:
                st.warning(erro_modelo)

        if erro_alvo is not None:
            st.warning(erro_alvo)

        if pagina == "Home":
            pagina_home(df, df_modelo, resultados_modelo)
        elif pagina == "Análise de Dados":
            pagina_analise(df, df_modelo)
        elif pagina == "Gráficos":
            pagina_graficos(df)
        elif pagina == "Previsão de Evasão":
            pagina_previsao(df, df_modelo, resultados_modelo)

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar a aplicação: {e}")


if __name__ == "__main__":
    main()
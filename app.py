# app.py
# -*- coding: utf-8 -*-

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import joblib
import pickle
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
    page_title="Dashboard Acadêmico de Reprovação - EBTT",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
RESULTADO_DIR = BASE_DIR / "estatistica_descritiva" / "resultado"
MODELOS_DIR = BASE_DIR / "modelos_pickle"


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

            .metric-green .analytics-metric-value,
            .metric-green .analytics-metric-head i {
                color: #2F855A !important;
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
                color: #7A8798 !important;
                margin-bottom: 0.9rem;
            }

            .pred-form-shell {
                background: linear-gradient(180deg, #F8FBFF 0%, #FFFFFF 100%);
                border: 1px solid #E3EDF7;
                border-radius: 18px;
                padding: 1rem 1rem 0.35rem 1rem;
                margin-top: 0.35rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
            }

            .pred-form-section-title {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 0.92rem;
                font-weight: 800;
                color: #24579B !important;
                margin: 0.2rem 0 0.75rem 0;
                padding-bottom: 0.45rem;
                border-bottom: 1px solid #E8EFF7;
            }

            .pred-form-help {
                font-size: 0.80rem;
                color: #7A8798 !important;
                margin: -0.15rem 0 0.75rem 0;
            }

            .pred-divider-space {
                height: 0.2rem;
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
            div[data-baseweb="input"] > div,
            div[data-baseweb="base-input"] > div {
                background: #FFFFFF !important;
                border-radius: 12px !important;
                border: 1px solid #D7E3F1 !important;
                box-shadow: 0 2px 8px rgba(25, 55, 99, 0.05) !important;
                min-height: 46px !important;
            }

            div[data-baseweb="select"] * ,
            div[data-baseweb="input"] * ,
            div[data-baseweb="base-input"] * ,
            div[data-baseweb="select"] span,
            div[data-baseweb="select"] div,
            div[data-baseweb="select"] input,
            div[data-baseweb="input"] input,
            div[data-baseweb="base-input"] input,
            div[data-baseweb="select"] svg,
            div[data-testid="stNumberInput"] input,
            div[data-testid="stNumberInput"] div,
            div[data-testid="stSelectbox"] div,
            div[data-testid="stSelectbox"] span {
                color: #173153 !important;
                fill: #173153 !important;
                -webkit-text-fill-color: #173153 !important;
                opacity: 1 !important;
            }

            div[data-baseweb="select"] input::placeholder,
            div[data-baseweb="input"] input::placeholder,
            div[data-baseweb="base-input"] input::placeholder {
                color: #6B7A90 !important;
                -webkit-text-fill-color: #6B7A90 !important;
                opacity: 1 !important;
            }

            div[data-testid="stNumberInput"] button {
                color: #173153 !important;
                background: #F6FAFF !important;
                border-color: #D7E3F1 !important;
            }

            div[data-testid="stCheckbox"] {
                background: #FFFFFF !important;
                border: 1px solid #DCE6F2 !important;
                border-radius: 12px !important;
                padding: 0.45rem 0.7rem !important;
                min-height: 50px !important;
                display: flex !important;
                align-items: center !important;
                box-shadow: 0 2px 8px rgba(25, 55, 99, 0.04) !important;
                margin-bottom: 0.35rem !important;
            }

            div[data-testid="stCheckbox"] label {
                margin-bottom: 0 !important;
                width: 100% !important;
            }

            /* =====================================================
               LEGENDAS / LABELS DOS CAMPOS
            ===================================================== */
            label,
            .stSelectbox label,
            .stNumberInput label,
            .stCheckbox label,
            .stTextInput label,
            .stSlider label,
            .stRadio label,
            .stMultiSelect label,
            div[data-testid="stWidgetLabel"],
            div[data-testid="stWidgetLabel"] p,
            div[data-testid="stWidgetLabel"] span,
            .stCheckbox span,
            .stCheckbox p {
                color: #173153 !important;
                opacity: 1 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DEMO DATA / MODELO
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




@st.cache_resource
def carregar_modelos_pickle():
    arquivos_modelo = {
        "Logit": MODELOS_DIR / "logit.pkl",
    }

    modelos = {}
    erros = {}

    nome = "Logit"
    caminho = arquivos_modelo[nome]

    if not caminho.exists():
        erros[nome] = f"Arquivo não encontrado: {caminho.name}"
        return modelos, erros, arquivos_modelo

    try:
        with open(caminho, "rb") as f:
            modelos[nome] = pickle.load(f)
    except Exception:
        try:
            modelos[nome] = joblib.load(caminho)
        except Exception as e:
            erros[nome] = f"Falha ao carregar {caminho.name}: {e}"

    return modelos, erros, arquivos_modelo


def extrair_features_modelo(modelo):
    candidatos = [
        modelo,
        getattr(modelo, "best_estimator_", None),
        getattr(getattr(modelo, "named_steps", {}), "get", lambda *_: None)("classifier"),
    ]

    for cand in candidatos:
        if cand is not None and hasattr(cand, "feature_names_in_"):
            return [str(x) for x in list(cand.feature_names_in_)]

    return None


def set_feature_value(X, feature_names, feature_name, value):
    idxs = [i for i, col in enumerate(feature_names) if col == feature_name]
    if idxs:
        X.iloc[0, idxs] = value


def montar_input_pkl(
    feature_names,
    sexo,
    estado_civil,
    raca,
    dummy_tecnico,
    tem_laboratorio,
    tem_ead,
    faixa_renda,
    ano_periodo,
    reprov_ate_entao,
    reprov_disc_ate_entao,
    curso_id,
    area_conhecimento,
):
    X = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names, dtype=float)

    # Binárias/dummies diretas
    set_feature_value(X, feature_names, "sexo_masculino", 1.0 if sexo == "Masculino" else 0.0)
    set_feature_value(X, feature_names, "estado_civil_outro", 1.0 if estado_civil == "Outro" else 0.0)
    set_feature_value(X, feature_names, "dummy_tecnico", 1.0 if dummy_tecnico else 0.0)
    set_feature_value(X, feature_names, "tem_laboratorio", 1.0 if tem_laboratorio else 0.0)
    set_feature_value(X, feature_names, "tem_ead", 1.0 if tem_ead else 0.0)

    # Numéricas
    try:
        set_feature_value(X, feature_names, "ano_periodo", float(ano_periodo))
    except Exception:
        set_feature_value(X, feature_names, "ano_periodo", 0.0)

    set_feature_value(X, feature_names, "reprov_ate_entao", float(reprov_ate_entao))
    set_feature_value(X, feature_names, "reprov_disc_ate_entao", float(reprov_disc_ate_entao))

    # Raça
    mapa_raca = {
        "Categoria de referência": None,
        "Nan": "raca_declarada_nan",
        "Não informado": "raca_declarada_nao_informado",
        "Negra": "raca_declarada_negra",
        "Outra": "raca_declarada_outra",
    }
    feat_raca = mapa_raca.get(raca)
    if feat_raca:
        set_feature_value(X, feature_names, feat_raca, 1.0)

    # Faixa de renda
    mapa_renda = {
        "Categoria de referência": None,
        "Até 1k": "faixa_renda_familiar_ate_1k",
        "2k a 4k": "faixa_renda_familiar_2k_4k",
        "4k a 8k": "faixa_renda_familiar_4k_8k",
        "8k ou mais": "faixa_renda_familiar_8k_mais",
        "Nan": "faixa_renda_familiar_nan",
    }
    feat_renda = mapa_renda.get(faixa_renda)
    if feat_renda:
        set_feature_value(X, feature_names, feat_renda, 1.0)

    # Curso
    if curso_id is not None:
        set_feature_value(X, feature_names, f"id_curso_cursos_{curso_id}", 1.0)

    # Área do conhecimento
    if area_conhecimento is not None:
        set_feature_value(X, feature_names, f"area_conhecimento_{area_conhecimento}", 1.0)

    # Segurança extra
    X = X.fillna(0.0)
    return X


def prever_probabilidade_pkl(modelo, X):
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0])

    if hasattr(modelo, "decision_function"):
        score = float(modelo.decision_function(X)[0])
        return float(1 / (1 + np.exp(-score)))

    pred = float(modelo.predict(X)[0])
    return max(0.0, min(1.0, pred))


def format_pct(x):
    return f"{100 * x:.1f}%"


def fmt_int_br(x):
    return f"{int(x):,}".replace(",", ".")


def fmt_float_br(x, casas=2):
    try:
        return f"{float(x):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return x


def h1_dashboard():
    st.markdown(
        """
        <div class="main-title">Reprovação Acadêmica do EBTT</div>
        <div class="sub-title">Predição de reprovação por modelos de aprendizado de máquina</div>
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
        return "Baixo Risco", "pred-risk-low", "O estudante apresenta baixo risco de reprovação. O acompanhamento rotineiro é suficiente no momento."
    if prob < 0.60:
        return "Médio Risco", "pred-risk-mid", "Estudante em situação de atenção quanto à reprovação. Recomenda-se acompanhamento pedagógico e monitoramento do desempenho."
    return "Alto Risco", "pred-risk-high", "Estudante com alto risco de reprovação. Recomenda-se intervenção prioritária e acompanhamento ativo da equipe pedagógica."


# =========================================================
# LEITURA DOS CSVs DA ANÁLISE
# =========================================================
@st.cache_data
def ler_csv_resultado(nome_arquivo):
    caminho = RESULTADO_DIR / nome_arquivo
    if not caminho.exists():
        return None

    df = pd.read_csv(caminho)
    df = normalizar_colunas(df)

    primeira_col = df.columns[0]
    if str(primeira_col).lower().startswith("unnamed") or str(primeira_col).strip() == "":
        df = df.rename(columns={primeira_col: "indicador"})

    return df


@st.cache_data
def carregar_resultados_analise():
    return {
        "comparacao": ler_csv_resultado("comparacao_aprov_reprov.csv"),
        "ranking_cursos": ler_csv_resultado("ranking_cursos.csv"),
        "ranking_disciplinas": ler_csv_resultado("ranking_disciplinas.csv"),
        "serie_temporal": ler_csv_resultado("serie_temporal.csv"),
    }


def preparar_serie_temporal(df):
    if df is None or df.empty:
        return None

    df = df.copy()
    if "ano_periodo" not in df.columns or "reprovado" not in df.columns:
        return None

    df["ano_periodo"] = df["ano_periodo"].astype(str).str.strip()
    partes = df["ano_periodo"].str.split(".", n=1, expand=True)

    df["ano"] = pd.to_numeric(partes[0], errors="coerce")
    df["periodo"] = pd.to_numeric(partes[1], errors="coerce")
    df["reprovado"] = pd.to_numeric(df["reprovado"], errors="coerce")

    df = df.dropna(subset=["ano", "periodo", "reprovado"]).copy()
    df = df[(df["ano"] > 2015) | ((df["ano"] == 2015) & (df["periodo"] >= 1))].copy()

    if df.empty:
        return None

    df = df.sort_values(["ano", "periodo"]).reset_index(drop=True)

    df["reprovado"] = df["reprovado"].clip(lower=0, upper=1)
    df["aprovado"] = 1 - df["reprovado"]
    df["reprovado_pct"] = df["reprovado"] * 100
    df["aprovado_pct"] = df["aprovado"] * 100

    return df


def extrair_valor_comparacao(df_comp, indicador, coluna):
    if df_comp is None:
        return np.nan

    if "indicador" not in df_comp.columns:
        primeira = df_comp.columns[0]
        df_comp = df_comp.rename(columns={primeira: "indicador"})

    alvo = df_comp["indicador"].astype(str).str.strip().str.lower() == str(indicador).strip().lower()
    if alvo.sum() == 0 or coluna not in df_comp.columns:
        return np.nan

    return pd.to_numeric(df_comp.loc[alvo, coluna], errors="coerce").iloc[0]


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
            options=["Análise de Dados", "Previsão de Reprovação", "Home"],
            icons=["bar-chart-fill", "robot", "house-door-fill"],
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
                Reprovação Acadêmica EBTT © 2026
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
                <b>Ensino Básico, Técnico e Tecnológico (EBTT)</b> no apoio à predição de reprovação de estudantes.
            </p>
            <p>
                A partir de dados institucionais e do uso de técnicas de <b>aprendizado de máquina</b>, o dashboard busca apoiar a identificação de estudantes com maior probabilidade de reprovação, contribuindo para o monitoramento acadêmico e para ações de apoio à gestão.
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
                <p class="muted-note">Explore os resultados estatísticos sobre aprovação e reprovação</p>
                <ul>
                    <li>Comparação entre aprovados e reprovados</li>
                    <li>Ranking de cursos</li>
                    <li>Ranking de disciplinas</li>
                    <li>Série temporal da reprovação</li>
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
                    <li>Comparação aprovação vs reprovação</li>
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
            <p><b>Predição inteligente do risco de reprovação</b></p>
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
                e realizar previsões do risco de reprovação para estudantes específicos.
            </p>
            <p>
                <b>Dica:</b> A aba <b>Análise de Dados</b> lê diretamente os CSVs processados
                em <b>estatistica_descritiva/resultado</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    footer()


# =========================================================
# ANÁLISE DE DADOS
# =========================================================
def pagina_analise(df, df_modelo):
    h1_dashboard()

    st.markdown(
        """
        <div class="section-title">
            <i class="bi bi-database-fill-check icon-inline"></i>Análise de Dados
        </div>
        """,
        unsafe_allow_html=True,
    )

    resultados = carregar_resultados_analise()
    comp = resultados["comparacao"]
    ranking_cursos = resultados["ranking_cursos"]
    ranking_disciplinas = resultados["ranking_disciplinas"]
    serie_temporal = preparar_serie_temporal(resultados["serie_temporal"])

    if all(x is None for x in [comp, ranking_cursos, ranking_disciplinas, serie_temporal]):
        st.error(f"Nenhum CSV foi encontrado em: {RESULTADO_DIR}")
        footer()
        return

    media_aprov = extrair_valor_comparacao(comp, "media_geral", "Aprovados")
    media_reprov = extrair_valor_comparacao(comp, "media_geral", "Reprovados")
    ch_pendente_diff = extrair_valor_comparacao(comp, "ch_pendente", "Diferença")

    total_cursos = len(ranking_cursos) if ranking_cursos is not None else 0
    total_disciplinas = len(ranking_disciplinas) if ranking_disciplinas is not None else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            metric_card_html(
                "Média Geral Aprovados",
                "bi-mortarboard-fill",
                fmt_float_br(media_aprov, 2) if pd.notna(media_aprov) else "-",
                "",
                "metric-blue",
            ),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            metric_card_html(
                "Média Geral Reprovados",
                "bi-exclamation-triangle-fill",
                fmt_float_br(media_reprov, 2) if pd.notna(media_reprov) else "-",
                "",
                "metric-red",
            ),
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            metric_card_html(
                "Diferença CH Pendente",
                "bi-hourglass-split",
                fmt_float_br(ch_pendente_diff, 2) if pd.notna(ch_pendente_diff) else "-",
                "Aprovados vs reprovados",
                "metric-yellow",
            ),
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            metric_card_html(
                "Cursos no Ranking",
                "bi-journal-richtext",
                fmt_int_br(total_cursos),
                f"Disciplinas: {fmt_int_br(total_disciplinas)}",
                "metric-green",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    cursos = None
    disciplinas = None

    col_chart_curso, col_chart_disc = st.columns(2)

    with col_chart_curso:
        if ranking_cursos is not None and not ranking_cursos.empty:
            st.markdown(
                """
                <div class="analytics-panel">
                    <div class="analytics-panel-title">
                        <i class="bi bi-bar-chart-fill"></i>
                        <span>Top Cursos por Taxa de Reprovação</span>
                    </div>
                    <div class="analytics-panel-subtitle">
                        Ranking dos cursos com maior taxa de reprovação
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cursos = ranking_cursos.copy()
            if "taxa_reprov" in cursos.columns:
                cursos["taxa_reprov"] = pd.to_numeric(cursos["taxa_reprov"], errors="coerce")
            if "n" in cursos.columns:
                cursos["n"] = pd.to_numeric(cursos["n"], errors="coerce")

            cursos = cursos.sort_values("taxa_reprov", ascending=False).head(10).copy()
            cursos["curso_nome_curto"] = cursos["curso_nome"].astype(str).apply(
                lambda x: x if len(x) <= 55 else x[:55] + "..."
            )

            cursos_plot = cursos.sort_values("taxa_reprov", ascending=True).copy()

            fig_cursos = px.bar(
                cursos_plot,
                x="taxa_reprov",
                y="curso_nome_curto",
                orientation="h",
                text="taxa_reprov",
            )
            fig_cursos.update_traces(
                texttemplate="%{text:.1%}",
                hovertemplate="<b>%{customdata[0]}</b><br>Taxa de reprovação: %{x:.1%}<extra></extra>",
                customdata=cursos_plot[["curso_nome"]],
            )
            fig_cursos.update_layout(
                height=450,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Taxa de Reprovação",
                yaxis_title="",
                showlegend=False,
                yaxis=dict(tickfont=dict(color="#000000")),
                xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
            )
            st.plotly_chart(fig_cursos, use_container_width=True, config={"displayModeBar": False})

    with col_chart_disc:
        if ranking_disciplinas is not None and not ranking_disciplinas.empty:
            st.markdown(
                """
                <div class="analytics-panel">
                    <div class="analytics-panel-title">
                        <i class="bi bi-bar-chart-steps"></i>
                        <span>Top Disciplinas por Taxa de Reprovação</span>
                    </div>
                    <div class="analytics-panel-subtitle">
                        Ranking das disciplinas com maior taxa de reprovação
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            disciplinas = ranking_disciplinas.copy()
            if "taxa_reprov" in disciplinas.columns:
                disciplinas["taxa_reprov"] = pd.to_numeric(disciplinas["taxa_reprov"], errors="coerce")
            if "n" in disciplinas.columns:
                disciplinas["n"] = pd.to_numeric(disciplinas["n"], errors="coerce")

            disciplinas = disciplinas.sort_values("taxa_reprov", ascending=False).head(10).copy()

            nome_col_disc = None
            for cand in ["nome_componente_curricular", "nome_componete_curricular"]:
                if cand in disciplinas.columns:
                    nome_col_disc = cand
                    break

            if nome_col_disc is None:
                st.warning("A coluna com o nome da disciplina não foi encontrada em ranking_disciplinas.csv.")
            else:
                disciplinas["nome_disciplina_curto"] = disciplinas[nome_col_disc].astype(str).apply(
                    lambda x: x if len(x) <= 50 else x[:50] + "..."
                )

                disciplinas_plot = disciplinas.sort_values("taxa_reprov", ascending=True).copy()

                fig_disciplinas = px.bar(
                    disciplinas_plot,
                    x="taxa_reprov",
                    y="nome_disciplina_curto",
                    orientation="h",
                    text="taxa_reprov",
                )
                fig_disciplinas.update_traces(
                    texttemplate="%{text:.1%}",
                    hovertemplate="<b>%{customdata[0]}</b><br>Taxa de reprovação: %{x:.1%}<extra></extra>",
                    customdata=disciplinas_plot[[nome_col_disc]],
                )
                fig_disciplinas.update_layout(
                    height=450,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Taxa de Reprovação",
                    yaxis_title="",
                    showlegend=False,
                    yaxis=dict(tickfont=dict(color="#000000")),
                    xaxis=dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
                )
                st.plotly_chart(fig_disciplinas, use_container_width=True, config={"displayModeBar": False})

    if serie_temporal is not None and not serie_temporal.empty:
        col_serie, col_pizza = st.columns(2)

        with col_serie:
            st.markdown(
                """
                <div class="analytics-panel">
                    <div class="analytics-panel-title">
                        <i class="bi bi-graph-up-arrow"></i>
                        <span>Série Temporal da Reprovação</span>
                    </div>
                    <div class="analytics-panel-subtitle">
                        Evolução da taxa de reprovação ao longo dos períodos
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_serie = go.Figure()
            fig_serie.add_trace(
                go.Scatter(
                    x=serie_temporal["ano_periodo"],
                    y=serie_temporal["reprovado_pct"],
                    mode="lines+markers",
                    name="Reprovação (%)",
                    line=dict(color="#60A5FA", width=2),
                    marker=dict(color="#60A5FA", size=7),
                    hovertemplate="<b>%{x}</b><br>Reprovação: %{y:.2f}%<extra></extra>",
                )
            )
            fig_serie.update_layout(
                height=360,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=25, r=20, t=10, b=40),
                xaxis_title="Ano/Período",
                yaxis_title="Taxa de Reprovação (%)",
                showlegend=False,
                font=dict(color="#000000"),
                xaxis=dict(
                    title_font=dict(color="#000000"),
                    tickfont=dict(color="#000000"),
                    gridcolor="rgba(0,0,0,0.18)",
                    zerolinecolor="rgba(0,0,0,0.18)",
                ),
                yaxis=dict(
                    title_font=dict(color="#000000"),
                    tickfont=dict(color="#000000"),
                    gridcolor="rgba(0,0,0,0.18)",
                    zerolinecolor="rgba(0,0,0,0.18)",
                ),
            )
            st.plotly_chart(fig_serie, use_container_width=True, config={"displayModeBar": False})

        with col_pizza:
            st.markdown(
                """
                <div class="analytics-panel">
                    <div class="analytics-panel-title">
                        <i class="bi bi-pie-chart-fill"></i>
                        <span>Taxa de Aprovação vs Reprovação</span>
                    </div>
                    <div class="analytics-panel-subtitle">
                        Distribuição no período mais recente da série temporal
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ultimo = serie_temporal.iloc[-1]
            taxa_reprov = float(ultimo["reprovado"])
            taxa_aprov = float(ultimo["aprovado"])

            fig_pizza = go.Figure(
                data=[
                    go.Pie(
                        labels=["Aprovação", "Reprovação"],
                        values=[taxa_aprov, taxa_reprov],
                        textinfo="label+percent",
                        textfont=dict(color="#000000", size=16),
                        hovertemplate="<b>%{label}</b><br>Taxa: %{percent}<extra></extra>",
                    )
                ]
            )
            fig_pizza.update_layout(
                height=360,
                paper_bgcolor="#FFFFFF",
                margin=dict(l=20, r=20, t=10, b=10),
                showlegend=True,
                font=dict(color="#000000"),
                legend=dict(font=dict(color="#000000")),
            )
            st.plotly_chart(fig_pizza, use_container_width=True, config={"displayModeBar": False})

    footer()


# =========================================================
# GRÁFICOS
# =========================================================
def pagina_graficos(df):
    st.markdown(
        """
        <div class="viz-page-title">Gráficos e Visualizações</div>
        <div class="viz-page-subtitle">Análise visual dos dados de reprovação acadêmica</div>
        """,
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

    status_evasao = {"CANCELADO", "TRANCADO", "DESISTENCIA", "DESISTÊNCIA", "EVADIDO", "ABANDONO"}
    evadidos = int(status_limpo.str.upper().isin(status_evasao).sum())
    nao_evadidos = int(len(df) - evadidos)

    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-card-title">
                <i class="bi bi-bar-chart-line-fill"></i>
                <span>Distribuição por Situação de Matrícula</span>
            </div>
            <div class="chart-card-subtitle">Quantidade de alunos em cada situação</div>
            """,
            unsafe_allow_html=True,
        )

        fig_bar = px.bar(status_df, x="Situação", y="Quantidade", text="Quantidade")
        fig_bar.update_layout(
            height=350,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=25, r=20, t=10, b=70),
            xaxis_title="",
            yaxis_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='chart-gap-top'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-card-title">
                    <i class="bi bi-pie-chart-fill"></i>
                    <span>Reprovação vs Aprovação</span>
                </div>
                <div class="chart-card-subtitle">Proporção entre reprovação e aprovação</div>
                """,
                unsafe_allow_html=True,
            )

            total_alunos = nao_evadidos + evadidos
            taxa_reprovacao = (evadidos / total_alunos) if total_alunos > 0 else 0.0

            fig_pie = go.Figure()

            fig_pie.add_trace(
                go.Pie(
                    values=[taxa_reprovacao, 1 - taxa_reprovacao],
                    labels=["Reprovados", "Demais"],
                    hole=0.78,
                    sort=False,
                    direction="clockwise",
                    textinfo="none",
                    hoverinfo="skip",
                    marker=dict(
                        colors=["#C87412", "#E5E7EB"],
                        line=dict(color="#FFFFFF", width=2),
                    ),
                    showlegend=False,
                )
            )

            fig_pie.add_annotation(
                text=f"<b>{taxa_reprovacao * 100:.1f}%</b><br><span style='font-size:16px;'>Reprovação</span>",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=28, color="#000000"),
                align="center",
            )

            fig_pie.update_layout(
                height=320,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )

            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with col2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-card-title">
                    <i class="bi bi-graph-up-arrow"></i>
                    <span>Evolução da Taxa de Reprovação</span>
                </div>
                <div class="chart-card-subtitle">Taxa de reprovação por ano de ingresso</div>
                """,
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

                fig_line = px.line(evol_df, x="ano_ingresso", y="taxa_pct", markers=True)
                fig_line.update_layout(
                    height=320,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    xaxis_title="",
                    yaxis_title="Taxa (%)",
                )
                st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("A coluna 'ano_ingresso' não está disponível para montar a série temporal.")

    footer()


# =========================================================
# PREVISÃO
# =========================================================
def pagina_previsao(df, df_modelo, resultados_modelo):
    h1_dashboard()

    st.markdown(
        """
        <div class="section-title">
            <i class="bi bi-robot icon-inline"></i>Previsão de Reprovação
        </div>
        """,
        unsafe_allow_html=True,
    )

    modelos_pickle, erros_modelo, arquivos_modelo = carregar_modelos_pickle()

    modelo_escolhido = "Logit"

    if modelo_escolhido not in modelos_pickle:
        st.error("Não foi possível carregar o modelo Logit da pasta modelos_pickle.")
        if erros_modelo:
            for nome, erro in erros_modelo.items():
                st.warning(f"{nome}: {erro}")
        footer()
        return

    modelo = modelos_pickle[modelo_escolhido]
    nome_arquivo_modelo = arquivos_modelo[modelo_escolhido].name
    feature_names = extrair_features_modelo(modelo)

    if feature_names is None:
        st.error(f"O modelo {modelo_escolhido} não expõe feature_names_in_.")
        footer()
        return

    # Derivação de opções a partir das features esperadas
    curso_ids = []
    for feat in feature_names:
        if feat.startswith("id_curso_cursos_"):
            cid = feat.replace("id_curso_cursos_", "")
            if cid not in curso_ids:
                curso_ids.append(cid)

    area_opts = []
    for feat in feature_names:
        if feat.startswith("area_conhecimento_"):
            area = feat.replace("area_conhecimento_", "")
            if area not in area_opts:
                area_opts.append(area)

    # Períodos vindos da série temporal, quando existir
    resultados = carregar_resultados_analise()
    serie_temporal = preparar_serie_temporal(resultados["serie_temporal"])
    periodo_labels = []
    if serie_temporal is not None and not serie_temporal.empty and "ano_periodo" in serie_temporal.columns:
        periodo_labels = serie_temporal["ano_periodo"].astype(str).dropna().unique().tolist()

    if not periodo_labels:
        periodo_labels = ["2024.1", "2024.2"]

    def periodo_para_float(label):
        try:
            return float(str(label).replace(",", "."))
        except Exception:
            return 0.0

    st.markdown(
        """
        <div class="pred-chip-wrap">
            <div class="pred-chip-title">Variáveis do formulário usadas no vetor de previsão:</div>
            <span class="pred-chip">Sexo</span>
            <span class="pred-chip">Estado civil</span>
            <span class="pred-chip">Raça</span>
            <span class="pred-chip">Curso técnico</span>
            <span class="pred-chip">Laboratório</span>
            <span class="pred-chip">EAD</span>
            <span class="pred-chip">Renda</span>
            <span class="pred-chip">Ano/Período</span>
            <span class="pred-chip">Reprovações acumuladas</span>
            <span class="pred-chip">Curso</span>
            <span class="pred-chip">Área</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Defaults
    default_curso = curso_ids[0] if curso_ids else None
    default_area = area_opts[0] if area_opts else None
    default_periodo = periodo_labels[-1] if periodo_labels else "2024.2"

    if st.session_state.get("pred_model_name") != modelo_escolhido:
        st.session_state.pred_model_name = modelo_escolhido
        st.session_state.pred_prob = 0.0
        st.session_state.pred_form = {
            "sexo": "Feminino",
            "estado_civil": "Solteiro(a)",
            "raca": "Categoria de referência",
            "dummy_tecnico": False,
            "tem_laboratorio": False,
            "tem_ead": False,
            "faixa_renda": "Categoria de referência",
            "ano_periodo": default_periodo,
            "reprov_ate_entao": 0,
            "reprov_disc_ate_entao": 0,
            "curso_id": default_curso,
            "area_conhecimento": default_area,
        }

        X_default = montar_input_pkl(
            feature_names=feature_names,
            sexo=st.session_state.pred_form["sexo"],
            estado_civil=st.session_state.pred_form["estado_civil"],
            raca=st.session_state.pred_form["raca"],
            dummy_tecnico=st.session_state.pred_form["dummy_tecnico"],
            tem_laboratorio=st.session_state.pred_form["tem_laboratorio"],
            tem_ead=st.session_state.pred_form["tem_ead"],
            faixa_renda=st.session_state.pred_form["faixa_renda"],
            ano_periodo=periodo_para_float(st.session_state.pred_form["ano_periodo"]),
            reprov_ate_entao=st.session_state.pred_form["reprov_ate_entao"],
            reprov_disc_ate_entao=st.session_state.pred_form["reprov_disc_ate_entao"],
            curso_id=st.session_state.pred_form["curso_id"],
            area_conhecimento=st.session_state.pred_form["area_conhecimento"],
        )
        st.session_state.pred_prob = prever_probabilidade_pkl(modelo, X_default)

    col_form, col_result = st.columns(2)

    with col_form:
        with st.container(border=True):
            st.markdown(
                """
                <div class="pred-card-title">Dados do Estudante</div>
                <div class="pred-card-sub">Preencha as variáveis compatíveis com o modelo .pkl</div>
                """,
                unsafe_allow_html=True,
            )
            with st.form("form_previsao_pkl", clear_on_submit=False):
                st.markdown(
                    """
                    <div class="pred-form-shell">
                        <div class="pred-form-section-title">
                            <i class="bi bi-person-vcard-fill"></i>
                            <span>Perfil e contexto do estudante</span>
                        </div>
                        <div class="pred-form-help">Preencha os campos principais para gerar a estimativa de risco.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)
                with c1:
                    sexo = st.selectbox(
                        "Sexo",
                        ["Feminino", "Masculino"],
                        index=["Feminino", "Masculino"].index(st.session_state.pred_form["sexo"]),
                    )
                    estado_civil = st.selectbox(
                        "Estado civil",
                        ["Solteiro(a)", "Casado(a)", "Outro"],
                        index=["Solteiro(a)", "Casado(a)", "Outro"].index(st.session_state.pred_form["estado_civil"]),
                    )
                    raca = st.selectbox(
                        "Raça declarada",
                        ["Categoria de referência", "Nan", "Não informado", "Negra", "Outra"],
                        index=["Categoria de referência", "Nan", "Não informado", "Negra", "Outra"].index(st.session_state.pred_form["raca"]),
                    )
                    faixa_renda = st.selectbox(
                        "Faixa de renda familiar",
                        ["Categoria de referência", "Até 1k", "2k a 4k", "4k a 8k", "8k ou mais", "Nan"],
                        index=["Categoria de referência", "Até 1k", "2k a 4k", "4k a 8k", "8k ou mais", "Nan"].index(st.session_state.pred_form["faixa_renda"]),
                    )

                with c2:
                    ano_periodo = st.selectbox(
                        "Ano/Período",
                        periodo_labels,
                        index=periodo_labels.index(st.session_state.pred_form["ano_periodo"]) if st.session_state.pred_form["ano_periodo"] in periodo_labels else 0,
                    )
                    n1, n2 = st.columns(2)
                    with n1:
                        reprov_ate_entao = st.number_input(
                            "Reprovações até então",
                            min_value=0,
                            step=1,
                            value=int(st.session_state.pred_form["reprov_ate_entao"]),
                        )
                    with n2:
                        reprov_disc_ate_entao = st.number_input(
                            "Na disciplina",
                            min_value=0,
                            step=1,
                            value=int(st.session_state.pred_form["reprov_disc_ate_entao"]),
                        )

                    st.markdown(
                        """
                        <div class="pred-form-section-title" style="margin-top:0.35rem;">
                            <i class="bi bi-toggles2"></i>
                            <span>Características acadêmicas</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        dummy_tecnico = st.checkbox("Curso técnico", value=st.session_state.pred_form["dummy_tecnico"])
                    with b2:
                        tem_laboratorio = st.checkbox("Laboratório", value=st.session_state.pred_form["tem_laboratorio"])
                    with b3:
                        tem_ead = st.checkbox("EAD", value=st.session_state.pred_form["tem_ead"])

                st.markdown(
                    """
                    <div class="pred-form-section-title" style="margin-top:0.7rem;">
                        <i class="bi bi-book-half"></i>
                        <span>Curso e área do conhecimento</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                k1, k2 = st.columns(2)
                with k1:
                    curso_id = st.selectbox(
                        "ID do curso",
                        curso_ids if curso_ids else ["nan"],
                        index=(curso_ids.index(st.session_state.pred_form["curso_id"]) if st.session_state.pred_form["curso_id"] in curso_ids else 0) if curso_ids else 0,
                    )
                with k2:
                    area_conhecimento = st.selectbox(
                        "Área do conhecimento",
                        area_opts if area_opts else ["educacao"],
                        index=(area_opts.index(st.session_state.pred_form["area_conhecimento"]) if st.session_state.pred_form["area_conhecimento"] in area_opts else 0) if area_opts else 0,
                        format_func=lambda x: str(x).replace("_", " ").title(),
                    )

                submitted = st.form_submit_button("🔍 Prever Risco de Reprovação", use_container_width=True)

            if submitted:
                st.session_state.pred_form = {
                    "sexo": sexo,
                    "estado_civil": estado_civil,
                    "raca": raca,
                    "dummy_tecnico": dummy_tecnico,
                    "tem_laboratorio": tem_laboratorio,
                    "tem_ead": tem_ead,
                    "faixa_renda": faixa_renda,
                    "ano_periodo": ano_periodo,
                    "reprov_ate_entao": int(reprov_ate_entao),
                    "reprov_disc_ate_entao": int(reprov_disc_ate_entao),
                    "curso_id": curso_id,
                    "area_conhecimento": area_conhecimento,
                }

                X_pred = montar_input_pkl(
                    feature_names=feature_names,
                    sexo=sexo,
                    estado_civil=estado_civil,
                    raca=raca,
                    dummy_tecnico=dummy_tecnico,
                    tem_laboratorio=tem_laboratorio,
                    tem_ead=tem_ead,
                    faixa_renda=faixa_renda,
                    ano_periodo=periodo_para_float(ano_periodo),
                    reprov_ate_entao=reprov_ate_entao,
                    reprov_disc_ate_entao=reprov_disc_ate_entao,
                    curso_id=curso_id,
                    area_conhecimento=area_conhecimento,
                )

                st.session_state.pred_prob = prever_probabilidade_pkl(modelo, X_pred)

    with col_result:
        prob = float(st.session_state.pred_prob)
        classe_texto, classe_css, recomendacao = classificacao_risco(prob)

        with st.container(border=True):
            st.markdown(
                """
                <div class="pred-card-title">Resultado da Previsão</div>
                <div class="pred-card-sub">Análise do risco de reprovação</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="pred-prob-label">Probabilidade de Reprovação</div>
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
                        <i class="bi bi-cpu-fill"></i> &nbsp;Modelo utilizado:
                    </div>
                    <div class="pred-mini-text">{modelo_escolhido} ({nome_arquivo_modelo})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            form = st.session_state.pred_form

            st.markdown(
                f"""
                <div class="pred-mini-box pred-factors">
                    <div class="pred-mini-title">Entradas utilizadas:</div>
                    <ul>
                        <li>Sexo: {form["sexo"]}</li>
                        <li>Estado civil: {form["estado_civil"]}</li>
                        <li>Raça: {form["raca"]}</li>
                        <li>Curso técnico: {"Sim" if form["dummy_tecnico"] else "Não"}</li>
                        <li>Tem laboratório: {"Sim" if form["tem_laboratorio"] else "Não"}</li>
                        <li>Tem EAD: {"Sim" if form["tem_ead"] else "Não"}</li>
                        <li>Faixa de renda: {form["faixa_renda"]}</li>
                        <li>Ano/Período: {form["ano_periodo"]}</li>
                        <li>Reprovações até então: {form["reprov_ate_entao"]}</li>
                        <li>Reprovações na disciplina até então: {form["reprov_disc_ate_entao"]}</li>
                        <li>ID do curso: {form["curso_id"]}</li>
                        <li>Área: {str(form["area_conhecimento"]).replace("_", " ").title()}</li>
                    </ul>
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
        """
        <div class="pred-note">
            <b>Como interpretar:</b> O formulário monta um vetor no formato exato esperado pelo arquivo .pkl.
            A previsão exibida corresponde à probabilidade de reprovação calculada pelo modelo Logit.
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
        elif pagina == "Previsão de Reprovação":
            pagina_previsao(df, df_modelo, resultados_modelo)

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar a aplicação: {e}")


if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd

st.set_page_config(page_title="Evasão Acadêmica do EBTT", layout="wide")

st.title("📊 Evasão Acadêmica do EBTT")

st.markdown("Aplicação simples para visualização de evasão acadêmica.")

# Upload de dados
uploaded_file = st.file_uploader("Envie um CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Visualização dos dados")
    st.dataframe(df)

    if "situacao" in df.columns:
        evasao = df[df["situacao"].isin(["DESISTENCIA", "CANCELADO", "TRANCADO", "TRANSFERIDO"])]
        taxa = len(evasao) / len(df)

        st.metric("Taxa de evasão", f"{taxa:.2%}")

        st.subheader("Distribuição da situação")
        st.bar_chart(df["situacao"].value_counts())

else:
    st.info("Envie um arquivo CSV para começar.")

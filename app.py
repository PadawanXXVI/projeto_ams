import streamlit as st
import pandas as pd
import pickle

# Carregar o modelo salvo
with open("modelo_evasao.pickle", "rb") as file:
    modelo = pickle.load(file)

st.set_page_config(page_title="Preditor de Evasão Escolar", layout="centered")
st.title("🎓 Preditor de Risco de Evasão Escolar")
st.markdown("""
Este aplicativo utiliza um modelo supervisionado para prever se um(a) estudante apresenta **alto risco de evasão escolar** com base em seus dados acadêmicos, comportamentais e familiares.
""")

st.sidebar.header("📋 Preencha os dados do aluno")

# Inputs via sidebar
gender = st.sidebar.radio("Gênero", ["Feminino", "Masculino"])
nationality = st.sidebar.selectbox("Nacionalidade (código interno)", list(range(14)))
stage = st.sidebar.selectbox("Etapa Escolar", ["LowerLevel", "MiddleSchool", "HighSchool"])
raisedhands = st.sidebar.slider("Levantou a mão (vezes)", 0, 100, 25)
resources = st.sidebar.slider("Acessou recursos (vezes)", 0, 100, 50)
announcements = st.sidebar.slider("Visualizou avisos (vezes)", 0, 100, 30)
discussion = st.sidebar.slider("Participou de discussões (vezes)", 0, 100, 10)
absence = st.sidebar.radio("Faltas", ["under-7", "above-7"])

# Dicionários de codificação
stage_map = {"LowerLevel": 0, "MiddleSchool": 1, "HighSchool": 2}
absence_map = {"under-7": 0, "above-7": 1}
gender_map = {"Feminino": 0, "Masculino": 1}

# Montar entrada compatível com o modelo
entrada = pd.DataFrame([{
    'gender': gender_map[gender],
    'NationalITy': nationality,
    'PlaceofBirth': nationality,
    'StageID': stage_map[stage],
    'GradeID': 3,
    'SectionID': 1,
    'Topic': 3,  # Exemplo: Matemática
    'Semester': 1,
    'Relation': 1,  # Responsável: pai
    'raisedhands': raisedhands,
    'VisITedResources': resources,
    'AnnouncementsView': announcements,
    'Discussion': discussion,
    'ParentAnsweringSurvey': 1,
    'ParentschoolSatisfaction': 1,
    'StudentAbsenceDays': absence_map[absence]
}])

# Mostrar dados simulados
with st.expander("🔍 Ver dados de entrada do aluno"):
    st.dataframe(entrada)

# Prever e exibir resultado
if st.button("🔮 Prever Risco de Evasão"):
    resultado = modelo.predict(entrada)
    prob = modelo.predict_proba(entrada)[0][1] if hasattr(modelo, "predict_proba") else None
    if resultado[0] == 1:
        st.error(f"⚠️ Previsão: ALTO RISCO de evasão escolar.")
        if prob is not None:
            st.markdown(f"**Probabilidade estimada:** {prob:.2%}")
    else:
        st.success(f"✅ Previsão: Sem risco de evasão.")
        if prob is not None:
            st.markdown(f"**Probabilidade estimada:** {prob:.2%}")
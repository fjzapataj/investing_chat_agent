# app/dashboard.py

import pandas as pd
import streamlit as st
import json
from streamlit_echarts import st_echarts

st.set_page_config(page_title="📊 Dashboard General de Evaluación", layout="wide")
st.title("📈 Resultados de Evaluación")

modo = st.sidebar.selectbox("Selecciona el modo:", ["📊 Dashboard", "📈 Métricas"])

if modo == "📊 Dashboard":
    # Cargar resultados desde el archivo run_eval_results.json
    try:
        with open("src/app/evaluation/run_eval_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        st.error("No se encontró el archivo run_eval_results.json. Asegúrate de ejecutar primero el script de evaluación.")
        st.stop()

    if not results:
        st.warning("El archivo de resultados está vacío.")
        st.stop()

    # Convertir resultados a DataFrame
    data = []
    for item in results:
        data.append({
            "pregunta": item.get("question"),
            "respuesta_referencia": item.get("reference_answer"),
            "respuesta_bot": item.get("bot_answer"),
            "evaluación": item.get("evaluation")
        })

    df = pd.DataFrame(data)

    # Mostrar tabla completa
    st.subheader("📋 Resultados individuales por pregunta")
    st.dataframe(df)

    # Análisis de evaluaciones
    if not df.empty:
        grouped = df["evaluación"].value_counts().reset_index()
        grouped.columns = ["evaluación", "cantidad"]

        st.subheader("📊 Resumen de evaluaciones")
        st.bar_chart(grouped.set_index("evaluación")[["cantidad"]])

        # Calcular eficiencia general
        total = grouped["cantidad"].sum()
        correctas = grouped[grouped["evaluación"] == "GRADE: CORRECT"]["cantidad"].sum()
        eficiencia = (correctas / total) * 100 if total > 0 else 0

        st.subheader("📈 Eficiencia general del modelo")

        # Configuración del gráfico de tacómetro
        options = {
            "series": [
                {
                    "type": "gauge",
                    "startAngle": 180,
                    "endAngle": 0,
                    "min": 0,
                    "max": 100,
                    "splitNumber": 10,
                    "axisLine": {
                        "lineStyle": {
                            "width": 10
                        }
                    },
                    "pointer": {
                        "length": "70%",
                        "width": 5
                    },
                    "progress": {
                        "show": True,
                        "width": 10
                    },
                    "detail": {
                        "valueAnimation": True,
                        "formatter": "{value}%"
                    },
                    "data": [
                        {
                            "value": round(eficiencia, 2)
                        }
                    ]
                }
            ]
        }

        st_echarts(options=options, height="300px")

    else:
        st.warning("No hay datos para analizar.")

    # Detalle adicional
    st.subheader("🔍 Detalles adicionales de las evaluaciones")
    for index, row in df.iterrows():
        with st.expander(f"Pregunta {index + 1}: {row['pregunta']}"):
            st.write(f"**Respuesta esperada:** {row['respuesta_referencia']}")
            st.write(f"**Respuesta del bot:** {row['respuesta_bot']}")
            st.write(f"**Evaluación:** {row['evaluación']}")

elif modo == "📈 Métricas":
    st.title("📈 Resultados de Evaluación")

    # Cargar resultados desde el archivo run_eval_results.json
    try:
        with open("run_eval_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        st.error("No se encontró el archivo run_eval_results.json. Asegúrate de ejecutar primero el script de evaluación.")
        st.stop()

    if not results:
        st.warning("El archivo de resultados está vacío.")
        st.stop()

    # Convertir resultados a DataFrame
    data = []
    for item in results:
        data.append({
            "Pregunta": item.get("question"),
            "Prompt": item.get("reference_answer"),
            "Bot": item.get("bot_answer"),
            "Evaluación": item.get("evaluation")
        })

    df = pd.DataFrame(data)
    st.dataframe(df)

    # Agrupado
    st.subheader("📈 Promedio por configuración")
    grouped = df["Evaluación"].value_counts(normalize=True).reset_index()
    grouped.columns = ["Evaluación", "Precisión"]
    grouped["Precisión"] *= 100

    st.bar_chart(grouped.set_index("Evaluación")[["Precisión"]])
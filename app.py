import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load model dan metadata
model = pickle.load(open("churn_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

#Load dataset untuk distribusi churn
df = pd.read_csv("custome.csv")
churn_counts = df["Churn_Status"].value_counts()

#Judul
st.markdown("<h1 style='text-align: center;'>🔮 Prediksi Customer Churn</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Masukkan informasi pelanggan di bawah untuk mengetahui apakah mereka berpotensi churn atau tidak.</p>", unsafe_allow_html=True)
st.markdown("---")

#Input Data
input_data = []
col1, col2 = st.columns(2)

for i, col in enumerate(features):
    with (col1 if i % 2 == 0 else col2):
        if col in label_encoders:
            options = label_encoders[col].classes_.tolist()
            value = st.selectbox(f"{col}", options)
            encoded = label_encoders[col].transform([value])[0]
            input_data.append(encoded)
        else:
            value = st.number_input(f"{col}", step=1.0)
            input_data.append(value)

#Prediksi
st.markdown("")

if st.button("🚀 Prediksi Churn", use_container_width=True):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("❌ *Customer kemungkinan akan **Churn*** 😟")
    else:
        st.success("✅ *Customer kemungkinan akan **bertahan*** 😄")

# Distribusi Pie Chart
st.markdown("### 📊 Distribusi Churn Status")
labels = ["Tidak Churn", "Churn"]
sizes = [churn_counts[0], churn_counts[1]]
colors = ["#66bb6a", "#ef5350"]
explode = (0, 0.1)

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", colors=colors, shadow=True, startangle=90)
ax.axis("equal")  

st.pyplot(fig)

#Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Dibuat oleh Mubarok982 •2025</p>", unsafe_allow_html=True)

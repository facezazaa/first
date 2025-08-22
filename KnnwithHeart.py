from sklearn.neighbors import KNeighborsRegressor
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายราคาบ้านด้วยเทคนิค K-Nearest Neighbor')

# -------------------------------
# โหลดข้อมูล
# -------------------------------
dt = pd.read_csv("./data/Housing.csv")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# -------------------------------
# สถิติพื้นฐาน
# -------------------------------
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# -------------------------------
# เลือกฟีเจอร์และแสดงกราฟ
# -------------------------------
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[1:])  # ยกเว้น price

# Boxplot: feature vs price
st.write(f"### 🎯 Boxplot: {feature} เทียบกับราคา")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x=feature, y="price", ax=ax)
st.pyplot(fig)

# Pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):

    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue=None)  # ไม่มี HeartDisease แล้ว
    st.pyplot(fig2)

# -------------------------------
# แบบฟอร์มป้อนค่าทำนาย
# -------------------------------
st.subheader("🔮 กรอกข้อมูลบ้านเพื่อทำนายราคา")

A1 = st.number_input("พื้นที่ (area)", value=1000)
A2 = st.number_input("จำนวนห้องนอน (bedrooms)", value=2)
A3 = st.number_input("จำนวนห้องน้ำ (bathrooms)", value=1)
A4 = st.number_input("จำนวนชั้น (stories)", value=1)
A5 = st.selectbox("ติดถนนใหญ่ (mainroad)", ["yes", "no"])
A6 = st.selectbox("มีห้องรับแขก (guestroom)", ["yes", "no"])
A7 = st.selectbox("มีห้องใต้ดิน (basement)", ["yes", "no"])
A8 = st.selectbox("มีระบบน้ำร้อน (hotwaterheating)", ["yes", "no"])
A9 = st.selectbox("มีแอร์ (airconditioning)", ["yes", "no"])
A10 = st.number_input("ที่จอดรถ (parking)", value=1)
A11 = st.selectbox("ใกล้สถานที่สำคัญ (prefarea)", ["yes", "no"])
A12 = st.selectbox("การตกแต่ง (furnishingstatus)", ["furnished", "semi-furnished", "unfurnished"])

# -------------------------------
# ปรับข้อมูลให้อยู่ในรูป numeric
# -------------------------------
def preprocess(df):
    df2 = df.copy()
    for col in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
        df2[col] = df2[col].map({"yes": 1, "no": 0})
    df2 = pd.get_dummies(df2, columns=["furnishingstatus"], drop_first=True)
    return df2

dt_proc = preprocess(dt)

X = dt_proc.drop("price", axis=1)
y = dt_proc["price"]

# -------------------------------
# Train Model
# -------------------------------
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

# -------------------------------
# ทำนายผล
# -------------------------------
if st.button("ทำนายราคา"):
    # เอาค่าจาก input มาใส่ DataFrame 1 แถว
    input_dict = {
        "area": [A1],
        "bedrooms": [A2],
        "bathrooms": [A3],
        "stories": [A4],
        "mainroad": [A5],
        "guestroom": [A6],
        "basement": [A7],
        "hotwaterheating": [A8],
        "airconditioning": [A9],
        "parking": [A10],
        "prefarea": [A11],
        "furnishingstatus": [A12]
    }
    x_input = pd.DataFrame(input_dict)

    # ใช้ preprocess() แบบเดียวกับตอน train
    x_input_proc = preprocess(x_input)

    # ทำให้คอลัมน์ตรงกับ X
    x_input_proc = x_input_proc.reindex(columns=X.columns, fill_value=0)

    # ทำนายราคา
    prediction = model.predict(x_input_proc)[0]
    st.success(f"🏡 ราคาบ้านที่คาดการณ์: {prediction:,.2f} บาท")


else:
    st.write("กรอกข้อมูลแล้วกดปุ่มเพื่อทำนายราคา")

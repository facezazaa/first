from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('🔮 การคาดการณ์อัตราการรอดชีวิตของผู้ป่วยโรคตับแข็งด้วย K-Nearest Neighbor')

# -------------------------------
# โหลดข้อมูล
# -------------------------------
dt = pd.read_csv("./data/cirrhosis.csv")

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
target_col = "Status"   # สมมติคอลัมน์บอกการรอดชีวิตชื่อ Status (0=เสียชีวิต, 1=รอดชีวิต)
feature = st.selectbox("เลือกฟีเจอร์", [c for c in dt.columns if c != target_col])

# Boxplot
st.write(f"### 🎯 Boxplot: {feature} เทียบกับสถานะผู้ป่วย")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x=target_col, y=feature, ax=ax)
st.pyplot(fig)

# Pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue=target_col)
    st.pyplot(fig2)

# -------------------------------
# Preprocess
# -------------------------------
def preprocess(df):
    df2 = df.copy()
    # แปลงข้อมูล yes/no → 1/0 ถ้ามี
    for col in df2.columns:
        if df2[col].dtype == "object":
            df2[col] = df2[col].astype("category").cat.codes
    return df2

dt_proc = preprocess(dt)

X = dt_proc.drop(target_col, axis=1)
y = dt_proc[target_col]

# -------------------------------
# Train Model
# -------------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# -------------------------------
# แบบฟอร์มป้อนค่าทำนาย
# -------------------------------
st.subheader("📝 กรอกข้อมูลผู้ป่วยเพื่อตรวจสอบโอกาสรอดชีวิต")

input_dict = {}
for col in X.columns:
    if dt[col].dtype == "object":
        input_dict[col] = st.selectbox(f"{col}", dt[col].unique())
    else:
        input_dict[col] = st.number_input(f"{col}", value=float(dt[col].mean()))

x_input = pd.DataFrame([input_dict])
x_input_proc = preprocess(x_input)
x_input_proc = x_input_proc.reindex(columns=X.columns, fill_value=0)

# -------------------------------
# ทำนายผล
# -------------------------------
if st.button("ทำนายผล"):
    prediction = model.predict(x_input_proc)[0]
    
    if prediction == 1:   # สมมติ 1 = รอดชีวิต
        st.success("✅ ผู้ป่วยมีโอกาสรอดชีวิตสูง")
        st.image("./img/12.jpg")
    else:
        st.error("⚠️ ผู้ป่วยมีความเสี่ยงสูงต่อการเสียชีวิต")
        st.image("./img/13.jpg")
else:
    st.write("กรอกข้อมูลแล้วกดปุ่มเพื่อทำนายผล")

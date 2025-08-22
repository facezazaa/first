from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cirrhosis Survival Prediction", layout="wide")

st.title('🔮 การคาดการณ์อัตราการรอดชีวิตของผู้ป่วยโรคตับแข็งด้วย K-Nearest Neighbor')

# -------------------------------
# โหลดข้อมูล
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("./data/cirrhosis.csv")

dt = load_data()

st.subheader("👀 ข้อมูลตัวอย่าง")
col1, col2 = st.columns(2)
with col1:
    st.write("ข้อมูลส่วนแรก 10 แถว")
    st.write(dt.head(10))
with col2:
    st.write("ข้อมูลส่วนสุดท้าย 10 แถว")
    st.write(dt.tail(10))

# -------------------------------
# สถิติพื้นฐาน
# -------------------------------
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe(include="all"))

# -------------------------------
# เลือกฟีเจอร์และแสดงกราฟ
# -------------------------------
target_col = "Status"   # สมมติว่าคอลัมน์นี้บอกผลลัพธ์ (0=เสียชีวิต, 1=รอดชีวิต)

if target_col not in dt.columns:
    st.error(f"⚠️ ไม่พบคอลัมน์ '{target_col}' ในไฟล์ CSV ของคุณ กรุณาตรวจสอบอีกครั้ง")
    st.stop()

st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", [c for c in dt.columns if c != target_col])

st.write(f"### 🎯 Boxplot: {feature} เทียบกับสถานะผู้ป่วย")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x=target_col, y=feature, ax=ax)
st.pyplot(fig)

if st.checkbox("✅ แสดง Pairplot (ใช้เวลาประมวลผล)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue=target_col)
    st.pyplot(fig2)

# -------------------------------
# Preprocess
# -------------------------------
def preprocess(df):
    df2 = df.copy()
    # เติมค่า missing
    df2 = df2.fillna(df2.mean(numeric_only=True))
    # แปลงข้อมูล categorical เป็นตัวเลข
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
        default_val = float(dt[col].mean()) if not pd.isna(dt[col].mean()) else 0.0
        input_dict[col] = st.number_input(f"{col}", value=default_val)

x_input = pd.DataFrame([input_dict])
x_input_proc = preprocess(x_input)
x_input_proc = x_input_proc.reindex(columns=X.columns, fill_value=0)

# -------------------------------
# ทำนายผล
# -------------------------------
if st.button("🔍 ทำนายผล"):
    prediction = model.predict(x_input_proc)[0]
    prob = model.predict_proba(x_input_proc)[0]

    if prediction == 1:   # สมมติ 1 = รอดชีวิต
        st.success(f"✅ ผู้ป่วยมีโอกาสรอดชีวิตสูง (ความมั่นใจ {prob[1]*100:.2f}%)")
        st.image("./img/12.jpg", width=300)
    else:
        st.error(f"⚠️ ผู้ป่วยมีความเสี่ยงสูงต่อการเสียชีวิต (ความมั่นใจ {prob[0]*100:.2f}%)")
        st.image("./img/13.jpg", width=300)
else:
    st.info("👉 กรอกข้อมูลแล้วกดปุ่มเพื่อทำนายผล")


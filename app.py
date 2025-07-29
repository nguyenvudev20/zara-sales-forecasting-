import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("📈 Zara Sales Forecasting App")

# --- Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("zara.csv", delimiter=";")
    df = df.dropna(subset=['name', 'description'])
    df = df.drop_duplicates()
    df['Promotion'] = df['Promotion'].str.strip().str.lower()
    df['Seasonal'] = df['Seasonal'].str.strip().str.lower()
    df['Product Category'] = df['Product Category'].str.strip().str.title()
    df['section'] = df['section'].str.strip().str.upper()
    df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
    df = pd.get_dummies(df, columns=['Promotion', 'Seasonal'], drop_first=True)
    return df

df = load_data()

st.subheader("1. Tổng quan dữ liệu")
st.dataframe(df.head())

# --- Trực quan hóa
st.subheader("2. Trực quan hóa dữ liệu")

tab1, tab2, tab3 = st.tabs(["Sales Distribution", "Promotion Impact", "Product Category"])

with tab1:
    fig, ax = plt.subplots()
    sns.histplot(df['Sales Volume'], kde=True, ax=ax)
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    sns.boxplot(x='Promotion_yes', y='Sales Volume', data=df, ax=ax)
    ax.set_xticklabels(['Không khuyến mãi', 'Có khuyến mãi'])
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots()
    df['Product Category'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# --- Mô hình hóa
st.subheader("3. Dự báo doanh số")

X = df[['price', 'Promotion_yes', 'Seasonal_yes']]
y = df['Sales Volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown(f"""
**📉 Mean Squared Error (MSE):** {mse:.4f}  
**📈 R-squared Score (R²):** {r2:.4f}
""")

# Biểu đồ so sánh
st.subheader("4. Biểu đồ: Dự đoán vs Thực tế")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.set_xlabel("Thực tế")
ax.set_ylabel("Dự đoán")
ax.set_title("Thực tế vs Dự đoán doanh số")
ax.plot([-100, 100], [-100, 100], 'r--')
st.pyplot(fig)

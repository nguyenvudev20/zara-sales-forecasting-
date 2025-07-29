import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

st.set_page_config(page_title="ZARA Sales Analysis", layout="wide")

st.title("👗 ZARA Sales Forecast App")

# Load dữ liệu
df = pd.read_csv("zara.csv", delimiter=';')

# Xử lý dữ liệu như trong Colab
df = df.dropna(subset=['name', 'description'])
df = df.drop_duplicates()

df['Promotion'] = df['Promotion'].str.strip().str.lower()
df['Seasonal'] = df['Seasonal'].str.strip().str.lower()
df['Product Category'] = df['Product Category'].str.strip().str.title()
df['section'] = df['section'].str.strip().str.upper()
df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
df['scraped_day'] = df['scraped_at'].dt.date
df['scraped_month'] = df['scraped_at'].dt.month
df['scraped_weekday'] = df['scraped_at'].dt.dayofweek
df = pd.get_dummies(df, columns=['Promotion', 'Seasonal'], drop_first=True)

scaler = StandardScaler()
df[['Sales Volume', 'price']] = scaler.fit_transform(df[['Sales Volume', 'price']])

st.subheader("📊 Phân tích dữ liệu")

tab1, tab2, tab3 = st.tabs(["Histogram", "Boxplot", "Mô hình ML"])

with tab1:
    fig, ax = plt.subplots()
    sns.histplot(df['Sales Volume'], bins=30, kde=True, ax=ax)
    ax.set_title("Phân bố số lượng bán")
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Promotion_yes', y='Sales Volume', data=df, ax=ax2)
    ax2.set_title("Doanh số theo tình trạng khuyến mãi")
    ax2.set_xticklabels(['Không khuyến mãi', 'Có khuyến mãi'])
    st.pyplot(fig2)

with tab3:
    X = df[['price', 'Promotion_yes', 'Seasonal_yes']]
    y = df['Sales Volume']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric("📉 Mean Squared Error", f"{mse:.4f}")
    st.metric("📈 R-squared", f"{r2:.4f}")

    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred, alpha=0.6)
    ax3.plot([-3, 3], [-3, 3], 'r--')
    ax3.set_xlabel("Thực tế")
    ax3.set_ylabel("Dự đoán")
    ax3.set_title("So sánh thực tế và dự đoán")
    st.pyplot(fig3)

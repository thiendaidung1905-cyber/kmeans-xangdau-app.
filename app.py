import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("Ứng dụng Phân cụm K-Means - Xăng Dầu")

# 1. Đường dẫn file dữ liệu
file_path = "nhom1_xangdau_test (1).xlsx"

@st.cache_data
def load_data(path):
    return pd.read_excel(path)

# Thử đọc dữ liệu
try:
    df = load_data(file_path)
    st.success("Đã tải file dữ liệu thành công!")
    
    st.subheader("Thông tin tổng quan dữ liệu")
    st.dataframe(df.head())
    
    # Chọn các cột số để phân cụm
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.write(f"Các cột số được chọn để phân cụm: {numeric_cols}")
    
    if len(numeric_cols) > 0:
        X = df[numeric_cols].dropna()
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Tính WCSS cho Elbow Method
        wcss = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
        # Vẽ biểu đồ Elbow
        st.subheader("Biểu đồ Elbow Method")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, wcss, marker='o', linestyle='--')
        ax.set_title('Phương pháp khuỷu tay (Elbow Method)')
        ax.set_xlabel('Số lượng cụm (k)')
        ax.set_ylabel('WCSS')
        ax.set_xticks(list(k_range))
        ax.grid(True)
        st.pyplot(fig)
        
        # Chọn k tối ưu (ví dụ k=3)
        optimal_k = st.slider("Chọn số lượng cụm (k) tối ưu:", min_value=2, max_value=6, value=3)
        
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        
        X['Cluster'] = clusters
        st.subheader(f"Kết quả phân cụm (k={optimal_k})")
        st.write(X['Cluster'].value_counts())
        st.dataframe(X.head())
        
    else:
        st.warning("Không tìm thấy cột số nào trong file Excel để phân cụm!")

except Exception as e:
    st.error(f"Đã xảy ra lỗi khi đọc file hoặc chạy chương trình: {e}")
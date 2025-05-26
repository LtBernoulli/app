import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plotting)

# Set the page title
st.set_page_config(page_title="PCA Visualization App", layout="centered")

st.title("Principal Component Analysis (PCA) - 3D Visualization")

# Load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    target_column = st.selectbox("Select the target column (optional for color)", df.columns)

    # Separate numerical and categorical
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Handle target
    labels = None
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
        labels = df[target_column]
    elif target_column in categorical_columns:
        categorical_columns.remove(target_column)
        labels = pd.factorize(df[target_column])[0]  # Convert to numbers

    # Standardize numeric features
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(df[numeric_columns]) if numeric_columns else np.array([]).reshape(len(df), 0)

    # One-hot encode categoricals
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_encoded = encoder.fit_transform(df[categorical_columns]) if categorical_columns else np.array([]).reshape(len(df), 0)

    # Combine features
    combined_features = np.hstack([numeric_scaled, categorical_encoded])

    # Apply PCA
    if combined_features.shape[1] >= 3:
        pca = PCA(n_components=3)
    else:
        pca = PCA(n_components=min(2, combined_features.shape[1]))
        
    X_pca = pca.fit_transform(combined_features)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    if X_pca.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='Set1', edgecolor='k', alpha=0.7)
    else:
        ax = fig.add_subplot(111)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', edgecolor='k', alpha=0.7)

    plt.title('PCA Visualization')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    if labels is not None:
        plt.colorbar(sc, label=target_column)

    st.pyplot(fig)

    # Explained variance
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

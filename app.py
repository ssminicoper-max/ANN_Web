import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="🍷 Wine Intelligence Lab", layout="wide")

# ==================================================
# UI STYLE
# ==================================================
st.markdown("""
<style>
.main {background-color:#fdf8f6;}
h1 {color:#6a1b2e;}
h2,h3 {color:#8e2437;}
section[data-testid="stSidebar"] {background-color:#ffffff;}
.stButton>button {
    background-color:#6a1b2e;
    color:white;
    border-radius:8px;
    padding:0.6em 2em;
}
.block-container {padding:2rem 3rem;}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🍷 Wine Intelligence Lab")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])

menu = st.sidebar.radio("🧭 Navigation", [
    "📊 Overview",
    "🧠 Neural Network Lab",
    "🪬 Prediction"
])

# ==================================================
# DATA LOADING (ONLY AFTER UPLOAD)
# ==================================================
if uploaded_file is not None:

    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file, sep=";")
        df["label"] = (df["quality"] >= 6).astype(int)
        return df

    df = load_data(uploaded_file)
    features_all = df.drop(["quality","label"], axis=1).columns.tolist()

else:
    df = None
    features_all = []

# ==================================================
# OVERVIEW
# ==================================================
if menu == "📊 Overview":

    st.title("🍷 Red Wine Quality Intelligence System")

    if df is None:
        st.info("👈 Please upload a CSV file to begin analysis.")
    else:

        col1, col2 = st.columns(2)
        col1.metric("📦 Total Samples", df.shape[0])
        col2.metric("🧪 Total Features", len(features_all))

        st.markdown("---")

        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe())

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🍷 Quality Distribution")
            fig = px.histogram(df, x="quality", color="quality")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🍾 Alcohol vs Quality")
            fig2 = px.scatter(df, x="alcohol", y="quality", color="label")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        st.subheader("🔗 Correlation Matrix")
        fig_corr, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(
            df.corr(),
            annot=True,
            fmt=".2f",
            cmap="RdPu",
            linewidths=0.5,
            ax=ax
        )
        st.pyplot(fig_corr)

# ==================================================
# TRAIN
# ==================================================
elif menu == "🧠 Neural Network Lab":

    st.title("🧠 Neural Network Training Laboratory")

    if df is None:
        st.warning("⚠️ Upload dataset first.")
    else:

        selected_features = st.multiselect(
            "🧪 Select Input Features",
            features_all,
            default=features_all
        )

        test_size = st.slider("📊 Test Size", 0.1, 0.5, 0.3)
        n_hidden = st.selectbox("🧩 Hidden Layers", [1,2,3])
        neurons = st.slider("🔢 Neurons per Layer", 4, 64, 16)
        activation = st.selectbox(
            "📐 Activation Function",
            ["relu","tanh","logistic","identity"]
        )
        learning_rate = st.number_input("📉 Learning Rate", 0.0001, 1.0, 0.01)
        solver = st.selectbox("🚀 Optimizer", ["adam","sgd","lbfgs"])
        epochs = st.slider("⏳ Epochs", 100, 1000, 500)

        st.markdown("---")

        st.subheader("📘 ANN Mathematical Model")
        st.latex(r"z = Wx + b")
        st.latex(r"a = f(z)")
        st.latex(r"E = y_{true} - y_{pred}")
        st.latex(r"W_{new} = W_{old} - \eta \nabla E")

        hidden_structure = tuple([neurons]*n_hidden)

        X = df[selected_features]
        y = df["label"]

        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=test_size,random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_structure,
            activation=activation,
            solver=solver,
            learning_rate_init=learning_rate,
            max_iter=epochs,
            random_state=42
        )

        if st.button("🎯 Train Model"):

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test,y_pred)
            st.success(f"✅ Model Accuracy: {acc:.4f}")

            st.subheader("📄 Classification Report")
            st.text(classification_report(y_test,y_pred))

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🧮 Confusion Matrix")
                cm = confusion_matrix(y_test,y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm,annot=True,fmt="d",cmap="RdPu",ax=ax_cm)
                st.pyplot(fig_cm)

            with col2:
                st.subheader("📉 Loss Curve")
                st.line_chart(model.loss_curve_)

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.features = selected_features

# ==================================================
# PREDICTION
# ==================================================
elif menu == "🪬 Prediction":

    st.title("🪬 Wine Quality Prediction Panel")

    if df is None:
        st.warning("⚠️ Upload dataset first.")
    elif "model" not in st.session_state:
        st.warning("⚠️ Train the model first.")
    else:

        model = st.session_state.model
        scaler = st.session_state.scaler
        selected_features = st.session_state.features

        input_data = []
        col1, col2 = st.columns(2)

        for i, feature in enumerate(selected_features):
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())

            if i % 2 == 0:
                with col1:
                    value = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=float(df[feature].mean())
                    )
            else:
                with col2:
                    value = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=float(df[feature].mean())
                    )

            input_data.append(value)

        if st.button("🍷 Run Prediction"):

            input_array = np.array(input_data).reshape(1,-1)
            input_scaled = scaler.transform(input_array)

            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]

            if prediction == 1:
                st.success("🍷 GOOD QUALITY WINE")
            else:
                st.error("🍷 LOW QUALITY WINE")

            st.write(f"Probability Good: {prob[1]:.4f}")
            st.write(f"Probability Low: {prob[0]:.4f}")
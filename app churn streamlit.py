import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# SESSION STATE INIT
# =====================================================
for key in ["data_ready", "prep_ready", "model_ready"]:
    if key not in st.session_state:
        st.session_state[key] = False

# =====================================================
st.set_page_config("Bank Churn Prediction", layout="wide")
st.title("🏦 Bank Churn Prediction – Multi Model (SMOTE Option)")

menu = st.sidebar.radio(
    "MENU",
    ["Upload Data", "Preprocessing", "Training Model", "Evaluasi Model", "Prediksi"]
)

# =====================================================
# 1. UPLOAD DATA
# =====================================================
if menu == "Upload Data":
    st.header("📂 Upload Dataset")

    file = st.file_uploader("Upload BankChurners.csv", type="csv")

    if file:
        df = pd.read_csv(file)
        st.session_state.df_raw = df
        st.session_state.data_ready = True
        st.session_state.prep_ready = False
        st.session_state.model_ready = False

        st.success("✅ Dataset berhasil dimuat")
        st.dataframe(df.head())

# =====================================================
# 2. PREPROCESSING
# =====================================================
elif menu == "Preprocessing":
    st.header("⚙️ Data Preprocessing")

    if not st.session_state.data_ready:
        st.warning("⚠️ Upload dataset terlebih dahulu")
        st.stop()

    df = st.session_state.df_raw.copy()

    drop_cols = [
        "CLIENTNUM",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df["Attrition_Flag"] = le.fit_transform(df["Attrition_Flag"])

    df = pd.get_dummies(
        df,
        columns=["Education_Level", "Marital_Status", "Income_Category", "Card_Category"],
        drop_first=True
    )

    y = df["Attrition_Flag"].astype(int)
    X = df.drop("Attrition_Flag", axis=1)

    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=np.number).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    df_clean = pd.concat([X, y], axis=1)

    st.session_state.df_clean = df_clean
    st.session_state.scaler = scaler
    st.session_state.prep_ready = True

    st.success("✅ Preprocessing selesai")
    st.dataframe(df_clean.head())

# =====================================================
# 3. TRAINING MODEL (FIXED)
# =====================================================
elif menu == "Training Model":
    st.header("🤖 Training Model")

    if not st.session_state.prep_ready:
        st.warning("⚠️ Lakukan preprocessing terlebih dahulu")
        st.stop()

    df = st.session_state.df_clean
    X = df.drop("Attrition_Flag", axis=1)
    y = df["Attrition_Flag"]

    metode = st.radio(
        "Pilih Metode Training",
        ["Tanpa SMOTE", "Dengan SMOTE"]
    )

    if st.button("🚀 TRAIN MODEL"):
        with st.spinner("Training model sedang berjalan..."):
            # =====================
            # HANDLE SMOTE
            # =====================
            if metode == "Dengan SMOTE":
                try:
                    smote = SMOTE(random_state=42)
                    X_used, y_used = smote.fit_resample(X, y)
                except Exception as e:
                    st.error(f"SMOTE gagal: {e}")
                    st.stop()
            else:
                X_used, y_used = X, y

            X_train, X_test, y_train, y_test = train_test_split(
                X_used, y_used,
                test_size=0.2,
                random_state=42,
                stratify=y_used
            )

            # =====================
            # MODELS (STABIL)
            # =====================
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    use_label_encoder=False,
                    verbosity=0
                ),
                "Deep Learning": MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    max_iter=300,
                    random_state=42
                )
            }

            for model in models.values():
                model.fit(X_train, y_train)

            voting = VotingClassifier(
                estimators=[
                    ("lr", models["Logistic Regression"]),
                    ("rf", models["Random Forest"]),
                    ("dt", models["Decision Tree"])
                ],
                voting="soft"
            )
            voting.fit(X_train, y_train)

            stacking = StackingClassifier(
                estimators=[
                    ("rf", models["Random Forest"]),
                    ("dt", models["Decision Tree"])
                ],
                final_estimator=LogisticRegression()
            )
            stacking.fit(X_train, y_train)

            models["Hybrid Voting"] = voting
            models["Hybrid Stacking"] = stacking

            st.session_state.models = models
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.features = X.columns
            st.session_state.model_ready = True

        st.success(f"✅ Training selesai ({metode})")

# =====================================================
# 4. EVALUASI MODEL
# =====================================================
elif menu == "Evaluasi Model":
    st.header("📊 Evaluasi Model")

    if not st.session_state.model_ready:
        st.warning("⚠️ Model belum dilatih")
        st.stop()

    models = st.session_state.models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    hasil = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        hasil.append([
            name,
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        ])

    df_eval = pd.DataFrame(
        hasil,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    )

    st.dataframe(df_eval)

    fig, ax = plt.subplots()
    sns.barplot(data=df_eval, x="Model", y="Accuracy", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    pilih = st.selectbox("Confusion Matrix", df_eval["Model"])
    cm = confusion_matrix(y_test, models[pilih].predict(X_test))

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    st.pyplot(fig2)

# =====================================================
# 5. PREDIKSI
# =====================================================
elif menu == "Prediksi":
    st.header("🔍 Prediksi Churn Nasabah")

    if not st.session_state.model_ready:
        st.warning("⚠️ Model belum siap")
        st.stop()

    model_name = st.selectbox(
        "Pilih Model",
        list(st.session_state.models.keys())
    )

    input_data = {}
    for col in st.session_state.features:
        input_data[col] = st.number_input(col)

    if st.button("Prediksi"):
        df_input = pd.DataFrame([input_data])
        hasil = st.session_state.models[model_name].predict(df_input)

        if hasil[0] == 1:
            st.success("✅ NASABAH BERTAHAN")
        else:
            st.error("⚠️ NASABAH CHURN")

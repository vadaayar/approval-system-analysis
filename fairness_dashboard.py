import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# -----------------------------
# Page Config & Title
# -----------------------------
st.set_page_config(page_title="Fairness Dashboard", layout="centered")
st.title("🎯 Fairness Dashboard for Model Approval System")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("notebooks/shap_importance_behavior_model.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # Drop Unnamed columns
    return df

df_encoded = load_data()

# -----------------------------
# Target Column Mapping
# -----------------------------
model_column_map = {
    "Biased": "approved_biased",
    "Behavior-Based": "approved_behavior",
    "Fairer": "approved_fairer"
}

# -----------------------------
# Model Selection
# -----------------------------
st.subheader("Select Model to Explain")
selected_model = st.selectbox("Choose model:", list(model_column_map.keys()))
target_col = model_column_map[selected_model]

# -----------------------------
# Explain Model
# -----------------------------
if target_col not in df_encoded.columns:
    st.warning(f"⚠️ Column '{target_col}' not found in dataset.")
else:
    # Drop ID and other target columns from features
    drop_cols = ["ID", "approved_biased", "approved_behavior", "approved_fairer"]
    X = df_encoded.drop(columns=[col for col in drop_cols if col in df_encoded.columns])
    y = df_encoded[target_col]

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Permutation Importance
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()

    # Plotting
    st.subheader(f"📊 Explaining {selected_model} Model")
    st.markdown("🔍 **Top Feature Importances**")
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
    ax.set_title("Permutation Importances (Logistic Regression)")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# -----------------------------
# Summary Tables
# -----------------------------
st.subheader("📊 View Summary Tables")

summary_choice = st.radio("Select Summary View:", ["Approval Summary", "Gender Fairness Summary"])

if summary_choice == "Approval Summary":
    approval_df = pd.read_csv("data/final_approval_summary.csv")
    st.dataframe(approval_df, use_container_width=True)
elif summary_choice == "Gender Fairness Summary":
    gender_df = pd.read_csv("data/gender_fairness_summary.csv")
    st.dataframe(gender_df, use_container_width=True)

# -----------------------------
# Optional Raw Data View
# -----------------------------
st.markdown("---")
if st.checkbox("🔍 Show raw encoded data"):
    st.write(df_encoded)











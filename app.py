# Import Library

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# 1. Page configuration

st.set_page_config(
    page_title="Admission Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Admission Analytics Dashboard")
st.caption(
    "Interactive admissions KPIs, applicant exploration, "
    "and machine-learning model evaluation"
)


# 2. Load and prepare data

@st.cache_data
def load_data():
    data = pd.read_csv(
        Path(__file__).with_name("Admission_Prediction.csv")
    )

    data = data.drop_duplicates().copy()

    data["Admission Status"] = (
        data["Admission Status"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    data["Application_Date"] = pd.to_datetime(
        data["Application_Date"],
        errors="coerce"
    )

    data["Year"] = pd.to_numeric(
        data["Year"],
        errors="coerce"
    )

    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "CSV file not found. Place "
        "'Admission_Prediction.csv' "
        "in the same folder as app.py."
    )
    st.stop()

required_columns = [
    "Admission Status",
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
    "Year",
    "region"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


# 3. Sidebar filters

st.sidebar.header("Dashboard Controls")

available_statuses = sorted(
    df["Admission Status"].dropna().unique().tolist()
)

selected_statuses = st.sidebar.multiselect(
    "Admission status",
    options=available_statuses,
    default=available_statuses
)

available_regions = sorted(
    df["region"].dropna().astype(str).unique().tolist()
)

selected_regions = st.sidebar.multiselect(
    "Region",
    options=available_regions,
    default=available_regions
)

available_years = sorted(
    df["Year"].dropna().unique().tolist()
)

selected_years = st.sidebar.multiselect(
    "Application year",
    options=available_years,
    default=available_years,
    format_func=lambda year: str(int(year))
)

filtered_df = df[
    df["Admission Status"].isin(selected_statuses)
    & df["region"].astype(str).isin(selected_regions)
    & df["Year"].isin(selected_years)
].copy()

st.sidebar.caption(
    f"{len(filtered_df):,} applicant records match the filters."
)


# 4. Navigation

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Admission Trends",
        "Applicant Explorer",
        "Model Performance",
        "Data Quality"
    ]
)


# 5. Overview page

if page == "Overview":
    st.subheader("Admissions Overview")

    total = len(filtered_df)

    accepted = (
        filtered_df["Admission Status"]
        .eq("Accepted")
        .sum()
    )

    rejected = (
        filtered_df["Admission Status"]
        .eq("Rejected")
        .sum()
    )

    acceptance_rate = (
        accepted / total * 100 if total else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Applicants", f"{total:,}")
    col2.metric("Accepted", f"{accepted:,}")
    col3.metric("Rejected", f"{rejected:,}")
    col4.metric("Acceptance Rate", f"{acceptance_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        status_data = (
            filtered_df["Admission Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Applicants")
        )

        fig = px.pie(
            status_data,
            names="Status",
            values="Applicants",
            title="Admission Outcome Distribution",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        score_data = filtered_df.dropna(
            subset=["GRE Score", "CGPA"]
        )

        fig = px.scatter(
            score_data,
            x="GRE Score",
            y="CGPA",
            color="Admission Status",
            hover_data=["University_Name", "region"],
            title="GRE Score vs CGPA"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Applicant Data Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)


# 6. Admission trends page

elif page == "Admission Trends":
    st.subheader("Admission Trends")

    trend_data = (
        filtered_df.dropna(subset=["Year"])
        .groupby(["Year", "Admission Status"])
        .size()
        .reset_index(name="Applicants")
    )

    fig = px.line(
        trend_data,
        x="Year",
        y="Applicants",
        color="Admission Status",
        markers=True,
        title="Admission Outcomes by Year"
    )

    st.plotly_chart(fig, use_container_width=True)

    region_data = (
        filtered_df.dropna(subset=["region"])
        .groupby(["region", "Admission Status"])
        .size()
        .reset_index(name="Applicants")
    )

    fig = px.bar(
        region_data,
        x="region",
        y="Applicants",
        color="Admission Status",
        barmode="group",
        title="Admission Outcomes by Region"
    )

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            filtered_df,
            x="GRE Score",
            color="Admission Status",
            barmode="overlay",
            opacity=0.7,
            title="GRE Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(
            filtered_df,
            x="Admission Status",
            y="CGPA",
            color="Admission Status",
            title="CGPA by Admission Outcome"
        )
        st.plotly_chart(fig, use_container_width=True)


# 7. Applicant explorer page

elif page == "Applicant Explorer":
    st.subheader("Applicant Explorer")

    search_text = st.text_input(
        "Search university name"
    )

    applicant_df = filtered_df.copy()

    if search_text:
        applicant_df = applicant_df[
            applicant_df["University_Name"]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

    st.write(f"Matching records: {len(applicant_df):,}")

    st.dataframe(
        applicant_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = applicant_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Applicant Data (CSV)",
        data=csv_data,
        file_name="applicant_data.csv",
        mime="text/csv"
    )


# 8. Model performance page

elif page == "Model Performance":
    st.subheader("Machine-Learning Model Evaluation")

    st.info(
        "The model uses GRE, TOEFL, university rating, SOP, "
        "LOR, CGPA, and Research. Chance of Admit is excluded "
        "to reduce target leakage."
    )

    features = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR",
        "CGPA",
        "Research"
    ]

    model_df = df[
        df["Admission Status"].isin(["Accepted", "Rejected"])
    ].copy()

    X = model_df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )
    y = model_df["Admission Status"]

    if y.value_counts().min() < 2:
        st.error(
            "Each admission class needs at least two records "
            "to perform a stratified train-test split."
        )
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                class_weight="balanced"
            ))
        ])
    }

    result_rows = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        fitted_models[name] = model

        result_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
            "Precision": precision_score(
                y_test,
                predictions,
                pos_label="Accepted",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                pos_label="Accepted",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                predictions,
                pos_label="Accepted",
                zero_division=0
            )
        })

    results = pd.DataFrame(result_rows).sort_values(
        "F1 Score",
        ascending=False
    )

    best_name = results.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    st.write(f"**Top model by test F1 score:** {best_name}")

    c1, c2, c3, c4 = st.columns(4)

    best_row = results.iloc[0]
    c1.metric("Accuracy", f"{best_row['Accuracy']:.3f}")
    c2.metric("Precision", f"{best_row['Precision']:.3f}")
    c3.metric("Recall", f"{best_row['Recall']:.3f}")
    c4.metric("F1 Score", f"{best_row['F1 Score']:.3f}")

    st.subheader("Model Comparison")
    st.dataframe(
        results.round(3),
        use_container_width=True,
        hide_index=True
    )

    metrics_long = results.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        metrics_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        range_y=[0, 1],
        title="Model Performance Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=["Accepted", "Rejected"]
    )

    fig = px.imshow(
        cm,
        x=["Predicted Accepted", "Predicted Rejected"],
        y=["Actual Accepted", "Actual Rejected"],
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix — {best_name}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Metrics are calculated on a held-out test split. "
        "They are not a guarantee of future performance."
    )

    st.download_button(
        "Download Model Comparison (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="model_performance_results.csv",
        mime="text/csv"
    )

# 9. Data quality page

elif page == "Data Quality":
    st.subheader("Data Quality Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{len(df.columns):,}")
    col3.metric(
        "Duplicate Rows",
        f"{df.duplicated().sum():,}"
    )

    missing_data = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing_data.columns = ["Column", "Missing Values"]

    missing_data = missing_data.sort_values(
        "Missing Values",
        ascending=False
    )

    fig = px.bar(
        missing_data,
        x="Column",
        y="Missing Values",
        title="Missing Values by Column"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Information")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(dtype) for dtype in df.dtypes
            ],
            "Missing Values": [
                int(value) for value in df.isnull().sum()
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    survival_columns = [
        col for col in df.columns
        if "survival" in col.lower()
        or "dropout" in col.lower()
    ]

    if survival_columns:
        st.warning(
            "Survival/dropout fields in this dataset include "
            "simulated proxy data. Do not interpret them as "
            "observed dropout or retention outcomes."
        )

st.sidebar.divider()
st.sidebar.caption("Admission Analytics | Data Exploration & ML")

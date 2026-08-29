# ============================================================
# APARTMENT RENTAL PRICE PREDICTION SYSTEM
# Streamlit Deployment Prototype
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Apartment Rental Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-weight: 700;
    }

    h2, h3 {
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 15px;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.20);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("apartments_prepared.csv")


df = load_data()


# ============================================================
# LOAD DEPLOYMENT MODEL
# ============================================================

@st.cache_resource
def load_deployment_model():

    try:
        return joblib.load(
            "random_forest_rent_model.pkl"
        )

    except Exception:
        return None


deployment_model = load_deployment_model()


# ============================================================
# MODEL FEATURES
# ============================================================

features = [
    "bathrooms",
    "bedrooms",
    "square_feet",
    "latitude",
    "longitude",
    "allows_cats",
    "allows_dogs",
    "has_pool",
    "has_parking",
    "has_fee",
    "has_photo",
    "state"
]

target = "price"


# ============================================================
# LATEST OFFICIAL MODEL RESULTS
# ============================================================

performance_df = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],

    "Training MAE": [
        369.01,
        188.75,
        192.65,
        254.37
    ],

    "Testing MAE": [
        367.70,
        235.97,
        245.24,
        259.31
    ],

    "Training RMSE": [
        599.08,
        328.30,
        285.85,
        396.23
    ],

    "Testing RMSE": [
        590.86,
        438.24,
        412.32,
        428.81
    ],

    "Training R²": [
        0.4816,
        0.8443,
        0.8820,
        0.7732
    ],

    "Testing R²": [
        0.4899,
        0.7194,
        0.7516,
        0.7313
    ]
})


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🏠 Apartment Rental"
)

st.sidebar.caption(
    "Machine Learning Prediction System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Report Navigation",
    [
        "1.0 Business Understanding",
        "2.0 Data Understanding",
        "3.0 Data Preparation",
        "4.0 Model Development",
        "5.0 Model Evaluation",
        "6.0 Deployment"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Target: Monthly Rental Price"
)

st.sidebar.caption(
    "Final Selected Algorithm: Random Forest"
)


# ============================================================
# 1.0 BUSINESS UNDERSTANDING
# ============================================================

if page == "1.0 Business Understanding":

    st.title(
        "1.0 Business Understanding"
    )

    st.write(
        """
        Apartment rental prices can vary significantly because
        properties differ in size, location, facilities and
        listing characteristics. Machine learning can be used
        to analyse these factors and estimate monthly rental prices.
        """
    )

    st.divider()

    st.subheader(
        "Business Problem"
    )

    st.write(
        """
        Rental-price estimation can be challenging because
        multiple apartment characteristics influence the final
        rental price.

        An automated prediction system can support more
        consistent and efficient rental-price estimation based
        on historical apartment listing data.
        """
    )

    st.divider()

    st.subheader(
        "Project Objective"
    )

    st.write(
        """
        The objective of this project is to develop regression
        models that can predict monthly apartment rental prices
        based on apartment characteristics, location and
        available facilities.

        The models are evaluated using MAE, RMSE and R² before
        the best-performing model is selected for deployment.
        """
    )

    st.divider()

    st.subheader(
        "Regression Models Developed"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info(
            """
            **Linear Regression**

            Baseline model
            """
        )

    with col2:
        st.info(
            """
            **Decision Tree**

            Tree-based model
            """
        )

    with col3:
        st.info(
            """
            **Random Forest**

            Ensemble model
            """
        )

    with col4:
        st.info(
            """
            **Gradient Boosting**

            Boosting model
            """
        )
        
# ============================================================
# 2.0 DATA UNDERSTANDING
# ============================================================

elif page == "2.0 Data Understanding":

    st.title(
        "2.0 Data Understanding"
    )

    st.write(
        """
        This section explores the prepared apartment rental
        dataset and identifies important patterns that may
        influence rental prices.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Total Variables",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Predictors Used",
            len(features)
        )

    with col4:

        st.metric(
            "Target Variable",
            "Price"
        )

    st.divider()

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.caption(
        "First 10 observations from the prepared dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Numerical Summary"
    )

    numerical_columns = [
        "bathrooms",
        "bedrooms",
        "price",
        "square_feet",
        "latitude",
        "longitude"
    ]

    numerical_summary = (
        df[numerical_columns]
        .describe()
        .T
    )

    numerical_summary["Median"] = (
        df[numerical_columns]
        .median()
    )

    numerical_summary["Skewness"] = (
        df[numerical_columns]
        .skew()
    )

    st.dataframe(
        numerical_summary.round(2),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # PRICE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "2.1 Rental Price Distribution"
    )

    price_limit = (
        df["price"]
        .quantile(0.99)
    )

    filtered_price = df[
        df["price"] <= price_limit
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        filtered_price["price"],
        bins=50
    )

    ax.set_title(
        "Distribution of Rental Prices up to the 99th Percentile"
    )

    ax.set_xlabel(
        "Monthly Rental Price (USD)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Most apartment listings are concentrated within the
        lower and middle rental-price ranges. The visualisation
        is limited to the 99th percentile so that extreme values
        do not dominate the chart.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # SIZE VS PRICE
    # --------------------------------------------------------

    st.subheader(
        "2.2 Apartment Size vs Rental Price"
    )

    size_limit = (
        df["square_feet"]
        .quantile(0.99)
    )

    scatter_data = df[
        (df["square_feet"] <= size_limit)
        &
        (df["price"] <= price_limit)
    ][
        [
            "square_feet",
            "price"
        ]
    ]

    if len(scatter_data) > 5000:

        scatter_data = scatter_data.sample(
            5000,
            random_state=42
        )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.scatter(
        scatter_data["square_feet"],
        scatter_data["price"],
        alpha=0.35
    )

    ax.set_title(
        "Apartment Size vs Rental Price"
    )

    ax.set_xlabel(
        "Square Feet"
    )

    ax.set_ylabel(
        "Monthly Rental Price (USD)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Apartment size shows a relationship with rental price,
        although size alone does not completely explain price
        differences. Location and facilities also contribute.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # BEDROOMS
    # --------------------------------------------------------

    st.subheader(
        "2.3 Median Rental Price by Bedrooms"
    )

    bedroom_summary = (
        df.groupby("bedrooms")["price"]
        .agg(
            ["count", "median"]
        )
        .reset_index()
    )

    bedroom_plot = bedroom_summary[
        bedroom_summary["count"] >= 500
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        bedroom_plot["bedrooms"]
        .astype(str),

        bedroom_plot["median"]
    )

    ax.set_title(
        "Median Rental Price by Number of Bedrooms"
    )

    ax.set_xlabel(
        "Bedrooms"
    )

    ax.set_ylabel(
        "Median Monthly Rental Price (USD)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Bedroom categories with at least 500 observations are
        displayed to reduce the influence of very small groups.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    st.subheader(
        "2.4 Rental Price Across Five Major States"
    )

    top_states = (
        df["state"]
        .value_counts()
        .head(5)
        .index
    )

    state_price = (
        df[
            df["state"].isin(
                top_states
            )
        ]
        .groupby("state")["price"]
        .median()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    state_price.columns = [
        "State",
        "Median Rental Price"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        state_price["State"],
        state_price["Median Rental Price"]
    )

    ax.set_title(
        "Median Rental Price Across Five Major States"
    )

    ax.set_xlabel(
        "State"
    )

    ax.set_ylabel(
        "Median Monthly Rental Price (USD)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Rental prices differ across states, supporting the
        inclusion of geographical information in the
        prediction model.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader(
        "2.5 Correlation Matrix"
    )

    correlation_variables = [
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude",
        "price"
    ]

    correlation_matrix = (
        df[
            correlation_variables
        ]
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    image = ax.imshow(
        correlation_matrix,
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(
        range(
            len(correlation_variables)
        )
    )

    ax.set_yticks(
        range(
            len(correlation_variables)
        )
    )

    ax.set_xticklabels(
        correlation_variables,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation_variables
    )

    for i in range(
        len(correlation_variables)
    ):

        for j in range(
            len(correlation_variables)
        ):

            ax.text(
                j,
                i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title(
        "Correlation Matrix of Numerical Variables"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# 3.0 DATA PREPARATION
# ============================================================

elif page == "3.0 Data Preparation":

    st.title(
        "3.0 Data Preparation"
    )

    st.write(
        """
        Data preparation was performed to improve data quality
        and prepare the apartment listings for machine learning.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # QUALITY CHECK
    # --------------------------------------------------------

    st.subheader(
        "Prepared Dataset Quality"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Final Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Missing Values",
            int(
                df.isna()
                .sum()
                .sum()
            )
        )

    with col3:

        st.metric(
            "Duplicate Records",
            int(
                df.duplicated()
                .sum()
            )
        )

    st.divider()

    # --------------------------------------------------------
    # PREPARATION STEPS
    # --------------------------------------------------------

    st.subheader(
        "Data Preparation Process"
    )

    preparation_df = pd.DataFrame({

        "Step": [
            "Data Type Conversion",
            "Missing Value Treatment",
            "Target Cleaning",
            "Location Cleaning",
            "Outlier Treatment",
            "Duplicate Removal",
            "Feature Engineering",
            "Categorical Encoding"
        ],

        "Action": [
            "Converted numerical variables to suitable numeric formats.",
            "Handled missing numerical and categorical values.",
            "Removed records without rental price.",
            "Removed observations without valid latitude or longitude.",
            "Treated extreme observations.",
            "Removed duplicate apartment records.",
            "Created binary indicators for listing characteristics.",
            "Applied One-Hot Encoding to State during modelling."
        ]
    })

    st.dataframe(
        preparation_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # VARIABLE TYPES
    # --------------------------------------------------------

    st.subheader(
        "Modelling Variable Types"
    )

    variable_types = pd.DataFrame({

        "Type": [
            "Numerical",
            "Binary",
            "Categorical",
            "Target"
        ],

        "Variables": [
            "bathrooms, bedrooms, square_feet, latitude, longitude",
            "allows_cats, allows_dogs, has_pool, has_parking, has_fee, has_photo",
            "state",
            "price"
        ],

        "Treatment": [
            "Passed directly to the model",
            "Represented as 0 and 1",
            "One-Hot Encoded",
            "Numerical regression target"
        ]
    })

    st.dataframe(
        variable_types,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        """
        **Important:** State remains a string/categorical variable
        in the prepared dataset. It is converted into numerical
        features using One-Hot Encoding before entering the
        regression models.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE AVAILABILITY
    # --------------------------------------------------------

    st.subheader(
        "Availability of Apartment Features"
    )

    binary_features = [
        "allows_cats",
        "allows_dogs",
        "has_pool",
        "has_parking",
        "has_fee"
    ]

    feature_counts = (
        df[binary_features]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    bars = ax.bar(
        feature_counts.index,
        feature_counts.values
    )

    ax.set_title(
        "Availability of Selected Apartment Features"
    )

    ax.set_xlabel(
        "Feature"
    )

    ax.set_ylabel(
        "Number of Listings"
    )

    plt.xticks(
        rotation=25
    )

    for bar in bars:

        value = int(
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value,

            f"{value:,}",

            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# 4.0 MODEL DEVELOPMENT
# ============================================================

elif page == "4.0 Model Development":

    st.title(
        "4.0 Model Development"
    )

    st.write(
        """
        Four regression algorithms were developed using the
        same prepared dataset and the same 80:20 training and
        testing split.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    st.subheader(
        "Training and Testing Data"
    )

    total_records = len(df)

    train_records = int(
        total_records * 0.80
    )

    test_records = (
        total_records
        - train_records
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Records",
            f"{total_records:,}"
        )

    with col2:

        st.metric(
            "Training Data",
            "80%"
        )

    with col3:

        st.metric(
            "Testing Data",
            "20%"
        )

    st.caption(
        """
        All models use random_state = 42 to maintain
        reproducibility of the train-test split.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL 1
    # --------------------------------------------------------

    st.subheader(
        "4.1 Linear Regression"
    )

    st.write(
        """
        Linear Regression was used as the baseline model.
        It provides a simple reference for comparing the
        performance of more complex regression algorithms.
        """
    )

    linear_info = pd.DataFrame({

        "Setting": [
            "Role",
            "Categorical Processing",
            "Model Type"
        ],

        "Value": [
            "Baseline model",
            "One-Hot Encoding for State",
            "Linear Regression"
        ]
    })

    st.dataframe(
        linear_info,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL 2
    # --------------------------------------------------------

    st.subheader(
        "4.2 Decision Tree Regressor"
    )

    st.write(
        """
        The Decision Tree Regressor captures nonlinear
        relationships through a sequence of decision rules.
        GridSearchCV was used to identify an improved
        combination of hyperparameters.
        """
    )

    dt_params = pd.DataFrame({

        "Parameter": [
            "max_depth",
            "min_samples_split",
            "min_samples_leaf"
        ],

        "Selected Value": [
            20,
            20,
            5
        ]
    })

    st.dataframe(
        dt_params,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL 3
    # --------------------------------------------------------

    st.subheader(
        "4.3 Random Forest Regressor"
    )

    st.write(
        """
        Random Forest combines predictions from multiple
        decision trees to improve predictive stability and
        reduce the variance of an individual decision tree.
        GridSearchCV was used for hyperparameter tuning.
        """
    )

    rf_params = pd.DataFrame({

        "Parameter": [
            "n_estimators",
            "max_depth",
            "max_features",
            "min_samples_split",
            "min_samples_leaf"
        ],

        "Selected Value": [
            100,
            20,
            "sqrt",
            2,
            1
        ]
    })

    st.dataframe(
        rf_params,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Best Cross-Validation RMSE: approximately 424.46"
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL 4
    # --------------------------------------------------------

    st.subheader(
        "4.4 Gradient Boosting Regressor"
    )

    st.write(
        """
        Gradient Boosting develops trees sequentially, where
        each new tree attempts to correct prediction errors
        made by previous trees.
        """
    )

    gb_params = pd.DataFrame({

        "Parameter": [
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "subsample"
        ],

        "Selected Value": [
            200,
            0.1,
            4,
            10,
            4,
            0.9
        ]
    })

    st.dataframe(
        gb_params,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# 5.0 MODEL EVALUATION
# ============================================================

elif page == "5.0 Model Evaluation":

    st.title(
        "5.0 Model Evaluation"
    )

    st.write(
        """
        The four regression models are compared using the same
        testing dataset before a final model is selected.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # METRIC EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "Evaluation Metrics"
    )

    metric_df = pd.DataFrame({

        "Metric": [
            "MAE",
            "RMSE",
            "R²"
        ],

        "Meaning": [
            "Average absolute prediction error",
            "Prediction error with greater penalty for large errors",
            "Proportion of rental-price variation explained"
        ],

        "Preferred Result": [
            "Lower",
            "Lower",
            "Higher"
        ]
    })

    st.dataframe(
        metric_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # COMPLETE RESULTS
    # --------------------------------------------------------

    st.subheader(
        "Overall Model Comparison"
    )

    st.dataframe(
        performance_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Official results from the original trained models."
    )

    st.divider()

    # --------------------------------------------------------
    # TEST MAE
    # --------------------------------------------------------

    st.subheader(
        "5.1 Testing MAE"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        performance_df["Model"],
        performance_df["Testing MAE"]
    )

    ax.set_title(
        "Testing MAE Comparison"
    )

    ax.set_ylabel(
        "MAE (USD)"
    )

    plt.xticks(
        rotation=15
    )

    for bar, value in zip(
        bars,
        performance_df["Testing MAE"]
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.2f}",

            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        """
        **Decision Tree achieved the lowest Testing MAE
        of USD 235.97.**

        This indicates the smallest average absolute
        prediction error.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TEST RMSE
    # --------------------------------------------------------

    st.subheader(
        "5.2 Testing RMSE"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        performance_df["Model"],
        performance_df["Testing RMSE"]
    )

    ax.set_title(
        "Testing RMSE Comparison"
    )

    ax.set_ylabel(
        "RMSE (USD)"
    )

    plt.xticks(
        rotation=15
    )

    for bar, value in zip(
        bars,
        performance_df["Testing RMSE"]
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.2f}",

            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        """
        **Random Forest achieved the lowest Testing RMSE
        of USD 412.32.**

        This indicates stronger performance when larger
        prediction errors receive greater penalties.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TEST R2
    # --------------------------------------------------------

    st.subheader(
        "5.3 Testing R²"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        performance_df["Model"],
        performance_df["Testing R²"]
    )

    ax.set_title(
        "Testing R² Comparison"
    )

    ax.set_ylabel(
        "R²"
    )

    plt.xticks(
        rotation=15
    )

    for bar, value in zip(
        bars,
        performance_df["Testing R²"]
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"{value:.4f}",

            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        """
        **Random Forest achieved the highest Testing R²
        of 0.7516.**

        The model explains approximately **75.16% of the
        variation in apartment rental prices**.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TRAIN TEST MAE
    # --------------------------------------------------------

    st.subheader(
        "5.4 Training vs Testing MAE"
    )

    x = np.arange(
        len(performance_df)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        x - width / 2,
        performance_df["Training MAE"],
        width,
        label="Training MAE"
    )

    ax.bar(
        x + width / 2,
        performance_df["Testing MAE"],
        width,
        label="Testing MAE"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        performance_df["Model"],
        rotation=15
    )

    ax.set_ylabel(
        "MAE (USD)"
    )

    ax.set_title(
        "Training and Testing MAE"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    st.subheader(
        "5.5 Best Model Selection"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Lowest Testing MAE",
            "235.97"
        )

        st.caption(
            "Decision Tree"
        )

    with col2:

        st.metric(
            "Lowest Testing RMSE",
            "412.32"
        )

        st.caption(
            "Random Forest"
        )

    with col3:

        st.metric(
            "Highest Testing R²",
            "0.7516"
        )

        st.caption(
            "Random Forest"
        )

    st.success(
        "🏆 Selected Best Model: Random Forest Regressor"
    )

    st.write(
        """
        Although Decision Tree achieved the lowest MAE,
        Random Forest produced the lowest RMSE and highest R².

        Therefore, considering the overall testing performance,
        Random Forest was selected as the strongest model for
        apartment rental-price prediction.
        """
    )


# ============================================================
# 6.0 DEPLOYMENT
# ============================================================

elif page == "6.0 Deployment":

    st.title(
        "6.0 Deployment"
    )

    st.write(
        """
        The selected Random Forest algorithm is implemented
        through an interactive Streamlit prototype that allows
        users to enter apartment information and obtain an
        estimated monthly rental price.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DEPLOYMENT INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Deployment Tool"
    )

    st.write(
        """
        **Streamlit** was selected because it integrates directly
        with Python machine-learning models and allows an
        interactive web application to be created without
        requiring complex front-end development.
        """
    )

    st.info(
        """
        The official model evaluation uses the original Random
        Forest results shown in Section 5.0.

        A lighter Random Forest model is used for the online
        prediction function because the original model file is
        too large for convenient online deployment.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # USER GUIDE
    # --------------------------------------------------------

    st.subheader(
        "How to Use the Predictor"
    )

    st.write(
        """
        **Step 1:** Enter the apartment characteristics.

        **Step 2:** Select the relevant facilities and state.

        **Step 3:** Click **Predict Rental Price**.

        **Step 4:** View the estimated monthly rental price.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    st.subheader(
        "Apartment Information"
    )

    col1, col2 = st.columns(2)

    # LEFT COLUMN
    with col1:

        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Total number of bathrooms."
        )

        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Total number of bedrooms."
        )

        square_feet = st.number_input(
            "Apartment Size (Square Feet)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=50,
            help="Total apartment floor area."
        )

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=40.7128,
            format="%.6f",
            help="North-south geographical position."
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
            format="%.6f",
            help="East-west geographical position."
        )

        state_list = sorted(
            df["state"]
            .dropna()
            .astype(str)
            .unique()
        )

        state = st.selectbox(
            "State",
            state_list
        )

    # RIGHT COLUMN
    with col2:

        has_photo_option = st.selectbox(
            "Listing Has Photo",
            ["Yes", "No"]
        )

        has_photo = (
            1
            if has_photo_option == "Yes"
            else 0
        )

        allows_cats_option = st.selectbox(
            "Cats Allowed",
            ["Yes", "No"]
        )

        allows_cats = (
            1
            if allows_cats_option == "Yes"
            else 0
        )

        allows_dogs_option = st.selectbox(
            "Dogs Allowed",
            ["Yes", "No"]
        )

        allows_dogs = (
            1
            if allows_dogs_option == "Yes"
            else 0
        )

        has_pool_option = st.selectbox(
            "Swimming Pool Available",
            ["Yes", "No"]
        )

        has_pool = (
            1
            if has_pool_option == "Yes"
            else 0
        )

        has_parking_option = st.selectbox(
            "Parking Available",
            ["Yes", "No"]
        )

        has_parking = (
            1
            if has_parking_option == "Yes"
            else 0
        )

        has_fee_option = st.selectbox(
            "Rental / Application Fee",
            ["Yes", "No"]
        )

        has_fee = (
            1
            if has_fee_option == "Yes"
            else 0
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if deployment_model is None:

        st.error(
            """
            The deployment model could not be loaded.
            Please check random_forest_rent_model.pkl.
            """
        )

    else:

        if st.button(
            "Predict Rental Price",
            type="primary",
            use_container_width=True
        ):

            input_data = pd.DataFrame({

                "bathrooms":
                    [bathrooms],

                "bedrooms":
                    [bedrooms],

                "square_feet":
                    [square_feet],

                "latitude":
                    [latitude],

                "longitude":
                    [longitude],

                "allows_cats":
                    [allows_cats],

                "allows_dogs":
                    [allows_dogs],

                "has_pool":
                    [has_pool],

                "has_parking":
                    [has_parking],

                "has_fee":
                    [has_fee],

                "has_photo":
                    [has_photo],

                "state":
                    [state]
            })

            try:

                prediction = (
                    deployment_model
                    .predict(
                        input_data
                    )
                )

                predicted_price = (
                    prediction[0]
                )

                st.success(
                    "Prediction completed successfully."
                )

                st.metric(
                    "Estimated Monthly Rental Price",
                    f"${predicted_price:,.2f}"
                )

                st.caption(
                    """
                    The prediction is generated using the
                    deployed Random Forest model.
                    """
                )

            except Exception as e:

                st.error(
                    "Unable to generate the prediction."
                )

                st.code(
                    str(e)
                )

    st.divider()

    # --------------------------------------------------------
    # PREDICTOR GUIDE
    # --------------------------------------------------------

    st.subheader(
        "Predictor Guide"
    )

    predictor_guide = pd.DataFrame({

        "Predictor": [
            "Bathrooms",
            "Bedrooms",
            "Square Feet",
            "Latitude",
            "Longitude",
            "Cats Allowed",
            "Dogs Allowed",
            "Swimming Pool",
            "Parking",
            "Fee",
            "Listing Photo",
            "State"
        ],

        "Description": [
            "Number of bathrooms",
            "Number of bedrooms",
            "Apartment floor area",
            "North-south geographical position",
            "East-west geographical position",
            "Whether cats are permitted",
            "Whether dogs are permitted",
            "Whether a pool is available",
            "Whether parking is available",
            "Whether a rental/application fee applies",
            "Whether the listing contains a photo",
            "State where the apartment is located"
        ]
    })

    st.dataframe(
        predictor_guide,
        hide_index=True,
        use_container_width=True
    )
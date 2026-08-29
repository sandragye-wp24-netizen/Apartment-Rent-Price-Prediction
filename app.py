# ============================================================
# APARTMENT RENTAL PRICE PREDICTION SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Apartment Rental Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Sidebar width */
    [data-testid="stSidebar"] {
        min-width: 310px;
        max-width: 310px;
        width: 310px;
        border-right: 1px solid rgba(128, 128, 128, 0.20);
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 310px;
    }

    /* Sidebar navigation */
    [data-testid="stSidebar"] .stRadio label {
        padding-top: 5px;
        padding-bottom: 5px;
        font-size: 16px;
    }

    /* Main headings */
    h1 {
        font-weight: 700;
    }

    h2, h3 {
        font-weight: 600;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 16px;
        border-radius: 10px;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("apartments_prepared.csv")


try:
    df = load_data()

except Exception as e:
    st.error("Unable to load apartments_prepared.csv.")
    st.code(str(e))
    st.stop()


# ============================================================
# 4. LOAD DEPLOYMENT MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("random_forest_rent_model.pkl")


try:
    deployment_model = load_model()

except Exception as e:
    deployment_model = None


# ============================================================
# 5. MODEL INPUT FEATURES
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
# 6. LATEST MODEL PERFORMANCE
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
# 7. SIDEBAR
# ============================================================

st.sidebar.title("🏠 Apartment Rental")

st.sidebar.caption(
    "Rental Price Prediction System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Model Exploration",
        "Model Performance",
        "Batch Prediction",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Target: Monthly Rental Price"
)


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "Prediction":

    st.title("🔮 Apartment Rental Price Prediction")

    st.write(
        """
        Enter the apartment information below to estimate its
        monthly rental price.
        """
    )

    st.info(
        """
        **How to Use**

        1. Enter the apartment characteristics.
        2. Select the relevant facilities and state.
        3. Click **Predict Rental Price**.
        4. View the estimated monthly rental price.
        """
    )

    st.divider()

    st.subheader("Apartment Information")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

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
            help="Apartment floor area in square feet."
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
            state_list,
            help="State where the apartment is located."
        )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        has_photo_option = st.selectbox(
            "Listing Has Photo",
            ["Yes", "No"]
        )

        has_photo = (
            1 if has_photo_option == "Yes" else 0
        )

        allows_cats_option = st.selectbox(
            "Cats Allowed",
            ["Yes", "No"]
        )

        allows_cats = (
            1 if allows_cats_option == "Yes" else 0
        )

        allows_dogs_option = st.selectbox(
            "Dogs Allowed",
            ["Yes", "No"]
        )

        allows_dogs = (
            1 if allows_dogs_option == "Yes" else 0
        )

        has_pool_option = st.selectbox(
            "Swimming Pool Available",
            ["Yes", "No"]
        )

        has_pool = (
            1 if has_pool_option == "Yes" else 0
        )

        has_parking_option = st.selectbox(
            "Parking Available",
            ["Yes", "No"]
        )

        has_parking = (
            1 if has_parking_option == "Yes" else 0
        )

        has_fee_option = st.selectbox(
            "Rental / Application Fee",
            ["Yes", "No"]
        )

        has_fee = (
            1 if has_fee_option == "Yes" else 0
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if deployment_model is None:

        st.error(
            "The prediction model could not be loaded."
        )

        st.caption(
            "Please check random_forest_rent_model.pkl."
        )

    else:

        if st.button(
            "Predict Rental Price",
            type="primary",
            use_container_width=True
        ):

            input_data = pd.DataFrame({

                "bathrooms": [bathrooms],
                "bedrooms": [bedrooms],
                "square_feet": [square_feet],
                "latitude": [latitude],
                "longitude": [longitude],
                "allows_cats": [allows_cats],
                "allows_dogs": [allows_dogs],
                "has_pool": [has_pool],
                "has_parking": [has_parking],
                "has_fee": [has_fee],
                "has_photo": [has_photo],
                "state": [state]
            })

            try:

                prediction = deployment_model.predict(
                    input_data
                )

                predicted_price = prediction[0]

                st.success(
                    "Prediction completed successfully."
                )

                col1, col2, col3 = st.columns(
                    [1, 2, 1]
                )

                with col2:

                    st.metric(
                        "Estimated Monthly Rental Price",
                        f"${predicted_price:,.2f}"
                    )

            except Exception as e:

                st.error(
                    "Unable to generate the prediction."
                )

                st.code(str(e))

    st.divider()

    # --------------------------------------------------------
    # INPUT GUIDE
    # --------------------------------------------------------

    st.subheader("Predictor Guide")

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
            "Whether a swimming pool is available",
            "Whether parking is available",
            "Whether a rental or application fee applies",
            "Whether the listing contains a photo",
            "State where the apartment is located"
        ]
    })

    st.dataframe(
        predictor_guide,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PAGE 2 — MODEL EXPLORATION
# ============================================================

elif page == "Model Exploration":

    st.title("📊 Model Exploration")

    st.write(
        """
        This section explores the apartment rental dataset and
        the variables used for model development.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------------

    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Variables",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Model Predictors",
            len(features)
        )

    with col4:

        st.metric(
            "Target",
            "Price"
        )

    st.divider()

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------------

    st.subheader("Numerical Summary")

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
    # RENTAL PRICE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Rental Price Distribution"
    )

    price_limit = (
        df["price"]
        .quantile(0.99)
    )

    price_data = df[
        df["price"] <= price_limit
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        price_data["price"],
        bins=50
    )

    ax.set_title(
        "Distribution of Rental Prices up to the 99th Percentile"
    )

    ax.set_xlabel(
        "Monthly Rental Price (USD)"
    )

    ax.set_ylabel(
        "Number of Listings"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        """
        The 99th percentile is used to prevent extreme rental
        values from dominating the visualisation.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # SIZE VS PRICE
    # --------------------------------------------------------

    st.subheader(
        "Apartment Size vs Rental Price"
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

        scatter_data = (
            scatter_data
            .sample(
                5000,
                random_state=42
            )
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
        "Apartment Size vs Monthly Rental Price"
    )

    ax.set_xlabel(
        "Apartment Size (Square Feet)"
    )

    ax.set_ylabel(
        "Monthly Rental Price (USD)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Apartment size has a relationship with rental price,
        although rental price is also influenced by other
        apartment and location characteristics.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # BEDROOMS VS PRICE
    # --------------------------------------------------------

    st.subheader(
        "Median Rental Price by Number of Bedrooms"
    )

    bedroom_summary = (
        df.groupby("bedrooms")["price"]
        .agg(
            count="count",
            median="median"
        )
        .reset_index()
    )

    bedroom_summary = bedroom_summary[
        bedroom_summary["count"] >= 500
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    bars = ax.bar(
        bedroom_summary[
            "bedrooms"
        ].astype(str),

        bedroom_summary[
            "median"
        ]
    )

    ax.set_title(
        "Median Rental Price by Number of Bedrooms"
    )

    ax.set_xlabel(
        "Number of Bedrooms"
    )

    ax.set_ylabel(
        "Median Monthly Rental Price (USD)"
    )

    for bar, value in zip(
        bars,
        bedroom_summary["median"]
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"${value:,.0f}",

            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    st.caption(
        """
        Only bedroom categories with at least 500 listings
        are included.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # STATES
    # --------------------------------------------------------

    st.subheader(
        "Rental Price Across Five Major States"
    )

    major_states = (
        df["state"]
        .value_counts()
        .head(5)
        .index
    )

    state_price = (
        df[
            df["state"].isin(
                major_states
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

    bars = ax.bar(
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

    for bar, value in zip(
        bars,
        state_price["Median Rental Price"]
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height(),

            f"${value:,.0f}",

            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Differences in median rental prices across states
        indicate that geographical location is an important
        predictor of apartment rental price.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader(
        "Correlation Matrix"
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
            len(
                correlation_variables
            )
        )
    )

    ax.set_yticks(
        range(
            len(
                correlation_variables
            )
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
        len(
            correlation_variables
        )
    ):

        for j in range(
            len(
                correlation_variables
            )
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
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        """
        Four regression algorithms were evaluated using the
        same testing dataset. Performance was assessed using
        MAE, RMSE and R².
        """
    )

    st.divider()

    # --------------------------------------------------------
    # METRIC GUIDE
    # --------------------------------------------------------

    st.subheader(
        "Evaluation Metrics"
    )

    metric_guide = pd.DataFrame({

        "Metric": [
            "MAE",
            "RMSE",
            "R²"
        ],

        "Meaning": [
            "Average absolute difference between actual and predicted rent",
            "Prediction error that gives greater weight to large errors",
            "Proportion of variation in rental price explained by the model"
        ],

        "Preferred": [
            "Lower",
            "Lower",
            "Higher"
        ]
    })

    st.dataframe(
        metric_guide,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    st.subheader(
        "Model Comparison"
    )

    st.dataframe(
        performance_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    st.subheader(
        "Testing MAE Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        performance_df["Model"],
        performance_df["Testing MAE"]
    )

    ax.set_title(
        "Testing MAE by Regression Model"
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
        **Decision Tree achieved the lowest Testing MAE:
        USD 235.97.**

        This means it produced the smallest average absolute
        prediction error.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    st.subheader(
        "Testing RMSE Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        performance_df["Model"],
        performance_df["Testing RMSE"]
    )

    ax.set_title(
        "Testing RMSE by Regression Model"
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
        **Random Forest achieved the lowest Testing RMSE:
        USD 412.32.**

        This indicates stronger control of larger
        prediction errors.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    st.subheader(
        "Testing R² Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        performance_df["Model"],
        performance_df["Testing R²"]
    )

    ax.set_title(
        "Testing R² by Regression Model"
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
        **Random Forest achieved the highest Testing R²:
        0.7516.**

        This means approximately **75.16% of the variation
        in apartment rental prices** was explained by the
        model.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TRAIN VS TEST MAE
    # --------------------------------------------------------

    st.subheader(
        "Training vs Testing MAE"
    )

    x = np.arange(
        len(
            performance_df
        )
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

    ax.set_title(
        "Training and Testing MAE"
    )

    ax.set_ylabel(
        "MAE (USD)"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Comparing training and testing errors helps evaluate
        how well each model generalises to unseen data.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    st.subheader(
        "Best Model Selection"
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
        Decision Tree achieved the lowest MAE. However,
        Random Forest achieved both the **lowest RMSE**
        and the **highest R²**.

        Considering the overall testing performance,
        Random Forest was therefore selected as the
        strongest regression model.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # RANDOM FOREST DETAILS
    # --------------------------------------------------------

    st.subheader(
        "Selected Random Forest Performance"
    )

    rf_performance = pd.DataFrame({

        "Metric": [
            "MAE",
            "RMSE",
            "R²"
        ],

        "Training": [
            192.65,
            285.85,
            0.8820
        ],

        "Testing": [
            245.24,
            412.32,
            0.7516
        ]
    })

    st.dataframe(
        rf_performance,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "Selected Hyperparameters"
    )

    rf_parameters = pd.DataFrame({

        "Parameter": [
            "n_estimators",
            "max_depth",
            "max_features",
            "min_samples_split",
            "min_samples_leaf"
        ],

        "Value": [
            100,
            20,
            "sqrt",
            2,
            1
        ]
    })

    st.dataframe(
        rf_parameters,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PAGE 4 — BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    st.title("📁 Batch Prediction")

    st.write(
        """
        Upload a CSV file containing multiple apartment records
        to generate rental-price predictions for all records at once.
        """
    )

    st.divider()

    st.subheader(
        "Required Input Columns"
    )

    required_columns_df = pd.DataFrame({
        "Required Column": features
    })

    st.dataframe(
        required_columns_df,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        """
        Binary variables should use **1 = Yes** and **0 = No**.

        The `state` column should contain the same type of
        state values used in the prepared dataset.
        """
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Apartment CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.subheader(
                "Uploaded Data Preview"
            )

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )

            st.write(
                f"Uploaded records: **{len(batch_df):,}**"
            )

            missing_columns = [
                col
                for col in features
                if col not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded file is missing required columns:"
                )

                st.write(
                    ", ".join(
                        missing_columns
                    )
                )

            elif deployment_model is None:

                st.error(
                    "The prediction model could not be loaded."
                )

            else:

                st.success(
                    "All required predictor columns were found."
                )

                if st.button(
                    "Generate Batch Predictions",
                    type="primary",
                    use_container_width=True
                ):

                    try:

                        prediction_data = (
                            batch_df[
                                features
                            ]
                            .copy()
                        )

                        predictions = (
                            deployment_model
                            .predict(
                                prediction_data
                            )
                        )

                        results_df = (
                            batch_df.copy()
                        )

                        results_df[
                            "predicted_monthly_rent"
                        ] = predictions

                        st.success(
                            f"""
                            Predictions generated successfully
                            for {len(results_df):,} apartment records.
                            """
                        )

                        st.subheader(
                            "Prediction Results"
                        )

                        st.dataframe(
                            results_df,
                            use_container_width=True
                        )

                        # ------------------------------------
                        # SUMMARY
                        # ------------------------------------

                        st.subheader(
                            "Batch Prediction Summary"
                        )

                        col1, col2, col3 = (
                            st.columns(3)
                        )

                        with col1:

                            st.metric(
                                "Predicted Records",
                                f"{len(results_df):,}"
                            )

                        with col2:

                            st.metric(
                                "Average Predicted Rent",
                                f"${results_df['predicted_monthly_rent'].mean():,.2f}"
                            )

                        with col3:

                            st.metric(
                                "Median Predicted Rent",
                                f"${results_df['predicted_monthly_rent'].median():,.2f}"
                            )

                        # ------------------------------------
                        # DOWNLOAD
                        # ------------------------------------

                        csv_output = (
                            results_df
                            .to_csv(
                                index=False
                            )
                            .encode(
                                "utf-8"
                            )
                        )

                        st.download_button(
                            label="Download Prediction Results",
                            data=csv_output,
                            file_name="apartment_rental_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception as e:

                        st.error(
                            "Unable to generate batch predictions."
                        )

                        st.code(
                            str(e)
                        )

        except Exception as e:

            st.error(
                "Unable to read the uploaded CSV file."
            )

            st.code(
                str(e)
            )


# ============================================================
# PAGE 5 — ABOUT
# ============================================================

elif page == "About":

    st.title("ℹ️ About")

    st.write(
        """
        This application was developed as a machine-learning
        prototype for apartment rental-price prediction.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.subheader(
        "Project Objective"
    )

    st.write(
        """
        The objective is to predict monthly apartment rental
        prices using apartment characteristics, location and
        available facilities.

        Four regression algorithms were developed and evaluated
        before the strongest overall model was selected for
        deployment.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader(
        "Dataset Information"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Prepared Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Predictors Used",
            len(features)
        )

    with col3:

        st.metric(
            "Target Variable",
            "Price"
        )

    st.divider()

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    st.subheader(
        "Data Preparation Summary"
    )

    preparation_df = pd.DataFrame({

        "Process": [
            "Data Type Conversion",
            "Missing Value Treatment",
            "Target Cleaning",
            "Location Cleaning",
            "Outlier Treatment",
            "Duplicate Removal",
            "Binary Feature Creation",
            "Categorical Encoding"
        ],

        "Description": [
            "Converted numerical variables to suitable numeric formats.",
            "Handled missing numerical and categorical values.",
            "Removed records without rental price.",
            "Removed observations without usable location coordinates.",
            "Treated extreme observations.",
            "Removed duplicate apartment records.",
            "Represented apartment facilities using binary variables.",
            "State is One-Hot Encoded during model preprocessing."
        ]
    })

    st.dataframe(
        preparation_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Models Developed"
    )

    model_df = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Gradient Boosting Regressor"
        ],

        "Purpose": [
            "Baseline regression model",
            "Nonlinear tree-based regression",
            "Ensemble of multiple decision trees",
            "Sequential boosting regression"
        ]
    })

    st.dataframe(
        model_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    st.subheader(
        "Final Model"
    )

    st.success(
        "Selected Model: Random Forest Regressor"
    )

    st.write(
        """
        Random Forest was selected because it achieved the
        **lowest Testing RMSE of 412.32** and the
        **highest Testing R² of 0.7516** among the four
        evaluated models.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # DEPLOYMENT
    # --------------------------------------------------------

    st.subheader(
        "Deployment"
    )

    st.write(
        """
        Streamlit is used to provide an interactive interface
        for individual and batch apartment rental-price predictions.
        """
    )

    st.info(
        """
        The official model evaluation is based on the original
        trained Random Forest model.

        A lighter Random Forest model is used for live online
        prediction because the original saved model file is
        considerably larger and less convenient for online
        deployment.
        """
    )
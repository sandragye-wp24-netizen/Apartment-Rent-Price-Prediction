# ============================================================
# APARTMENT RENTAL PRICE PREDICTION SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split


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

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    [data-testid="stSidebar"] {
        min-width: 310px;
        max-width: 310px;
        width: 310px;
        border-right: 1px solid rgba(128, 128, 128, 0.20);
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 310px;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding-top: 5px;
        padding-bottom: 5px;
        font-size: 16px;
    }

    h1 {
        font-weight: 700;
    }

    h2, h3 {
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 16px;
        border-radius: 10px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. LOAD DATASET
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

except Exception:
    deployment_model = None


# ============================================================
# 5. MODEL VARIABLES
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

binary_columns = [
    "allows_cats",
    "allows_dogs",
    "has_pool",
    "has_parking",
    "has_fee",
    "has_photo"
]

valid_states = set(
    df["state"]
    .dropna()
    .astype(str)
    .unique()
)


# ============================================================
# 6. OFFICIAL MODEL PERFORMANCE
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

st.sidebar.title(
    "🏠 Apartment Rental"
)

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

    st.title(
        "🔮 Apartment Rental Price Prediction"
    )

    st.write(
        """
        Enter the apartment information below to estimate
        its monthly rental price.
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

    st.subheader(
        "Apartment Information"
    )

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

        state_list = sorted(valid_states)

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
    # SINGLE PREDICTION
    # --------------------------------------------------------

    if deployment_model is None:

        st.error(
            "The prediction model could not be loaded."
        )

        st.write(
            """
            Please ensure that
            `random_forest_rent_model.pkl`
            is available.
            """
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

                result_col1, result_col2, result_col3 = (
                    st.columns([1, 2, 1])
                )

                with result_col2:

                    st.metric(
                        "Estimated Monthly Rental Price",
                        f"${predicted_price:,.2f}"
                    )

                st.write(
                    """
                    The estimated monthly rent is based on
                    the apartment characteristics entered above.
                    """
                )

            except Exception as e:

                st.error(
                    "Unable to generate the prediction."
                )

                st.code(str(e))


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

    st.title(
        "📊 Model Exploration"
    )

    st.write(
        """
        This section explores important characteristics of the
        apartment rental dataset and evaluates whether selected
        variables may provide useful information for rental-price
        prediction.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Dataset Summary"
    )

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
            "Predictors",
            len(features)
        )

    with col4:
        st.metric(
            "Target",
            "Price"
        )


    st.divider()


    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
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


    # ========================================================
    # GRAPH 1 — RENTAL PRICE DISTRIBUTION
    # ========================================================

    st.subheader(
        "1. Rental Price Distribution"
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

    plt.close(fig)

    st.write(
        """
        This graph is used to understand the distribution of the
        target variable, monthly rental price. Examining the target
        distribution is important because extreme or highly skewed
        rental values may affect regression model performance.

        The graph is limited to the 99th percentile so that a small
        number of unusually high rental prices do not dominate the
        visualisation. This provides a clearer representation of the
        rental-price range covering most apartment listings.
        """
    )


    st.divider()


    # ========================================================
    # GRAPH 2 — SIZE VS PRICE
    # ========================================================

    st.subheader(
        "2. Apartment Size vs Rental Price"
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

    plt.close(fig)

    st.write(
        """
        This scatter plot is used to examine whether apartment size
        is associated with rental price and therefore provides useful
        predictive information.

        Square footage represents the physical size of an apartment
        and may contribute to differences in rental value. Both rental
        price and apartment size are limited to the 99th percentile
        to reduce the visual influence of extreme observations and
        make the main relationship easier to identify.
        """
    )


    st.divider()


    # ========================================================
    # GRAPH 3 — BEDROOMS
    # ========================================================

    st.subheader(
        "3. Median Rental Price by Number of Bedrooms"
    )

    bedroom_summary = (
        df
        .groupby("bedrooms")["price"]
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
        bedroom_summary["bedrooms"].astype(str),
        bedroom_summary["median"]
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
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${value:,.0f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        This graph is used to investigate how rental prices differ
        across bedroom categories. The number of bedrooms represents
        apartment capacity and is an important property characteristic
        that may influence rental value.

        Median rental price is used instead of the mean because it is
        less affected by unusually expensive listings. Only bedroom
        categories containing at least 500 listings are included so
        that comparisons are based on sufficiently represented groups.
        """
    )


    st.divider()


    # ========================================================
    # GRAPH 4 — STATE
    # ========================================================

    st.subheader(
        "4. Rental Price Across Five Major States"
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
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${value:,.0f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        This graph is used to examine whether geographical location
        contributes to differences in apartment rental prices.
        Location is an important pricing factor because rental markets
        vary between states.

        The five states with the largest number of listings are used
        so that comparisons are based on well-represented groups.
        Median rental price is used to reduce the influence of unusually
        expensive properties within each state.
        """
    )


    st.divider()


    # ========================================================
    # GRAPH 5 — BINARY FEATURES
    # ========================================================

    st.subheader(
        "5. Rental Price by Apartment Features"
    )

    binary_feature_labels = {
        "allows_cats": "Cats Allowed",
        "allows_dogs": "Dogs Allowed",
        "has_pool": "Swimming Pool",
        "has_parking": "Parking",
        "has_fee": "Fee Required",
        "has_photo": "Listing Photo"
    }

    feature_price_data = []

    for column, label in binary_feature_labels.items():

        median_without = (
            df.loc[
                df[column] == 0,
                "price"
            ]
            .median()
        )

        median_with = (
            df.loc[
                df[column] == 1,
                "price"
            ]
            .median()
        )

        feature_price_data.append({
            "Feature": label,
            "Without Feature": median_without,
            "With Feature": median_with
        })

    feature_price_df = pd.DataFrame(
        feature_price_data
    )

    x = np.arange(
        len(feature_price_df)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        x - width / 2,
        feature_price_df["Without Feature"],
        width,
        label="No"
    )

    ax.bar(
        x + width / 2,
        feature_price_df["With Feature"],
        width,
        label="Yes"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        feature_price_df["Feature"],
        rotation=20
    )

    ax.set_ylabel(
        "Median Monthly Rental Price (USD)"
    )

    ax.set_title(
        "Median Rental Price With and Without Apartment Features"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        This graph is used to investigate whether apartment facilities
        and listing characteristics are associated with differences
        in rental prices.

        Binary variables such as pet permission, swimming pool,
        parking, fees and listing photos may provide additional
        predictive information beyond apartment size and location.
        Comparing median rental prices between listings with and without
        each feature helps assess whether these variables may contribute
        useful information to the regression models.
        """
    )


    st.divider()


    # ========================================================
    # GRAPH 6 — CORRELATION MATRIX
    # ========================================================

    st.subheader(
        "6. Correlation Matrix"
    )

    correlation_variables = [
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude",
        "price"
    ]

    correlation_matrix = df[
        correlation_variables
    ].corr()

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
        range(len(correlation_variables))
    )

    ax.set_yticks(
        range(len(correlation_variables))
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

            value = correlation_matrix.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title(
        "Correlation Matrix of Numerical Variables"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation Coefficient"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        This correlation matrix is used to examine the strength and
        direction of relationships among numerical predictors and
        monthly rental price.

        The analysis helps identify variables that may provide useful
        predictive information and also reveals relationships between
        the predictors themselves. Understanding these relationships
        supports feature assessment and helps determine whether a
        simple linear model is sufficient to represent the structure
        of the data.
        """
    )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "📈 Model Performance"
    )

    st.write(
        """
        Four regression algorithms are compared using the same
        testing dataset. Their predictive performance is evaluated
        using MAE, RMSE and R².
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

        "Preferred Result": [
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
    # MODEL COMPARISON
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
    # TESTING MAE
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
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        MAE measures the average absolute prediction error.
        Decision Tree records the lowest Testing MAE of
        **USD 235.97**.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # TESTING RMSE
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
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        RMSE gives greater weight to larger prediction errors.
        Random Forest achieves the lowest Testing RMSE of
        **USD 412.32**.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # TESTING R²
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
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        R² measures how much variation in rental price is explained
        by the model. Random Forest obtains the highest Testing R²
        of **0.7516**, corresponding to approximately **75.16%**
        of the variation in rental prices.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # TRAINING VS TESTING
    # --------------------------------------------------------

    st.subheader(
        "Training vs Testing MAE"
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

    ax.set_title(
        "Training and Testing MAE"
    )

    ax.set_ylabel(
        "MAE (USD)"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        """
        Comparing training and testing errors helps assess model
        generalisation. A substantially lower training error than
        testing error may indicate that a model has learned the
        training data more closely than unseen observations.
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

        st.write(
            "**Decision Tree**"
        )

    with col2:

        st.metric(
            "Lowest Testing RMSE",
            "412.32"
        )

        st.write(
            "**Random Forest**"
        )

    with col3:

        st.metric(
            "Highest Testing R²",
            "0.7516"
        )

        st.write(
            "**Random Forest**"
        )

    st.success(
        "🏆 Selected Model: Random Forest Regressor"
    )

    st.write(
        """
        Decision Tree achieved the lowest MAE, while Random Forest
        achieved the lowest RMSE and highest R². Considering the
        overall testing performance, Random Forest was selected for
        deployment.
        """
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    if deployment_model is not None:

        st.divider()

        st.subheader(
            "Actual vs Predicted Rental Price"
        )

        try:

            X_diag = df[
                features
            ].copy()

            y_diag = df[
                target
            ].copy()

            (
                X_train_diag,
                X_test_diag,
                y_train_diag,
                y_test_diag
            ) = train_test_split(
                X_diag,
                y_diag,
                test_size=0.20,
                random_state=42
            )

            deployment_test_pred = (
                deployment_model.predict(
                    X_test_diag
                )
            )

            diagnostic_df = pd.DataFrame({

                "Actual":
                    y_test_diag.values,

                "Predicted":
                    deployment_test_pred
            })

            if len(
                diagnostic_df
            ) > 4000:

                diagnostic_plot = (
                    diagnostic_df.sample(
                        4000,
                        random_state=42
                    )
                )

            else:

                diagnostic_plot = (
                    diagnostic_df
                )

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            ax.scatter(
                diagnostic_plot["Actual"],
                diagnostic_plot["Predicted"],
                alpha=0.35
            )

            minimum_value = min(
                diagnostic_plot["Actual"].min(),
                diagnostic_plot["Predicted"].min()
            )

            maximum_value = max(
                diagnostic_plot["Actual"].max(),
                diagnostic_plot["Predicted"].max()
            )

            ax.plot(
                [
                    minimum_value,
                    maximum_value
                ],
                [
                    minimum_value,
                    maximum_value
                ],
                linestyle="--"
            )

            ax.set_title(
                "Actual vs Predicted Rental Prices"
            )

            ax.set_xlabel(
                "Actual Rental Price (USD)"
            )

            ax.set_ylabel(
                "Predicted Rental Price (USD)"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.write(
                """
                This graph is used to visually assess prediction
                accuracy. Points located closer to the diagonal
                reference line represent predictions that are
                closer to the actual rental values.

                This diagnostic is generated from the deployed
                Random Forest model.
                """
            )

        except Exception:

            st.info(
                """
                Actual-versus-predicted analysis is unavailable
                for the current deployment model.
                """
            )


        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        st.divider()

        st.subheader(
            "Random Forest Feature Importance"
        )

        try:

            if hasattr(
                deployment_model,
                "named_steps"
            ):

                pipeline_steps = (
                    deployment_model.named_steps
                )

                preprocessor = (
                    pipeline_steps["preprocessor"]
                )

                if "regressor" in pipeline_steps:

                    rf_estimator = (
                        pipeline_steps["regressor"]
                    )

                elif "model" in pipeline_steps:

                    rf_estimator = (
                        pipeline_steps["model"]
                    )

                else:

                    rf_estimator = None

                if (
                    rf_estimator is not None
                    and
                    hasattr(
                        rf_estimator,
                        "feature_importances_"
                    )
                ):

                    try:

                        processed_names = (
                            preprocessor
                            .get_feature_names_out()
                        )

                    except Exception:

                        processed_names = np.array([
                            f"Feature {i + 1}"
                            for i in range(
                                len(
                                    rf_estimator
                                    .feature_importances_
                                )
                            )
                        ])

                    importance_df = pd.DataFrame({

                        "Feature":
                            processed_names,

                        "Importance":
                            rf_estimator
                            .feature_importances_
                    })

                    importance_df[
                        "Feature"
                    ] = (
                        importance_df["Feature"]
                        .astype(str)
                        .str.replace(
                            "cat__",
                            "",
                            regex=False
                        )
                        .str.replace(
                            "remainder__",
                            "",
                            regex=False
                        )
                    )

                    importance_df = (
                        importance_df
                        .sort_values(
                            "Importance",
                            ascending=False
                        )
                        .head(15)
                        .sort_values(
                            "Importance",
                            ascending=True
                        )
                    )

                    fig, ax = plt.subplots(
                        figsize=(9, 6)
                    )

                    ax.barh(
                        importance_df["Feature"],
                        importance_df["Importance"]
                    )

                    ax.set_title(
                        "Top Random Forest Feature Importances"
                    )

                    ax.set_xlabel(
                        "Feature Importance"
                    )

                    ax.set_ylabel(
                        "Feature"
                    )

                    plt.tight_layout()

                    st.pyplot(fig)

                    plt.close(fig)

                    st.write(
                        """
                        Feature importance is used to identify
                        predictors that contribute more strongly
                        to the Random Forest prediction process.

                        This analysis improves model interpretability
                        by indicating which apartment characteristics,
                        location indicators and facilities have greater
                        influence on predicted rental prices.
                        """
                    )

                else:

                    st.info(
                        "Feature importance is unavailable."
                    )

            else:

                st.info(
                    "Feature importance is unavailable."
                )

        except Exception:

            st.info(
                """
                Feature importance could not be extracted from
                the current deployment model.
                """
            )


    st.divider()


    # ========================================================
    # MODEL ADVANTAGES AND LIMITATIONS
    # ========================================================

    st.subheader(
        "Model Advantages and Limitations"
    )


    # --------------------------------------------------------
    # LINEAR REGRESSION
    # --------------------------------------------------------

    st.markdown(
        "### Linear Regression"
    )

    linear_model_info = pd.DataFrame({

        "Advantages": [
            "Simple and easy to understand.",
            "Fast to train and computationally efficient.",
            "Useful as a baseline for evaluating more complex models."
        ],

        "Limitations": [
            "Assumes a linear relationship between predictors and rental price.",
            "May not capture complex interactions between apartment characteristics.",
            "Predictive performance may be weaker when relationships are nonlinear."
        ]
    })

    st.dataframe(
        linear_model_info,
        hide_index=True,
        use_container_width=True
    )


    st.divider()


    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    st.markdown(
        "### Decision Tree Regressor"
    )

    decision_tree_info = pd.DataFrame({

        "Advantages": [
            "Can capture nonlinear relationships.",
            "Relatively easy to interpret.",
            "Does not require a linear relationship between predictors and target."
        ],

        "Limitations": [
            "Can overfit the training dataset.",
            "Small changes in data can produce different tree structures.",
            "Predictions may be less stable than ensemble methods."
        ]
    })

    st.dataframe(
        decision_tree_info,
        hide_index=True,
        use_container_width=True
    )


    st.divider()


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    st.markdown(
        "### Random Forest Regressor"
    )

    random_forest_info = pd.DataFrame({

        "Advantages": [
            "Captures complex and nonlinear relationships.",
            "Combines multiple trees to produce more stable predictions.",
            "Reduces variance compared with a single Decision Tree.",
            "Provides feature-importance information."
        ],

        "Limitations": [
            "Requires more computational resources than simpler models.",
            "Saved model files can become large because many trees are stored.",
            "Less interpretable than Linear Regression or a single Decision Tree.",
            "Training and hyperparameter tuning may take longer."
        ]
    })

    st.dataframe(
        random_forest_info,
        hide_index=True,
        use_container_width=True
    )


    st.divider()


    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    st.markdown(
        "### Gradient Boosting Regressor"
    )

    gradient_boosting_info = pd.DataFrame({

        "Advantages": [
            "Can model complex nonlinear relationships.",
            "Sequentially improves predictions by correcting previous errors.",
            "Can achieve strong predictive performance.",
            "Provides flexibility through several tuning parameters."
        ],

        "Limitations": [
            "Can require more training time than simpler models.",
            "Performance is sensitive to hyperparameter settings.",
            "Sequential training is less parallelisable than Random Forest.",
            "More difficult to interpret than simpler regression models."
        ]
    })

    st.dataframe(
        gradient_boosting_info,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PAGE 4 — BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    st.title(
        "📁 Batch Prediction"
    )

    st.write(
        """
        Upload a CSV file containing multiple apartment records
        to generate rental-price predictions for several apartments
        at the same time.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # INPUT FORMAT
    # --------------------------------------------------------

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


    st.write(
        """
        **Required column order:**

        `bathrooms, bedrooms, square_feet, latitude, longitude, allows_cats, allows_dogs, has_pool, has_parking, has_fee, has_photo, state`
        """
    )


    st.info(
        """
        **CSV Input Format**

        For the binary columns below, use:

        - `1` = Yes
        - `0` = No

        Binary columns:

        `allows_cats`, `allows_dogs`, `has_pool`,
        `has_parking`, `has_fee`, `has_photo`

        For the `state` column, use the same state abbreviation
        format as the prepared dataset, such as:

        `CA`, `TX`, `FL`, `NY`

        Example row:

        `1, 2, 900, 34.0522, -118.2437, 1, 1, 0, 1, 0, 1, CA`
        """
    )


    st.divider()


    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Apartment CSV File",
        type=["csv"]
    )


    if uploaded_file is not None:

              try:
          batch_df = pd.read_csv(
         uploaded_file,
        encoding="utf-8"
        )

    except UnicodeDecodeError:

    uploaded_file.seek(0)

    try:
        batch_df = pd.read_csv(
            uploaded_file,
            encoding="cp1252"
        )

    except UnicodeDecodeError:

        uploaded_file.seek(0)

        batch_df = pd.read_csv(
            uploaded_file,
            encoding="latin-1"
        )

            st.subheader(
                "Uploaded Data Preview"
            )

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )


            # =================================================
            # CHECK REQUIRED COLUMNS
            # =================================================

            missing_columns = [
                column
                for column in features
                if column not in batch_df.columns
            ]

            extra_columns = [
                column
                for column in batch_df.columns
                if column not in features
            ]


            # -------------------------------------------------
            # IF REQUIRED COLUMNS ARE MISSING
            # -------------------------------------------------

            if missing_columns:

                st.error(
                    """
                    Prediction cannot continue because the uploaded
                    file is missing required predictor columns.
                    """
                )

                st.write(
                    "**Missing required columns:**"
                )

                st.write(
                    ", ".join(
                        missing_columns
                    )
                )


            # -------------------------------------------------
            # IF ALL REQUIRED COLUMNS EXIST
            # -------------------------------------------------

            else:

                prediction_input = (
                    batch_df[
                        features
                    ]
                    .copy()
                )


                # =============================================
                # DATA QUALITY CHECKS
                # =============================================

                total_records = len(
                    batch_df
                )

                duplicate_mask = (
                    prediction_input
                    .duplicated(
                        keep=False
                    )
                )

                duplicate_records = int(
                    duplicate_mask.sum()
                )


                missing_value_mask = (
                    prediction_input[
                        features
                    ]
                    .isna()
                    .any(
                        axis=1
                    )
                )

                missing_value_records = int(
                    missing_value_mask.sum()
                )


                # ---------------------------------------------
                # INVALID BINARY VALUES
                # ---------------------------------------------

                invalid_binary_mask = (
                    pd.Series(
                        False,
                        index=batch_df.index
                    )
                )

                for column in binary_columns:

                    invalid_binary_mask = (
                        invalid_binary_mask
                        |
                        (
                            prediction_input[column]
                            .notna()
                            &
                            ~prediction_input[column]
                            .isin(
                                [0, 1]
                            )
                        )
                    )

                invalid_binary_records = int(
                    invalid_binary_mask.sum()
                )


                # ---------------------------------------------
                # INVALID STATE VALUES
                # ---------------------------------------------

                state_as_string = (
                    prediction_input[
                        "state"
                    ]
                    .astype(str)
                )

                invalid_state_mask = (
                    prediction_input[
                        "state"
                    ]
                    .notna()
                    &
                    ~state_as_string.isin(
                        valid_states
                    )
                )

                invalid_state_records = int(
                    invalid_state_mask.sum()
                )


                # ---------------------------------------------
                # VALID ROWS
                # ---------------------------------------------

                invalid_row_mask = (
                    missing_value_mask
                    |
                    invalid_binary_mask
                    |
                    invalid_state_mask
                )

                valid_row_mask = (
                    ~invalid_row_mask
                )

                valid_records = int(
                    valid_row_mask.sum()
                )


                # =================================================
                # DATA QUALITY SUMMARY
                # =================================================

                st.divider()

                st.subheader(
                    "Data Quality Summary"
                )

                quality_col1, quality_col2, quality_col3 = (
                    st.columns(3)
                )

                with quality_col1:

                    st.metric(
                        "Uploaded Records",
                        f"{total_records:,}"
                    )

                    st.metric(
                        "Extra Columns",
                        len(
                            extra_columns
                        )
                    )


                with quality_col2:

                    st.metric(
                        "Duplicate Records",
                        f"{duplicate_records:,}"
                    )

                    st.metric(
                        "Rows with Missing Values",
                        f"{missing_value_records:,}"
                    )


                with quality_col3:

                    st.metric(
                        "Invalid Binary Rows",
                        f"{invalid_binary_records:,}"
                    )

                    st.metric(
                        "Valid Rows",
                        f"{valid_records:,}"
                    )


                if invalid_state_records > 0:

                    st.warning(
                        f"""
                        {invalid_state_records:,} row(s) contain
                        state values that are not available in the
                        prepared dataset.
                        """
                    )


                # ---------------------------------------------
                # EXTRA COLUMNS
                # ---------------------------------------------

                if extra_columns:

                    st.info(
                        """
                        Extra columns were detected. They will remain
                        in the output file but will not be used by the
                        prediction model.
                        """
                    )

                    st.write(
                        "**Extra columns:**"
                    )

                    st.write(
                        ", ".join(
                            extra_columns
                        )
                    )


                # ---------------------------------------------
                # DUPLICATES
                # ---------------------------------------------

                if duplicate_records > 0:

                    st.info(
                        """
                        Duplicate predictor records were detected.
                        They are retained because separate apartment
                        records may legitimately contain identical
                        predictor values.
                        """
                    )


                # ---------------------------------------------
                # MISSING VALUES
                # ---------------------------------------------

                if missing_value_records > 0:

                    st.warning(
                        """
                        Rows containing missing required predictor
                        values will not be sent to the prediction model.
                        """
                    )


                # ---------------------------------------------
                # INVALID BINARY VALUES
                # ---------------------------------------------

                if invalid_binary_records > 0:

                    st.warning(
                        """
                        Some binary predictor values are invalid.
                        Binary variables must contain only 0 or 1.
                        These rows will not be predicted.
                        """
                    )


                # =================================================
                # PREDICTION
                # =================================================

                if deployment_model is None:

                    st.error(
                        """
                        The prediction model could not be loaded.
                        """
                    )

                elif valid_records == 0:

                    st.error(
                        """
                        No valid rows are available for prediction.
                        Please correct the uploaded data.
                        """
                    )

                else:

                    if st.button(
                        "Generate Batch Predictions",
                        type="primary",
                        use_container_width=True
                    ):

                        try:

                            results_df = (
                                batch_df.copy()
                            )

                            results_df[
                                "predicted_monthly_rent"
                            ] = np.nan

                            results_df[
                                "prediction_status"
                            ] = "Predicted"


                            # ---------------------------------
                            # SET INVALID STATUS
                            # ---------------------------------

                            results_df.loc[
                                missing_value_mask,
                                "prediction_status"
                            ] = (
                                "Missing required value"
                            )


                            results_df.loc[
                                invalid_binary_mask,
                                "prediction_status"
                            ] = (
                                "Invalid binary value"
                            )


                            results_df.loc[
                                invalid_state_mask,
                                "prediction_status"
                            ] = (
                                "Invalid state value"
                            )


                            # ---------------------------------
                            # COMBINED INVALID CONDITIONS
                            # ---------------------------------

                            multiple_issue_mask = (
                                (
                                    missing_value_mask.astype(int)
                                    +
                                    invalid_binary_mask.astype(int)
                                    +
                                    invalid_state_mask.astype(int)
                                )
                                > 1
                            )

                            results_df.loc[
                                multiple_issue_mask,
                                "prediction_status"
                            ] = (
                                "Multiple data quality issues"
                            )


                            # ---------------------------------
                            # PREDICT VALID ROWS ONLY
                            # ---------------------------------

                            valid_prediction_data = (
                                prediction_input.loc[
                                    valid_row_mask,
                                    features
                                ]
                                .copy()
                            )

                            predictions = (
                                deployment_model
                                .predict(
                                    valid_prediction_data
                                )
                            )

                            results_df.loc[
                                valid_row_mask,
                                "predicted_monthly_rent"
                            ] = predictions


                            st.success(
                                f"""
                                Predictions generated successfully
                                for {valid_records:,} valid
                                apartment record(s).
                                """
                            )


                            # =================================================
                            # PREDICTION RESULTS
                            # =================================================

                            st.subheader(
                                "Prediction Results"
                            )

                            st.dataframe(
                                results_df,
                                use_container_width=True
                            )


                            st.divider()


                            # =================================================
                            # PREDICTION SUMMARY
                            # =================================================

                            st.subheader(
                                "Prediction Summary"
                            )

                            predicted_values = (
                                results_df.loc[
                                    valid_row_mask,
                                    "predicted_monthly_rent"
                                ]
                            )


                            summary_col1, summary_col2, summary_col3 = (
                                st.columns(3)
                            )

                            with summary_col1:

                                st.metric(
                                    "Predicted Records",
                                    f"{valid_records:,}"
                                )


                            with summary_col2:

                                st.metric(
                                    "Average Predicted Rent",
                                    f"${predicted_values.mean():,.2f}"
                                )


                            with summary_col3:

                                st.metric(
                                    "Median Predicted Rent",
                                    f"${predicted_values.median():,.2f}"
                                )


                            # =================================================
                            # DOWNLOAD
                            # =================================================

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
                                label=
                                    "Download Prediction Results",

                                data=
                                    csv_output,

                                file_name=
                                    "apartment_rental_predictions.csv",

                                mime=
                                    "text/csv",

                                use_container_width=True
                            )


                        except Exception as e:

                            st.error(
                                """
                                Unable to generate batch predictions.
                                """
                            )

                            st.code(
                                str(e)
                            )


        except Exception as e:

            st.error(
                """
                Unable to read the uploaded CSV file.
                """
            )

            st.code(
                str(e)
            )


# ============================================================
# PAGE 5 — ABOUT
# ============================================================

elif page == "About":

    st.title(
        "ℹ️ About the Study"
    )

    st.write(
        """
        This study investigates the application of machine learning
        techniques to apartment rental-price estimation using
        historical property listing data.

        The analysis examines how physical property characteristics,
        geographical information, listing attributes and apartment
        facilities contribute to differences in monthly rental prices.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # RESEARCH PURPOSE
    # --------------------------------------------------------

    st.subheader(
        "Research Purpose"
    )

    st.write(
        """
        The research evaluates the predictive capability of several
        regression techniques for apartment rental-price estimation.

        Linear Regression, Decision Tree, Random Forest and Gradient
        Boosting are compared using MAE, RMSE and R² to investigate
        differences in predictive accuracy, error behaviour and
        generalisation performance.

        The comparison also provides insight into whether nonlinear
        and ensemble-learning approaches can capture rental-price
        relationships more effectively than a traditional linear
        baseline model.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # BUSINESS PURPOSE
    # --------------------------------------------------------

    st.subheader(
        "Business Purpose"
    )

    st.write(
        """
        Apartment rental prices can vary substantially because
        properties differ in size, room configuration, geographical
        location, pet policies, facilities and listing characteristics.

        A machine-learning-based rental-price estimation system can
        support rental platforms, property owners and property managers
        by providing a more consistent and efficient method of estimating
        rental prices from historical listing information.

        For rental platforms, automated rental-price estimation may
        reduce the time required to manually assess listings and support
        more consistent pricing recommendations across a large number
        of properties.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # STUDY DATASET
    # --------------------------------------------------------

    st.subheader(
        "Study Dataset"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prepared Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Predictors",
            len(features)
        )

    with col3:

        st.metric(
            "Target",
            "Monthly Rental Price"
        )


    st.write(
        """
        The modelling dataset contains apartment characteristics,
        geographical information and property-related indicators.
        These variables are analysed to understand their relationship
        with monthly rental prices and to develop predictive models.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # VARIABLES USED
    # --------------------------------------------------------

    st.subheader(
        "Variables Used in Modelling"
    )

    variable_df = pd.DataFrame({

        "Variable Type": [
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
            "Used as numerical predictors",
            "Represented using 0 and 1",
            "One-Hot Encoded during model preprocessing",
            "Monthly rental price to be predicted"
        ]
    })

    st.dataframe(
        variable_df,
        hide_index=True,
        use_container_width=True
    )


    st.divider()


    # --------------------------------------------------------
    # REGRESSION METHODS
    # --------------------------------------------------------

    st.subheader(
        "Regression Techniques Investigated"
    )

    models_df = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Gradient Boosting Regressor"
        ],

        "Role in the Study": [
            "Provides a baseline for evaluating more complex models.",
            "Examines nonlinear relationships using decision rules.",
            "Examines ensemble learning through multiple decision trees.",
            "Examines sequential boosting for improved predictive performance."
        ]
    })

    st.dataframe(
        models_df,
        hide_index=True,
        use_container_width=True
    )